import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ninecoder.agent import AgentConfig, CodingAgent
from ninecoder.model_client import ModelResponse, ToolCall
from ninecoder.permissions import PermissionMode
from ninecoder.persistence import atomic_write_text
from ninecoder.session import SessionState, SessionStore, build_session_tree
from ninecoder.ui.base import AgentUI
from ninecoder.workspace import Workspace


class FinishModel:
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        return ModelResponse("done", [ToolCall("call_1", "finish", {"summary": "ok"})])


class ResumeModel:
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        return ModelResponse(
            "write",
            [ToolCall("call_2", "write_file", {"path": "resumed.txt", "content": "yes"})],
        )


class RecordingFinishModel:
    def __init__(self) -> None:
        self.user_messages: list[str] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.user_messages = [
            str(message.get("content", "")) for message in messages if message["role"] == "user"
        ]
        return ModelResponse("done", [ToolCall("call_1", "finish", {"summary": "ok"})])


class RecordingUI(AgentUI):
    def __init__(self) -> None:
        super().__init__()
        self.histories: list[list[dict[str, Any]]] = []
        self.user_messages: list[str] = []

    def session_history(self, messages: list[dict[str, Any]]) -> None:
        self.histories.append(messages)

    def user_message(self, text: str) -> None:
        self.user_messages.append(text)


class SubagentTaskModel:
    def __init__(self) -> None:
        self.agent_calls = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        if not tools:
            return ModelResponse("subagent result")
        self.agent_calls += 1
        if self.agent_calls == 1:
            return ModelResponse(
                "ask reviewer",
                [
                    ToolCall(
                        "call_1",
                        "start_subagent_task",
                        {"role": "reviewer", "prompt": "review"},
                    )
                ],
            )
        return ModelResponse("finish", [ToolCall("call_2", "finish", {"summary": "done"})])


class SessionTest(unittest.TestCase):
    def test_agent_persists_session_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(tmp)
            result = CodingAgent(
                FinishModel(),
                workspace,
                AgentConfig(permission_mode=PermissionMode.AUTO),
            ).run("finish now")

            state = SessionStore(Path(tmp) / "runs" / "sessions").load(result.session_id)

            self.assertEqual(state.status, "finished")
            self.assertEqual(state.summary, "ok")

    def test_agent_can_resume_saved_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "runs" / "sessions")
            state = store.create(
                "resume task",
                tmp,
                PermissionMode.AUTO.value,
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "task"},
                ],
                session_id="saved-session",
            )
            state.status = "running"
            store.save(state)

            result = CodingAgent(
                ResumeModel(),
                Workspace(root),
                AgentConfig(
                    max_steps=1,
                    permission_mode=PermissionMode.AUTO,
                    resume_session="saved-session",
                ),
            ).run("")

            self.assertEqual(result.session_id, "saved-session")
            self.assertEqual((root / "resumed.txt").read_text(encoding="utf-8"), "yes")

    def test_agent_can_open_saved_session_and_continue_same_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "runs" / "sessions")
            store.create(
                "resume task",
                tmp,
                PermissionMode.AUTO.value,
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "first turn"},
                ],
                session_id="saved-session",
            )
            model = RecordingFinishModel()
            agent = CodingAgent(
                model,
                Workspace(root),
                AgentConfig(
                    max_steps=1,
                    permission_mode=PermissionMode.AUTO,
                    resume_session="saved-session",
                    memory=False,
                ),
            )

            opened = agent.open_session()
            result = agent.continue_turn("second turn")

            self.assertEqual(opened.id, "saved-session")
            self.assertEqual(result.session_id, "saved-session")
            self.assertIn("second turn", model.user_messages)
            state = store.load("saved-session")
            user_messages = [
                message["content"] for message in state.messages if message["role"] == "user"
            ]
            self.assertIn("second turn", user_messages)

    def test_agent_displays_history_when_resuming_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "runs" / "sessions")
            store.create(
                "resume task",
                tmp,
                PermissionMode.AUTO.value,
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "first turn"},
                    {"role": "assistant", "content": "first answer"},
                ],
                session_id="saved-session",
            )
            ui = RecordingUI()

            CodingAgent(
                FinishModel(),
                Workspace(root),
                AgentConfig(
                    max_steps=1,
                    permission_mode=PermissionMode.AUTO,
                    resume_session="saved-session",
                    memory=False,
                ),
                ui=ui,
            ).run("")

            self.assertEqual(len(ui.histories), 1)
            self.assertEqual(ui.histories[0][1]["content"], "first turn")
            self.assertEqual(ui.histories[0][2]["content"], "first answer")
            self.assertEqual(ui.user_messages, [])

    def test_agent_persists_subagent_tasks_in_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = CodingAgent(
                SubagentTaskModel(),
                Workspace(tmp),
                AgentConfig(permission_mode=PermissionMode.AUTO),
            ).run("ask reviewer")

            state = SessionStore(Path(tmp) / "runs" / "sessions").load(result.session_id)

            self.assertEqual(state.subagent_tasks[0]["result"], "subagent result")


class SessionTreeTest(unittest.TestCase):
    def test_parent_id_roundtrips_through_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / "runs" / "sessions")
            store.create(
                "task",
                tmp,
                PermissionMode.AUTO.value,
                [{"role": "user", "content": "hi"}],
                session_id="child",
                parent_id="parent-1",
            )
            self.assertEqual(store.load("child").parent_id, "parent-1")

    def test_list_returns_saved_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / "runs" / "sessions")
            store.create("a", tmp, PermissionMode.AUTO.value, [], session_id="s-1")
            store.create("b", tmp, PermissionMode.AUTO.value, [], session_id="s-2")
            self.assertEqual(sorted(s.id for s in store.list()), ["s-1", "s-2"])

    def test_list_skips_malformed_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / "runs" / "sessions")
            store.create("a", tmp, PermissionMode.AUTO.value, [], session_id="s-1")
            (Path(tmp) / "runs" / "sessions" / "s-2").mkdir(parents=True)
            (Path(tmp) / "runs" / "sessions" / "s-2" / "session.json").write_text(
                "{not json", encoding="utf-8"
            )
            self.assertEqual([s.id for s in store.list()], ["s-1"])

    def test_build_session_tree_roots_and_children(self) -> None:
        root = SessionState(id="root", task="", workspace="", permission_mode="")
        child = SessionState(
            id="child", task="", workspace="", permission_mode="", parent_id="root"
        )
        grand = SessionState(
            id="grand", task="", workspace="", permission_mode="", parent_id="child"
        )
        roots, children = build_session_tree([root, child, grand])
        self.assertEqual(roots, ["root"])
        self.assertEqual(children, {"root": ["child"], "child": ["grand"]})

    def test_build_session_tree_missing_parent_becomes_root(self) -> None:
        orphan = SessionState(
            id="orphan", task="", workspace="", permission_mode="", parent_id="gone"
        )
        roots, children = build_session_tree([orphan])
        self.assertEqual(roots, ["orphan"])
        self.assertEqual(children, {})

    def test_rewind_messages_rejects_before_compaction_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / "runs" / "sessions")
            state = store.create(
                "task",
                tmp,
                PermissionMode.AUTO.value,
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "summary"},
                    {"role": "user", "content": "new"},
                ],
                session_id="s-1",
                compaction_floor=3,
            )

            with self.assertRaises(ValueError):
                store.rewind_messages(state.id, 2)

            rewound = store.rewind_messages(state.id, 3)
            self.assertEqual(len(rewound.messages), 3)


class AtomicWriteTest(unittest.TestCase):
    def test_atomic_write_keeps_old_file_if_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text("old", encoding="utf-8")

            with patch("ninecoder.persistence.os.replace", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    atomic_write_text(path, "new")

            self.assertEqual(path.read_text(encoding="utf-8"), "old")
            self.assertEqual(list(path.parent.glob(".state.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
