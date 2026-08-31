import tempfile
import unittest
from pathlib import Path
from typing import Any

from ninecoder.agent import AgentConfig, CodingAgent
from ninecoder.model_client import ModelResponse, ToolCall
from ninecoder.permissions import PermissionMode
from ninecoder.session import SessionStore
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

    def test_agent_persists_subagent_tasks_in_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = CodingAgent(
                SubagentTaskModel(),
                Workspace(tmp),
                AgentConfig(permission_mode=PermissionMode.AUTO),
            ).run("ask reviewer")

            state = SessionStore(Path(tmp) / "runs" / "sessions").load(result.session_id)

            self.assertEqual(state.subagent_tasks[0]["result"], "subagent result")


if __name__ == "__main__":
    unittest.main()
