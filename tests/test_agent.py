import tempfile
import unittest
from pathlib import Path
from typing import Any

from ninecoder.agent import AgentConfig, CodingAgent
from ninecoder.model_client import ModelResponse, ToolCall
from ninecoder.permissions import PermissionMode
from ninecoder.session import SessionStore
from ninecoder.workspace import Workspace


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                "write a file",
                [ToolCall("call_1", "write_file", {"path": "ok.txt", "content": "done"})],
            )
        return ModelResponse(
            "finish",
            [ToolCall("call_2", "finish", {"summary": "created ok.txt"})],
        )


class RecordingFinishModel:
    def __init__(self) -> None:
        self.user_lists: list[list[str]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.user_lists.append(
            [m.get("content", "") for m in messages if m["role"] == "user"]
        )
        return ModelResponse("done", [ToolCall("call_1", "finish", {"summary": "ok"})])


class StreamingFinishModel:
    """A model exposing the optional ``stream_complete`` capability."""

    def __init__(self) -> None:
        self.stream_ends = 0

    def stream_complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_chunk: Any,
    ) -> ModelResponse:
        for token in ["do", "ne"]:
            on_chunk(token)
        return ModelResponse(
            "done", [ToolCall("call_1", "finish", {"summary": "ok"})], streamed=True
        )


class StreamRecordingUI:
    def __init__(self) -> None:
        self.stream_chunks: list[str] = []
        self.stream_ends = 0

    def assistant_stream_chunk(self, chunk: str) -> None:
        self.stream_chunks.append(chunk)

    def assistant_stream_end(self) -> None:
        self.stream_ends += 1

    # Remaining AgentUI methods the loop touches during a finish-only run.
    def user_message(self, text: str) -> None:
        pass

    def model_started(self) -> None:
        pass

    def model_finished(self, elapsed: float, stop_reason: str) -> None:
        pass

    def tool_started(self, name: str, arguments: dict[str, Any]) -> None:
        pass

    def tool_finished(self, name: str, arguments: dict[str, Any], result: Any) -> None:
        pass

    def tool_failed(self, name: str, arguments: dict[str, Any], message: str) -> None:
        pass

    def debug(self, message: str) -> None:
        pass

    def session_finished(self, *, summary: str, steps: int, stopped_by: str) -> None:
        pass


class AgentTest(unittest.TestCase):
    def test_agent_runs_tool_loop_until_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(tmp)
            agent = CodingAgent(
                FakeModel(),
                workspace,
                AgentConfig(max_steps=5, permission_mode=PermissionMode.AUTO),
            )

            result = agent.run("create ok.txt")

            self.assertEqual(result.stopped_by, "finished")
            self.assertEqual((Path(tmp) / "ok.txt").read_text(encoding="utf-8"), "done")


class MultiTurnTest(unittest.TestCase):
    def test_continue_turn_appends_user_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = RecordingFinishModel()
            agent = CodingAgent(
                model,
                Workspace(tmp),
                AgentConfig(max_steps=5, permission_mode=PermissionMode.AUTO, memory=False),
            )
            agent.run("first task")
            agent.continue_turn("second turn")

            self.assertIn("second turn", model.user_lists[-1])
            user_contents = [
                m["content"] for m in agent.messages if m["role"] == "user"
            ]
            self.assertIn("second turn", user_contents)

    def test_manual_compact_context_persists_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = CodingAgent(
                FakeModel(),
                Workspace(tmp),
                AgentConfig(max_steps=5, permission_mode=PermissionMode.AUTO, memory=False),
            )
            result = agent.run("create ok.txt")
            if agent.context_manager is not None:
                agent.context_manager.model = None

            session_path = agent.compact_context()

            self.assertEqual(session_path, result.session_path)
            summary_file = Path(tmp) / "runs" / "context" / result.session_id / "summary.md"
            self.assertTrue(summary_file.exists())
            self.assertIn("write a file", summary_file.read_text(encoding="utf-8"))
            state = SessionStore(Path(tmp) / "runs" / "sessions").load(result.session_id)
            self.assertIn("Context compacted", state.messages[2]["content"])

    def test_fork_creates_child_session_with_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = CodingAgent(
                RecordingFinishModel(),
                Workspace(tmp),
                AgentConfig(max_steps=5, permission_mode=PermissionMode.AUTO, memory=False),
            )
            result = agent.run("base task")
            parent_id = result.session_id

            child = agent.fork(parent_id, "branch turn")

            self.assertNotEqual(child.session_id, parent_id)
            store = SessionStore(Path(tmp) / "runs" / "sessions")
            state = store.load(child.session_id)
            self.assertEqual(state.parent_id, parent_id)
            child_user = [m["content"] for m in state.messages if m["role"] == "user"]
            self.assertIn("branch turn", child_user)
            # The parent session is left untouched by the fork.
            parent_user = [
                m["content"]
                for m in store.load(parent_id).messages
                if m["role"] == "user"
            ]
            self.assertNotIn("branch turn", parent_user)


class StreamingAgentTest(unittest.TestCase):
    def test_agent_uses_stream_complete_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui = StreamRecordingUI()
            agent = CodingAgent(
                StreamingFinishModel(),
                Workspace(tmp),
                AgentConfig(max_steps=5, permission_mode=PermissionMode.AUTO, memory=False),
                ui=ui,
            )

            result = agent.run("stream it")

            self.assertEqual(result.stopped_by, "finished")
            self.assertEqual(ui.stream_chunks, ["do", "ne"])
            self.assertEqual(ui.stream_ends, 1)

    def test_agent_falls_back_to_complete_without_stream_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = CodingAgent(
                FakeModel(),
                Workspace(tmp),
                AgentConfig(max_steps=5, permission_mode=PermissionMode.AUTO, memory=False),
            )

            result = agent.run("create ok.txt")

            self.assertEqual(result.stopped_by, "finished")
            self.assertEqual((Path(tmp) / "ok.txt").read_text(encoding="utf-8"), "done")


if __name__ == "__main__":
    unittest.main()
