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


if __name__ == "__main__":
    unittest.main()
