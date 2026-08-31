import tempfile
import unittest
from pathlib import Path
from typing import Any

from ninecoder.agent import AgentConfig, CodingAgent
from ninecoder.model_client import ModelResponse, ToolCall
from ninecoder.permissions import PermissionMode
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


if __name__ == "__main__":
    unittest.main()
