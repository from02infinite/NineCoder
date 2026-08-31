import tempfile
import unittest
from pathlib import Path
from typing import Any

from ninecoder.agent import AgentConfig, CodingAgent
from ninecoder.hooks import (
    AgentStartRequest,
    ModelRequest,
    StopEvent,
    ToolDecision,
    ToolRequest,
    ToolResponse,
)
from ninecoder.model_client import ModelResponse, ToolCall
from ninecoder.permissions import PermissionMode
from ninecoder.workspace import Workspace


class CaptureModel:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.messages = messages
        return ModelResponse("finish", [ToolCall("call_1", "finish", {"summary": "ok"})])


class WriteModel:
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        return ModelResponse(
            "write",
            [ToolCall("call_1", "write_file", {"path": "a.txt", "content": "original"})],
        )


class NoToolModel:
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        return ModelResponse("plain answer without a tool call")


class RewriteSystemPromptHook:
    def before_agent_start(self, request: AgentStartRequest) -> AgentStartRequest:
        return AgentStartRequest(
            request.task,
            request.workspace,
            request.test_cmd,
            request.system_prompt + "\n\nHook-added startup policy.",
        )


class AppendModelContextHook:
    def before_model(self, request: ModelRequest) -> ModelRequest:
        return ModelRequest(
            [*request.messages, {"role": "user", "content": "hook context"}],
            request.tools,
        )


class RewriteToolHook:
    def before_tool(self, request: ToolRequest) -> ToolRequest:
        return ToolRequest(
            request.name,
            request.arguments | {"content": "rewritten"},
        )


class RewriteResultHook:
    def after_tool(self, response: ToolResponse) -> ToolResponse:
        if response.terminate:
            return ToolResponse("hook summary", terminate=True)
        return response


class BlockWriteHook:
    def before_tool(self, request: ToolRequest) -> ToolDecision | None:
        if request.name == "write_file":
            return ToolDecision(
                request=request,
                blocked_result=ToolResponse(
                    "Blocked write_file from hook policy.",
                    is_error=True,
                ),
            )
        return None


class RecordStopHook:
    def __init__(self) -> None:
        self.events: list[StopEvent] = []

    def on_stop(self, event: StopEvent) -> None:
        self.events.append(event)


class HookPipelineTest(unittest.TestCase):
    def test_before_agent_start_can_change_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = CaptureModel()

            CodingAgent(
                model,
                Workspace(tmp),
                AgentConfig(permission_mode=PermissionMode.AUTO, memory=False),
                hooks=[RewriteSystemPromptHook()],
            ).run("finish")

            self.assertIn("Hook-added startup policy.", model.messages[0]["content"])

    def test_before_model_can_change_model_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = CaptureModel()

            CodingAgent(
                model,
                Workspace(tmp),
                AgentConfig(permission_mode=PermissionMode.AUTO, memory=False),
                hooks=[AppendModelContextHook()],
            ).run("finish")

            self.assertEqual(model.messages[-1]["content"], "hook context")

    def test_before_tool_can_rewrite_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            CodingAgent(
                WriteModel(),
                Workspace(root),
                AgentConfig(max_steps=1, permission_mode=PermissionMode.AUTO),
                hooks=[RewriteToolHook()],
            ).run("write")

            self.assertEqual((root / "a.txt").read_text(encoding="utf-8"), "rewritten")

    def test_before_tool_can_block_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            result = CodingAgent(
                WriteModel(),
                Workspace(root),
                AgentConfig(max_steps=1, permission_mode=PermissionMode.AUTO, memory=False),
                hooks=[BlockWriteHook()],
            ).run("write")

            self.assertFalse((root / "a.txt").exists())
            self.assertEqual(result.stopped_by, "max_steps")

    def test_after_tool_can_rewrite_terminal_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = CodingAgent(
                CaptureModel(),
                Workspace(tmp),
                AgentConfig(permission_mode=PermissionMode.AUTO),
                hooks=[RewriteResultHook()],
            ).run("finish")

            self.assertEqual(result.summary, "hook summary")

    def test_on_stop_runs_for_non_finished_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hook = RecordStopHook()

            result = CodingAgent(
                NoToolModel(),
                Workspace(tmp),
                AgentConfig(max_steps=1, permission_mode=PermissionMode.AUTO, memory=False),
                hooks=[hook],
            ).run("never calls a tool")

            self.assertEqual(result.stopped_by, "max_steps")
            self.assertEqual(len(hook.events), 1)
            self.assertEqual(hook.events[0].stopped_by, "max_steps")


if __name__ == "__main__":
    unittest.main()
