from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ninecoder.context import compact_messages
from ninecoder.model_client import ModelResponse
from ninecoder.permissions import PermissionMode
from ninecoder.prompts import SYSTEM_PROMPT, no_tool_retry, task_prompt
from ninecoder.tools import ToolRegistry
from ninecoder.trajectory import Trajectory
from ninecoder.workspace import Workspace


class ChatModel(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse: ...


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 30
    permission_mode: PermissionMode = PermissionMode.ASK
    test_cmd: str = ""
    runs_dir: str = "runs"


@dataclass(frozen=True)
class AgentRun:
    summary: str
    steps: int
    trajectory_path: Path
    stopped_by: str


class CodingAgent:
    def __init__(
        self,
        model: ChatModel,
        workspace: Workspace,
        config: AgentConfig,
    ):
        self.model = model
        self.workspace = workspace
        self.config = config
        self.tools = ToolRegistry(workspace, config.permission_mode)
        self.messages: list[dict[str, Any]] = []
        self.trajectory = Trajectory(Path(workspace.root) / config.runs_dir)

    def run(self, task: str) -> AgentRun:
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": task_prompt(task, str(self.workspace.root), self.config.test_cmd),
            },
        ]
        self.trajectory.write(
            "run_start",
            {
                "task": task,
                "workspace": str(self.workspace.root),
                "permission_mode": self.config.permission_mode.value,
            },
        )
        consecutive_format_errors = 0
        for step in range(1, self.config.max_steps + 1):
            response = self.model.complete(compact_messages(self.messages), self.tools.schemas())
            assistant_message = {
                "role": "assistant",
                "content": response.content,
            }
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    call.to_openai() for call in response.tool_calls
                ]
            self.messages.append(assistant_message)
            self.trajectory.write(
                "assistant",
                {
                    "step": step,
                    "content": response.content,
                    "tool_calls": [call.to_openai() for call in response.tool_calls],
                    "finish_reason": response.finish_reason,
                    "usage": response.usage,
                },
            )
            if not response.tool_calls:
                consecutive_format_errors += 1
                if consecutive_format_errors >= 3:
                    return self._stop(
                        "The model repeatedly answered without tool calls.",
                        step,
                        "format_error",
                    )
                self.messages.append({"role": "user", "content": no_tool_retry(response.content)})
                continue
            consecutive_format_errors = 0
            for call in response.tool_calls:
                result = self.tools.execute(call.name, call.arguments)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": result.content,
                }
                self.messages.append(tool_message)
                self.trajectory.write(
                    "tool_result",
                    {
                        "step": step,
                        "tool": call.name,
                        "arguments": call.arguments,
                        "is_error": result.is_error,
                        "terminate": result.terminate,
                        "content": result.content,
                        "metadata": result.metadata,
                    },
                )
                if result.terminate:
                    return self._stop(result.content, step, "finished")
        return self._stop(
            f"Reached max_steps={self.config.max_steps} before finish.",
            self.config.max_steps,
            "max_steps",
        )

    def _stop(self, summary: str, steps: int, stopped_by: str) -> AgentRun:
        self.trajectory.write(
            "run_end",
            {"summary": summary, "steps": steps, "stopped_by": stopped_by},
        )
        return AgentRun(summary, steps, self.trajectory.path, stopped_by)
