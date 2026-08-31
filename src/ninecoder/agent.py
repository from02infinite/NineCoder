from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Any, Protocol

from ninecoder.context import ContextManager, StoredToolOutput, compact_messages
from ninecoder.hooks import AgentHook, ModelRequest, NullHook, ToolRequest, ToolResponse
from ninecoder.model_client import ModelResponse
from ninecoder.permissions import PermissionMode
from ninecoder.prompts import SYSTEM_PROMPT, no_tool_retry, task_prompt
from ninecoder.session import SessionState, SessionStore
from ninecoder.tools import ToolRegistry, ToolResult
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
    resume_session: str = ""


@dataclass(frozen=True)
class AgentRun:
    summary: str
    steps: int
    trajectory_path: Path
    session_id: str
    session_path: Path
    stopped_by: str


class CodingAgent:
    def __init__(
        self,
        model: ChatModel,
        workspace: Workspace,
        config: AgentConfig,
        hooks: list[AgentHook] | None = None,
    ):
        self.model = model
        self.workspace = workspace
        self.config = config
        self.tools = ToolRegistry(workspace, config.permission_mode, model=model)
        self.hooks: list[AgentHook] = hooks or [NullHook()]
        self.messages: list[dict[str, Any]] = []
        self.runs_root = Path(workspace.root) / config.runs_dir
        self.session_store = SessionStore(self.runs_root / "sessions")
        self.trajectory = Trajectory(self.runs_root)
        self.session: SessionState | None = None
        self.context_manager: ContextManager | None = None

    def run(self, task: str) -> AgentRun:
        start_step = 0
        if self.config.resume_session:
            self.session = self.session_store.load(self.config.resume_session)
            self.messages = list(self.session.messages)
            self.tools.todos = list(self.session.todos)
            self.tools.task_graph = list(self.session.task_graph)
            start_step = self.session.step
            task = self.session.task
            self.trajectory = Trajectory(self.runs_root, run_name=self.session.id)
        else:
            self.messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": task_prompt(task, str(self.workspace.root), self.config.test_cmd),
                },
            ]
            self.session = self.session_store.create(
                task,
                str(self.workspace.root),
                self.config.permission_mode.value,
                self.messages,
            )
            self.trajectory = Trajectory(self.runs_root, run_name=self.session.id)
        self.context_manager = ContextManager(
            self.workspace.root,
            self.config.runs_dir,
            self.session.id,
        )
        self.trajectory.write(
            "run_start",
            {
                "task": task,
                "workspace": str(self.workspace.root),
                "permission_mode": self.config.permission_mode.value,
                "session_id": self.session.id,
                "resumed": bool(self.config.resume_session),
            },
        )
        consecutive_format_errors = 0
        for step in range(start_step + 1, start_step + self.config.max_steps + 1):
            model_request = self._before_model(
                ModelRequest(self._model_messages(), self.tools.schemas())
            )
            response = self.model.complete(model_request.messages, model_request.tools)
            response = self._after_model(response)
            assistant_message = {
                "role": "assistant",
                "content": response.content,
            }
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    call.to_openai() for call in response.tool_calls
                ]
            self.messages.append(assistant_message)
            self._save_session(step)
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
                tool_request = self._before_tool(ToolRequest(call.name, call.arguments))
                result = self.tools.execute(tool_request.name, tool_request.arguments)
                result = self._after_tool(tool_request.name, result)
                stored_result = self._store_tool_result(
                    call.id,
                    tool_request.name,
                    result.content,
                )
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": tool_request.name,
                    "content": stored_result.content_for_model,
                }
                self.messages.append(tool_message)
                self._save_session(step)
                self.trajectory.write(
                    "tool_result",
                    {
                        "step": step,
                        "tool": tool_request.name,
                        "arguments": tool_request.arguments,
                        "is_error": result.is_error,
                        "terminate": result.terminate,
                        "content": result.content,
                        "stored_content_path": (
                            str(stored_result.path) if stored_result.path else ""
                        ),
                        "metadata": result.metadata,
                    },
                )
                if result.terminate:
                    for hook in self.hooks:
                        method = getattr(hook, "on_finish", None)
                        if method is not None:
                            method(result.content)
                    return self._stop(result.content, step, "finished")
        return self._stop(
            f"Reached max_steps={self.config.max_steps} before finish.",
            start_step + self.config.max_steps,
            "max_steps",
        )

    def _stop(self, summary: str, steps: int, stopped_by: str) -> AgentRun:
        session_path = self._save_session(steps, summary=summary, stopped_by=stopped_by)
        self.trajectory.write(
            "run_end",
            {"summary": summary, "steps": steps, "stopped_by": stopped_by},
        )
        return AgentRun(
            summary,
            steps,
            self.trajectory.path,
            self.session.id if self.session else "",
            session_path,
            stopped_by,
        )

    def _save_session(
        self,
        step: int,
        *,
        summary: str = "",
        stopped_by: str = "",
    ) -> Path:
        if self.session is None:
            raise RuntimeError("session is not initialized")
        self.session.step = step
        self.session.messages = list(self.messages)
        self.session.todos = list(self.tools.todos)
        self.session.task_graph = list(self.tools.task_graph)
        if stopped_by:
            self.session.status = "finished" if stopped_by == "finished" else "stopped"
            self.session.stopped_by = stopped_by
            self.session.summary = summary
        return self.session_store.save(self.session)

    def _model_messages(self) -> list[dict[str, Any]]:
        if self.context_manager is None:
            return compact_messages(self.messages)
        return self.context_manager.compact_messages(self.messages)

    def _before_model(self, request: ModelRequest) -> ModelRequest:
        for hook in self.hooks:
            method = getattr(hook, "before_model", None)
            if method is None:
                continue
            updated = method(request)
            if isinstance(updated, ModelRequest):
                request = updated
        return request

    def _after_model(self, response: ModelResponse) -> ModelResponse:
        for hook in self.hooks:
            method = getattr(hook, "after_model", None)
            if method is None:
                continue
            updated = method(response)
            if isinstance(updated, ModelResponse):
                response = updated
        return response

    def _before_tool(self, request: ToolRequest) -> ToolRequest:
        for hook in self.hooks:
            method = getattr(hook, "before_tool", None)
            if method is None:
                continue
            if _expects_one_argument(method):
                updated = method(request)
            else:
                updated = method(request.name, request.arguments)
            if isinstance(updated, ToolRequest):
                request = updated
            elif isinstance(updated, tuple) and len(updated) == 2:
                name, arguments = updated
                if isinstance(name, str) and isinstance(arguments, dict):
                    request = ToolRequest(name, arguments)
        return request

    def _after_tool(self, tool_name: str, result: ToolResult) -> ToolResult:
        response = ToolResponse(
            content=result.content,
            is_error=result.is_error,
            terminate=result.terminate,
            metadata=result.metadata,
        )
        for hook in self.hooks:
            method = getattr(hook, "after_tool", None)
            if method is None:
                continue
            if _expects_one_argument(method):
                updated = method(response)
            else:
                updated = method(tool_name, response.content, response.is_error)
            if isinstance(updated, ToolResponse):
                response = updated
            elif isinstance(updated, str):
                response = ToolResponse(
                    content=updated,
                    is_error=response.is_error,
                    terminate=response.terminate,
                    metadata=response.metadata,
                )
        return ToolResult(
            content=response.content,
            metadata=response.metadata,
            is_error=response.is_error,
            terminate=response.terminate,
        )

    def _store_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        content: str,
    ) -> StoredToolOutput:
        if self.context_manager is None:
            return StoredToolOutput(content)
        return self.context_manager.store_tool_result(tool_call_id, tool_name, content)


def _expects_one_argument(method: Any) -> bool:
    try:
        return len(signature(method).parameters) == 1
    except (TypeError, ValueError):
        return True
