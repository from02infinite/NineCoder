from __future__ import annotations

import json
import time
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ninecoder.context import ContextManager, StoredToolOutput, compact_messages
from ninecoder.hooks import (
    AgentHook,
    AgentStartRequest,
    ModelRequest,
    NullHook,
    StopEvent,
    ToolDecision,
    ToolRequest,
    ToolResponse,
)
from ninecoder.memory import MemoryStore, extract_facts, memory_block
from ninecoder.model_client import ModelResponse
from ninecoder.permissions import PermissionMode
from ninecoder.prompts import SYSTEM_PROMPT, no_tool_retry, skills_block, task_prompt
from ninecoder.session import SessionState, SessionStore
from ninecoder.skills import SkillLibrary
from ninecoder.subagents import SubagentTaskRunner
from ninecoder.tools import ToolRegistry, ToolResult
from ninecoder.trajectory import Trajectory
from ninecoder.workspace import Workspace

if TYPE_CHECKING:
    from ninecoder.ui.base import AgentUI


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
    memory: bool = True
    memory_file: str = "MEMORY.md"
    context_window_tokens: int = 32000


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
        ui: "AgentUI | None" = None,
        skill_library: SkillLibrary | None = None,
    ):
        from ninecoder.ui.base import AgentUI

        self.model = model
        self.workspace = workspace
        self.config = config
        self.ui: "AgentUI" = ui if ui is not None else AgentUI()
        self.tools = ToolRegistry(
            workspace,
            config.permission_mode,
            model=model,
            ui=self.ui,
            skill_library=skill_library,
        )
        self.hooks: list[AgentHook] = hooks or [NullHook()]
        self.messages: list[dict[str, Any]] = []
        self.runs_root = Path(workspace.root) / config.runs_dir
        self.session_store = SessionStore(self.runs_root / "sessions")
        self.trajectory = Trajectory(self.runs_root)
        self.session: SessionState | None = None
        self.context_manager: ContextManager | None = None

    def run(self, task: str) -> AgentRun:
        self._start(task)
        return self._loop()

    def open_session(self) -> SessionState:
        """Load ``config.resume_session`` without running a model turn."""
        if not self.config.resume_session:
            raise RuntimeError("no resume session configured")
        self.session = self.session_store.load(self.config.resume_session)
        self.messages = list(self.session.messages)
        self.tools.todos = list(self.session.todos)
        self.tools.task_graph = list(self.session.task_graph)
        if self.tools.subagent_runner is not None:
            self.tools.subagent_runner.import_tasks(self.session.subagent_tasks)
        self.context_manager = ContextManager(
            self.workspace.root,
            self.config.runs_dir,
            self.session.id,
            max_context_tokens=self.config.context_window_tokens,
            model=self.model,
        )
        self.trajectory = Trajectory(self.runs_root, run_name=self.session.id)
        return self.session

    def continue_turn(self, text: str) -> AgentRun:
        """Continue the active session with a new user message."""
        if self.session is None:
            raise RuntimeError("no active session to continue")
        self.session.status = "running"
        self.messages.append({"role": "user", "content": text})
        self.ui.user_message(text)
        return self._loop()

    def compact_context(self) -> Path:
        """Manually compact the active session's model context and persist it."""
        if self.session is None:
            raise RuntimeError("no active session to compact")
        if self.context_manager is None:
            self.messages = compact_messages(self.messages, force=True)
        else:
            self.messages = self.context_manager.compact_messages(self.messages, force=True)
        return self._save_session(self.session.step)

    def fork(self, parent_id: str, text: str) -> AgentRun:
        """Branch a new session off ``parent_id`` and run one turn on it."""
        parent = self.session_store.load(parent_id)
        new_messages = list(parent.messages) + [{"role": "user", "content": text}]
        self.messages = new_messages
        self.session = self.session_store.create(
            _short_label(text),
            parent.workspace,
            parent.permission_mode,
            new_messages,
            parent_id=parent_id,
            compaction_floor=parent.compaction_floor,
        )
        self.tools.todos = []
        self.tools.task_graph = []
        if self.tools.subagent_runner is not None:
            self.tools.subagent_runner = SubagentTaskRunner(self.model)
        self.context_manager = ContextManager(
            self.workspace.root,
            self.config.runs_dir,
            self.session.id,
            max_context_tokens=self.config.context_window_tokens,
            model=self.model,
        )
        self.trajectory = Trajectory(self.runs_root, run_name=self.session.id)
        self.trajectory.write(
            "run_start",
            {
                "task": text,
                "workspace": str(self.workspace.root),
                "permission_mode": self.config.permission_mode.value,
                "session_id": self.session.id,
                "parent_id": parent_id,
                "resumed": False,
            },
        )
        self.ui.user_message(text)
        return self._loop()

    def _start(self, task: str) -> None:
        resumed = False
        if self.config.resume_session:
            self.open_session()
            task = self.session.task
            resumed = True
        else:
            system_content = SYSTEM_PROMPT
            if self.config.memory:
                memory = MemoryStore(
                    self.workspace.root, self.config.memory_file
                ).read().strip()
                if memory:
                    system_content += "\n\n" + memory_block(memory)
            skills = self.tools.skill_library.names()
            if skills:
                system_content += "\n\n" + skills_block(skills)
            start_request = self._before_agent_start(
                AgentStartRequest(
                    task,
                    str(self.workspace.root),
                    self.config.test_cmd,
                    system_content,
                )
            )
            system_content = start_request.system_prompt
            self.messages = [
                {"role": "system", "content": system_content},
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
                max_context_tokens=self.config.context_window_tokens,
                model=self.model,
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
        if resumed:
            self.ui.session_history(self.messages)
        else:
            self.ui.user_message(task)

    def _loop(self) -> AgentRun:
        if self.session is None:
            raise RuntimeError("session is not initialized")
        start_step = self.session.step
        consecutive_format_errors = 0
        for step in range(start_step + 1, start_step + self.config.max_steps + 1):
            self.ui.debug(f"iteration={step}")
            model_request = self._before_model(
                ModelRequest(self._model_messages(), self.tools.schemas())
            )
            self.ui.model_started()
            model_started_at = time.perf_counter()
            response = self.model.complete(model_request.messages, model_request.tools)
            elapsed = time.perf_counter() - model_started_at
            response = self._after_model(response)
            if self.context_manager is not None:
                self.context_manager.record_usage(model_request.messages, response.usage)
            self.ui.model_finished(elapsed, response.finish_reason)
            if response.content and response.content.strip():
                self.ui.assistant_text(response.content)
            assistant_message = {
                "role": "assistant",
                "content": response.content,
            }
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    call.to_openai() for call in response.tool_calls
                ]
            self.messages.append(assistant_message)
            # Persist once per model step, not per tool result. Tool results are
            # intermediate within a step; saving after each would rewrite the
            # whole session JSON on every tool call (O(n^2) disk I/O) and leave
            # a partial tool group that resume would have to drop anyway.
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
                self.ui.debug(f"no tool calls (finish_reason={response.finish_reason})")
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
                tool_decision = self._before_tool(ToolRequest(call.name, call.arguments))
                tool_request = tool_decision.request or ToolRequest(call.name, call.arguments)
                self.ui.debug(f"tool={tool_request.name} {_compact_args(tool_request.arguments)}")
                self.ui.tool_started(tool_request.name, tool_request.arguments)
                tool_started_at = time.perf_counter()
                if tool_decision.blocked_result is None:
                    result = self.tools.execute(tool_request.name, tool_request.arguments)
                else:
                    blocked = tool_decision.blocked_result
                    result = ToolResult(
                        blocked.content,
                        metadata=blocked.metadata,
                        is_error=blocked.is_error,
                        terminate=blocked.terminate,
                    )
                elapsed_ms = (time.perf_counter() - tool_started_at) * 1000
                result = self._after_tool(tool_request.name, result)
                status = "error" if result.is_error else "ok"
                self.ui.debug(f"tool {tool_request.name} {status}: {elapsed_ms:.0f}ms")
                if result.is_error:
                    self.ui.tool_failed(tool_request.name, tool_request.arguments, result.content)
                else:
                    self.ui.tool_finished(tool_request.name, tool_request.arguments, result)
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
                    return self._stop(result.content, step, "finished")
        return self._stop(
            f"Reached max_steps={self.config.max_steps} before finish.",
            start_step + self.config.max_steps,
            "max_steps",
        )

    def _stop(self, summary: str, steps: int, stopped_by: str) -> AgentRun:
        self._notify_stop(summary, steps, stopped_by)
        self.ui.session_finished(summary=summary, steps=steps, stopped_by=stopped_by)
        session_path = self._save_session(steps, summary=summary, stopped_by=stopped_by)
        self._remember(summary)
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
        if self.context_manager is not None:
            self.session.compaction_floor = max(
                self.session.compaction_floor,
                self.context_manager.compaction_floor,
            )
        self.session.todos = list(self.tools.todos)
        self.session.task_graph = list(self.tools.task_graph)
        self.session.subagent_tasks = (
            self.tools.subagent_runner.export_tasks()
            if self.tools.subagent_runner is not None
            else []
        )
        if stopped_by:
            self.session.status = "finished" if stopped_by == "finished" else "stopped"
            self.session.stopped_by = stopped_by
            self.session.summary = summary
        return self.session_store.save(self.session)

    def _remember(self, summary: str) -> None:
        """Distill the finished run into durable memory. A nicety, not a
        correctness requirement: any failure here must never break the run."""
        if not self.config.memory or self.session is None or not summary.strip():
            return
        try:
            facts = extract_facts(self.model, self.session.task, summary).strip()
        except Exception:
            return
        if not facts:
            return
        try:
            MemoryStore(self.workspace.root, self.config.memory_file).append(
                self.session.task, facts
            )
        except OSError:
            pass

    def _model_messages(self) -> list[dict[str, Any]]:
        if self.context_manager is None:
            return compact_messages(self.messages)
        return self.context_manager.compact_messages(self.messages)

    def _before_agent_start(self, request: AgentStartRequest) -> AgentStartRequest:
        for hook in self.hooks:
            method = getattr(hook, "before_agent_start", None)
            if method is None:
                continue
            updated = method(request)
            if isinstance(updated, AgentStartRequest):
                request = updated
        return request

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

    def _before_tool(self, request: ToolRequest) -> ToolDecision:
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
            elif isinstance(updated, ToolDecision):
                if updated.request is not None:
                    request = updated.request
                if updated.blocked_result is not None:
                    return ToolDecision(request, updated.blocked_result)
            elif isinstance(updated, tuple) and len(updated) == 2:
                name, arguments = updated
                if isinstance(name, str) and isinstance(arguments, dict):
                    request = ToolRequest(name, arguments)
        return ToolDecision(request)

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

    def _notify_stop(self, summary: str, steps: int, stopped_by: str) -> None:
        event = StopEvent(summary, steps, stopped_by)
        for hook in self.hooks:
            method = getattr(hook, "on_stop", None)
            if method is not None:
                method(event)
            if stopped_by == "finished":
                legacy_method = getattr(hook, "on_finish", None)
                if legacy_method is not None:
                    legacy_method(summary)


def _expects_one_argument(method: Any) -> bool:
    try:
        return len(signature(method).parameters) == 1
    except (TypeError, ValueError):
        return True


def _compact_args(arguments: dict[str, Any], limit: int = 160) -> str:
    text = json.dumps(arguments, ensure_ascii=False)
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _short_label(text: str, limit: int = 60) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else f"{text[: limit - 3]}..."
