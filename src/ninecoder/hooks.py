from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentStartRequest:
    task: str
    workspace: str
    test_cmd: str
    system_prompt: str


@dataclass(frozen=True)
class ModelRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]


@dataclass(frozen=True)
class ToolRequest:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResponse:
    content: str
    is_error: bool = False
    terminate: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolDecision:
    request: ToolRequest | None = None
    blocked_result: ToolResponse | None = None


@dataclass(frozen=True)
class StopEvent:
    summary: str
    steps: int
    stopped_by: str


class AgentHook(Protocol):
    def before_agent_start(self, request: AgentStartRequest) -> AgentStartRequest | None: ...

    def before_model(self, request: ModelRequest) -> ModelRequest | None: ...

    def after_model(self, response: Any) -> Any | None: ...

    def before_tool(self, request: ToolRequest) -> ToolRequest | ToolDecision | None: ...

    def after_tool(self, response: ToolResponse) -> ToolResponse | None: ...

    def on_stop(self, event: StopEvent) -> None: ...

    def on_finish(self, summary: str) -> None: ...


class NullHook:
    def before_agent_start(self, request: AgentStartRequest) -> AgentStartRequest | None:
        return None

    def before_model(self, request: ModelRequest) -> ModelRequest | None:
        return None

    def after_model(self, response: Any) -> Any | None:
        return None

    def before_tool(self, request: ToolRequest) -> ToolRequest | ToolDecision | None:
        return None

    def after_tool(self, response: ToolResponse) -> ToolResponse | None:
        return None

    def on_stop(self, event: StopEvent) -> None:
        return None

    def on_finish(self, summary: str) -> None:
        return None


class LegacyToolHook:
    def before_tool(self, name: str, arguments: dict[str, Any]) -> None:
        return None

    def after_tool(self, name: str, result: str, is_error: bool) -> None:
        return None
