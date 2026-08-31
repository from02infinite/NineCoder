from __future__ import annotations

from typing import Any, Protocol


class AgentHook(Protocol):
    def before_tool(self, name: str, arguments: dict[str, Any]) -> None: ...

    def after_tool(self, name: str, result: str, is_error: bool) -> None: ...

    def on_finish(self, summary: str) -> None: ...


class NullHook:
    def before_tool(self, name: str, arguments: dict[str, Any]) -> None:
        return None

    def after_tool(self, name: str, result: str, is_error: bool) -> None:
        return None

    def on_finish(self, summary: str) -> None:
        return None
