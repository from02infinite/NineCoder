from __future__ import annotations

from typing import Any

from ninecoder.tools import ToolRegistry


class LocalCapabilityRouter:
    """Tiny MCP-like router over NineCoder's local tool registry."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def handle(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = params or {}
        if method == "tools/list":
            return {"tools": self.registry.schemas()}
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})
            if not isinstance(name, str):
                return {"error": "tools/call requires string field: name"}
            if not isinstance(arguments, dict):
                return {"error": "tools/call requires object field: arguments"}
            result = self.registry.execute(name, arguments)
            return {
                "content": result.content,
                "is_error": result.is_error,
                "metadata": result.metadata,
                "terminate": result.terminate,
            }
        return {"error": f"unknown method: {method}"}
