from __future__ import annotations


class AgentError(Exception):
    """Base exception for user-visible agent failures."""


class ModelError(AgentError):
    """Raised when the model provider cannot return a usable response."""


class ToolError(AgentError):
    """Raised when a local tool cannot complete its work."""


class PermissionDenied(ToolError):
    """Raised when a requested operation is denied by policy."""
