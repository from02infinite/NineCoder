from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class PermissionMode(str, Enum):
    PLAN = "plan"
    ASK = "ask"
    AUTO = "auto"


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class GrantDecision(str, Enum):
    """How the user wants to grant an interactive ``ask`` request."""

    ALLOW_ONCE = "allow_once"
    ALLOW_ALWAYS = "allow_always"
    DENY = "deny"


READ_ONLY_TOOLS = {
    "list_files",
    "read_file",
    "search",
    "load_skill",
    "spawn_subagent",
    "start_subagent_task",
    "read_subagent_task",
    "list_subagent_tasks",
    "update_todo",
    "update_task_graph",
}
MUTATING_TOOLS = {"edit_file", "write_file", "run_shell"}

SENSITIVE_NAMES = {
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    ".pgpass",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
}

# Match these names and any dotted variant, e.g. ``.env``, ``.env.local``,
# ``.env.production``, ``.env.backup``.
SENSITIVE_NAME_PREFIXES = {".env"}

SENSITIVE_PARTS = {".ssh", ".aws", ".gnupg", ".kube", ".docker"}
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


@dataclass(frozen=True)
class Permission:
    decision: Decision
    reason: str = ""


def hits_sensitive_path(path: str | Path) -> bool:
    candidate = Path(path)
    names = {part.lower() for part in candidate.parts}
    if names & SENSITIVE_PARTS:
        return True
    filename = candidate.name.lower()
    if filename in SENSITIVE_NAMES:
        return True
    if any(
        filename == prefix or filename.startswith(prefix + ".")
        for prefix in SENSITIVE_NAME_PREFIXES
    ):
        return True
    return any(filename.endswith(suffix) for suffix in SENSITIVE_SUFFIXES)


def evaluate_permission(mode: PermissionMode, tool_name: str, target: str = "") -> Permission:
    if target and hits_sensitive_path(target):
        return Permission(Decision.DENY, f"sensitive path is protected: {target}")
    if tool_name == "finish":
        return Permission(Decision.ALLOW)
    if tool_name in READ_ONLY_TOOLS:
        return Permission(Decision.ALLOW)
    if mode is PermissionMode.PLAN and tool_name in MUTATING_TOOLS:
        return Permission(Decision.DENY, f"{tool_name} is not allowed in plan mode")
    if mode is PermissionMode.ASK and tool_name in MUTATING_TOOLS:
        return Permission(Decision.ASK, f"{tool_name} changes local state")
    return Permission(Decision.ALLOW)
