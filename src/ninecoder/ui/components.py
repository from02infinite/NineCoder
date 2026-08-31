"""Pure, testable formatters for tool summaries and diagnostics.

These functions take plain data (strings, dicts, and duck-typed results) and
return strings, so they have no dependency on the display backend and can be
unit-tested directly.
"""
from __future__ import annotations

import json
from typing import Any


# Tool name -> human verb. Fallback title-cases an unknown name.
_VERBS: dict[str, str] = {
    "read_file": "Read",
    "edit_file": "Edit",
    "write_file": "Write",
    "list_files": "List",
    "search": "Search",
    "run_shell": "Bash",
    "update_todo": "Update todo",
    "update_task_graph": "Update task graph",
    "load_skill": "Load skill",
    "spawn_subagent": "Subagent",
    "start_subagent_task": "Subagent",
    "read_subagent_task": "Read subagent",
    "list_subagent_tasks": "List subagent tasks",
    "finish": "Finish",
}


def tool_summary(name: str, arguments: dict[str, Any]) -> str:
    """A compact human label, e.g. ``Read calculator.py`` or ``Bash pytest``."""
    verb = _VERBS.get(name, name.replace("_", " ").title())
    target = _target_for(name, arguments)
    return f"{verb} {target}" if target else verb


def tool_result_summary(name: str, arguments: dict[str, Any], result: Any) -> str:
    """A short outcome line for a completed tool call, e.g. ``4 passed``."""
    content = getattr(result, "content", "") or ""
    metadata = getattr(result, "metadata", {}) or {}
    if name == "run_shell":
        return _run_shell_summary(content, metadata)
    if name == "finish":
        return first_line(content, 120) or "done"
    if name == "search":
        if _is_no_matches(content):
            return "no matches"
        return f"{_count_lines(content)} matches"
    if name in ("read_file", "list_files"):
        return f"{_count_lines(content)} lines"
    if name in ("edit_file", "write_file"):
        return _diff_summary(content)
    if name in ("update_todo", "update_task_graph"):
        return "updated"
    return first_line(content, 80) or "ok"


def permission_summary(name: str, arguments: dict[str, Any]) -> str:
    """The single line shown inside a permission panel."""
    if name == "run_shell":
        command = arguments.get("command")
        return f"$ {command}" if command else name
    if name in ("edit_file", "write_file", "read_file"):
        return f"{name} {arguments.get('path', '')}".strip()
    return tool_summary(name, arguments)


def format_args(arguments: dict[str, Any], *, limit: int = 160) -> str:
    """A compact ``key=value`` debug rendering of a tool's arguments."""
    if not arguments:
        return ""
    parts = []
    for key, value in arguments.items():
        text = _stringify(value)
        if len(text) > 60:
            text = text[:57] + "..."
        parts.append(f"{key}={text}")
    joined = ", ".join(parts)
    if len(joined) > limit:
        joined = joined[: limit - 3] + "..."
    return joined


def elapsed_human(seconds: float) -> str:
    """Human duration, e.g. ``1.42s`` / ``3ms`` / ``900µs``."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f}µs"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def first_line(text: str, limit: int = 120) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    if len(line) <= limit:
        return line
    return f"{line[: limit - 3]}..."


def _target_for(name: str, arguments: dict[str, Any]) -> str:
    if name in ("read_file", "edit_file", "write_file", "list_files"):
        return str(arguments.get("path") or "")
    if name == "search":
        pattern = str(arguments.get("pattern") or "")
        path = str(arguments.get("path") or "")
        if pattern and path and path not in (".", "./"):
            return f'"{pattern}" in {path}'
        return f'"{pattern}"' if pattern else ""
    if name == "run_shell":
        return first_line(str(arguments.get("command") or ""), 60)
    if name in ("load_skill", "spawn_subagent", "start_subagent_task"):
        return str(arguments.get("name") or arguments.get("role") or "")
    if name == "read_subagent_task":
        return str(arguments.get("task_id") or "")
    if name == "finish":
        return first_line(str(arguments.get("summary") or ""), 60)
    return ""


def _run_shell_summary(content: str, metadata: dict[str, Any]) -> str:
    payload = _parse_json_object(content)
    returncode = metadata.get("returncode")
    if returncode is None:
        returncode = payload.get("returncode")
    if returncode is None:
        return first_line(content, 80) or "ok"
    if returncode != 0:
        return f"exit code {returncode}"
    output = payload.get("output")
    tail = _last_nonempty_line(output) if isinstance(output, str) else ""
    return tail or "exit code 0"


def _diff_summary(content: str) -> str:
    added = sum(
        1 for line in content.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    removed = sum(
        1 for line in content.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    total = added + removed
    if total == 0:
        return "no changes"
    return f"changed {total} lines"


def _count_lines(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


def _is_no_matches(text: str) -> bool:
    return text.strip() == "(no matches)"


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
