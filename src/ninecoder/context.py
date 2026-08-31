from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


MAX_TOOL_CHARS = 8000
TOOL_INLINE_CHARS = 2400
KEEP_RECENT_MESSAGES = 18
SUMMARY_PREVIEW_CHARS = 240


def compact_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(messages) <= KEEP_RECENT_MESSAGES + 2:
        return [_compact_message(message) for message in messages]
    head = messages[:2]
    tail = messages[-KEEP_RECENT_MESSAGES:]
    omitted = len(messages) - len(head) - len(tail)
    summary = {
        "role": "user",
        "content": (
            f"[Context compacted: {omitted} older messages omitted. "
            "Use current files and recent tool results as the source of truth.]"
        ),
    }
    return [_compact_message(message) for message in [*head, summary, *tail]]


@dataclass(frozen=True)
class StoredToolOutput:
    content_for_model: str
    path: Path | None = None


class ContextManager:
    def __init__(
        self,
        workspace_root: str | Path,
        runs_dir: str,
        session_id: str,
        *,
        keep_recent_messages: int = KEEP_RECENT_MESSAGES,
        max_tool_chars: int = MAX_TOOL_CHARS,
    ):
        self.workspace_root = Path(workspace_root)
        self.root = self.workspace_root / runs_dir / "context" / session_id
        self.keep_recent_messages = keep_recent_messages
        self.max_tool_chars = max_tool_chars
        self.root.mkdir(parents=True, exist_ok=True)

    def store_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        content: str,
    ) -> StoredToolOutput:
        if len(content) <= self.max_tool_chars:
            return StoredToolOutput(content)
        path = self.root / "tool-results" / f"{_safe_name(tool_call_id)}-{_safe_name(tool_name)}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        rel_path = path.relative_to(self.workspace_root)
        preview = _head_tail(content, TOOL_INLINE_CHARS)
        return StoredToolOutput(
            (
                "Tool output was too large for the live context.\n"
                f"Full output saved at: {rel_path}\n"
                "Use read_file on that path if the omitted details are needed.\n\n"
                f"{preview}"
            ),
            path,
        )

    def compact_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(messages) <= self.keep_recent_messages + 2:
            return [_compact_message(message, self.max_tool_chars) for message in messages]
        head = messages[:2]
        tail = messages[-self.keep_recent_messages :]
        omitted = messages[2 : -self.keep_recent_messages]
        summary_text = summarize_messages(omitted)
        summary_path = self.root / "summary.md"
        summary_path.write_text(summary_text + "\n", encoding="utf-8")
        rel_path = summary_path.relative_to(self.workspace_root)
        summary = {
            "role": "user",
            "content": (
                f"[Context compacted: {len(omitted)} older messages summarized at {rel_path}.]\n\n"
                f"{summary_text}\n\n"
                "Use current files and recent tool results as the source of truth."
            ),
        }
        return [
            _compact_message(message, self.max_tool_chars)
            for message in [*head, summary, *tail]
        ]


def summarize_messages(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "No older messages were omitted."
    lines = ["Older conversation summary:"]
    for index, message in enumerate(messages, 1):
        role = str(message.get("role", "unknown"))
        name = message.get("name")
        label = f"{role}:{name}" if name else role
        content = str(message.get("content", "")).replace("\n", " ").strip()
        tool_calls = message.get("tool_calls")
        if tool_calls:
            content = f"{content} tool_calls={len(tool_calls)}".strip()
        if len(content) > SUMMARY_PREVIEW_CHARS:
            content = content[:SUMMARY_PREVIEW_CHARS] + "..."
        lines.append(f"- {index}. {label}: {content or '(no text)'}")
    return "\n".join(lines)


def _compact_message(message: dict[str, Any], max_tool_chars: int = MAX_TOOL_CHARS) -> dict[str, Any]:
    copied = deepcopy(message)
    if copied.get("role") != "tool":
        return copied
    content = copied.get("content")
    if not isinstance(content, str) or len(content) <= max_tool_chars:
        return copied
    copied["content"] = _head_tail(content, max_tool_chars)
    return copied


def _head_tail(text: str, limit: int) -> str:
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n\n[... {len(text) - len(head) - len(tail)} chars omitted ...]\n\n{tail}"


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-") or "output"
