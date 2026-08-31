from __future__ import annotations

from copy import deepcopy
from typing import Any


MAX_TOOL_CHARS = 8000
KEEP_RECENT_MESSAGES = 18


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


def _compact_message(message: dict[str, Any]) -> dict[str, Any]:
    copied = deepcopy(message)
    if copied.get("role") != "tool":
        return copied
    content = copied.get("content")
    if not isinstance(content, str) or len(content) <= MAX_TOOL_CHARS:
        return copied
    head = content[: MAX_TOOL_CHARS // 2]
    tail = content[-MAX_TOOL_CHARS // 2 :]
    copied["content"] = (
        f"{head}\n\n[... {len(content) - len(head) - len(tail)} chars omitted ...]\n\n{tail}"
    )
    return copied
