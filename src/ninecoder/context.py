from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from ninecoder.persistence import atomic_write_text


MAX_TOOL_CHARS = 8000
TOOL_INLINE_CHARS = 2400
KEEP_RECENT_MESSAGES = 18
SUMMARY_PREVIEW_CHARS = 240

_SUMMARY_SYSTEM = (
    "You condense a coding-agent conversation for context compaction. "
    "Preserve concrete facts: files read or edited, commands run and their "
    "results, decisions made, and outstanding work. Be concise."
)


def compact_messages(messages: list[dict[str, Any]], *, force: bool = False) -> list[dict[str, Any]]:
    if not force and len(messages) <= KEEP_RECENT_MESSAGES + 2:
        return [_compact_message(message) for message in valid_model_messages(messages)]
    head = messages[:2]
    groups = message_groups(messages[2:])
    tail_start = tail_start_for(groups, 1 if force else KEEP_RECENT_MESSAGES)
    omitted_groups = groups[:tail_start]
    tail = flatten_groups(groups[tail_start:])
    omitted = len(flatten_groups(omitted_groups))
    summary = {
        "role": "user",
        "content": (
            f"[Context compacted: {omitted} older messages omitted. "
            "Use current files and recent tool results as the source of truth.]"
        ),
    }
    compacted = valid_model_messages([*head, summary, *tail])
    return [_compact_message(message) for message in compacted]


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
        model: Any | None = None,
    ):
        self.workspace_root = Path(workspace_root)
        self.root = self.workspace_root / runs_dir / "context" / session_id
        self.keep_recent_messages = keep_recent_messages
        self.max_tool_chars = max_tool_chars
        self.model = model
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
        atomic_write_text(path, content, encoding="utf-8")
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

    def compact_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        if not force and len(messages) <= self.keep_recent_messages + 2:
            return [
                _compact_message(message, self.max_tool_chars)
                for message in valid_model_messages(messages)
            ]
        head = messages[:2]
        groups = message_groups(messages[2:])
        tail_start = tail_start_for(groups, 1 if force else self.keep_recent_messages)
        omitted = flatten_groups(groups[:tail_start])
        tail = flatten_groups(groups[tail_start:])
        summary_text = self._summarize(omitted)
        summary_path = self.root / "summary.md"
        atomic_write_text(summary_path, summary_text + "\n", encoding="utf-8")
        rel_path = summary_path.relative_to(self.workspace_root)
        summary = {
            "role": "user",
            "content": (
                f"[Context compacted: {len(omitted)} older messages summarized at {rel_path}.]\n\n"
                f"{summary_text}\n\n"
                "Use current files and recent tool results as the source of truth."
            ),
        }
        compacted = valid_model_messages([*head, summary, *tail])
        return [
            _compact_message(message, self.max_tool_chars)
            for message in compacted
        ]

    def _summarize(self, messages: list[dict[str, Any]]) -> str:
        if self.model is None:
            return summarize_messages(messages)
        try:
            return summarize_with_model(self.model, messages)
        except Exception:
            # A summary call is a nicety, not a correctness requirement. If the
            # model is unavailable mid-run, fall back to a mechanical summary.
            return summarize_messages(messages)


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


def summarize_with_model(model: Any, messages: list[dict[str, Any]]) -> str:
    """Ask the model to condense older messages, with a mechanical fallback."""
    transcript = summarize_messages(messages)
    response = model.complete(
        [
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {
                "role": "user",
                "content": f"Summarize this older conversation:\n\n{transcript}",
            },
        ],
        [],
    )
    content = (getattr(response, "content", "") or "").strip()
    return content or summarize_messages(messages)


def _compact_message(message: dict[str, Any], max_tool_chars: int = MAX_TOOL_CHARS) -> dict[str, Any]:
    copied = deepcopy(message)
    if copied.get("role") != "tool":
        return copied
    content = copied.get("content")
    if not isinstance(content, str) or len(content) <= max_tool_chars:
        return copied
    copied["content"] = _head_tail(content, max_tool_chars)
    return copied


def message_groups(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        group = [message]
        index += 1
        if message.get("role") == "assistant" and message.get("tool_calls"):
            while index < len(messages) and messages[index].get("role") == "tool":
                group.append(messages[index])
                index += 1
        groups.append(group)
    return groups


def tail_start_for(groups: list[list[dict[str, Any]]], keep_recent_messages: int) -> int:
    kept = 0
    for index in range(len(groups) - 1, -1, -1):
        kept += len(groups[index])
        if kept >= keep_recent_messages:
            return index
    return 0


def flatten_groups(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [message for group in groups for message in group]


def valid_model_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role == "tool":
            index += 1
            continue
        if role == "assistant" and message.get("tool_calls"):
            group = [message]
            index += 1
            while index < len(messages) and messages[index].get("role") == "tool":
                group.append(messages[index])
                index += 1
            if tool_group_is_complete(group):
                valid.extend(group)
            elif message.get("content"):
                copied = deepcopy(message)
                copied.pop("tool_calls", None)
                valid.append(copied)
            continue
        valid.append(message)
        index += 1
    return valid


def tool_group_is_complete(group: list[dict[str, Any]]) -> bool:
    assistant = group[0]
    expected = tool_call_ids(assistant)
    actual = [
        str(message.get("tool_call_id"))
        for message in group[1:]
        if message.get("role") == "tool" and message.get("tool_call_id")
    ]
    return bool(expected) and len(actual) == len(expected) and set(actual) == expected


def tool_call_ids(message: dict[str, Any]) -> set[str]:
    ids = set()
    for raw_call in message.get("tool_calls") or []:
        if not isinstance(raw_call, dict):
            continue
        call_id = raw_call.get("id")
        if isinstance(call_id, str) and call_id:
            ids.add(call_id)
    return ids


def _head_tail(text: str, limit: int) -> str:
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n\n[... {len(text) - len(head) - len(tail)} chars omitted ...]\n\n{tail}"


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-") or "output"
