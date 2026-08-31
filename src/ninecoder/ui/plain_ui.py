"""Plain-text UI backend.

Simple ``print``-based output with no ANSI escapes, no spinners, and no
interactive prompt machinery. Suitable for CI, pipes, and redirected files.
"""
from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from ninecoder.ui.base import AgentUI, UiContext
from ninecoder.ui.components import (
    format_args,
    tool_result_summary,
    tool_summary,
)


class PlainUI(AgentUI):
    def __init__(
        self,
        context: UiContext | None = None,
        *,
        stream: TextIO | None = None,
    ) -> None:
        super().__init__(context)
        self.stream = stream if stream is not None else sys.stderr

    @property
    def _debug(self) -> bool:
        return self.context.debug

    def _emit(self, text: str = "") -> None:
        print(text, file=self.stream)

    def session_started(self, *, task: str = "", resumed: bool = False) -> None:
        self._emit("NineCoder")
        self._emit(f"Workspace: {self.context.workspace}")
        self._emit(f"Model: {self.context.model}")
        self._emit(f"Permission: {self.context.permission}")
        if self.context.test_cmd:
            self._emit(f"Test: {self.context.test_cmd}")
        if resumed:
            self._emit(f"Resuming: {self.context.workspace}")
        self._emit()

    def session_finished(self, *, summary: str, steps: int, stopped_by: str) -> None:
        if stopped_by == "finished":
            self._emit("✓ Task completed")
        else:
            self._emit(f"✗ Stopped: {stopped_by}")
        if summary.strip():
            self._emit(f"  {summary.strip()}")

    def user_message(self, text: str) -> None:
        self._emit(f"> {text}")

    def assistant_text(self, text: str) -> None:
        for line in text.strip().splitlines():
            self._emit(line)

    def assistant_stream_chunk(self, chunk: str) -> None:
        print(chunk, end="", file=self.stream, flush=True)

    def session_history(self, messages: list[dict[str, Any]]) -> None:
        visible = [message for message in messages if message.get("role") != "system"]
        if not visible:
            return
        self._emit("Session history:")
        for message in visible:
            role = str(message.get("role", ""))
            content = str(message.get("content", "") or "")
            if role == "user":
                self._emit(f"> {content}")
            elif role == "assistant":
                if content.strip():
                    for line in content.strip().splitlines():
                        self._emit(line)
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    names = [
                        str(call.get("function", {}).get("name", "tool"))
                        for call in tool_calls
                        if isinstance(call, dict)
                    ]
                    self._emit(f"  Tool calls: {', '.join(names)}")
            elif role == "tool":
                name = str(message.get("name", "tool"))
                first = (content.strip().splitlines() or [""])[0]
                if len(first) > 100:
                    first = f"{first[:97]}..."
                self._emit(f"  {name}: {first}")
        self._emit()

    def model_started(self) -> None:
        return None

    def model_finished(self, elapsed: float, stop_reason: str) -> None:
        if self._debug:
            self._emit(f"[debug] model request: {elapsed:.2f}s (stop={stop_reason})")

    def tool_started(self, name: str, arguments: dict[str, Any]) -> None:
        self._emit(f"  ● {tool_summary(name, arguments)}")

    def tool_finished(self, name: str, arguments: dict[str, Any], result: Any) -> None:
        detail = tool_result_summary(name, arguments, result)
        label = tool_summary(name, arguments)
        self._emit(f"    ✓ {label}" + (f" — {detail}" if detail else ""))

    def tool_failed(self, name: str, arguments: dict[str, Any], message: str) -> None:
        label = tool_summary(name, arguments)
        self._emit(f"    ✗ {label}")
        for line in message.strip().splitlines()[:6]:
            self._emit(f"      {line}")

    def permission_requested(self, name: str, arguments: dict[str, Any], reason: str) -> bool:
        self._emit(f"Permission required: {name} ({reason})")
        self._emit(json.dumps(arguments, ensure_ascii=False, indent=2))
        try:
            reply = input("Allow once? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return reply in {"y", "yes"}

    def debug(self, message: str) -> None:
        if self._debug:
            self._emit(f"[debug] {message}")

    def info(self, text: str) -> None:
        for line in text.splitlines():
            self._emit(line)

    def error(self, message: str) -> None:
        self._emit(f"✗ {message}")

    def select_session(self, sessions: list[Any], head_id: str = "") -> str | None:
        if not sessions:
            return None
        for index, session in enumerate(sessions, 1):
            marker = "* " if getattr(session, "id", "") == head_id else "  "
            task = str(getattr(session, "task", "") or "(no task)").replace("\n", " ")
            if len(task) > 52:
                task = f"{task[:49]}..."
            self._emit(f"{marker}{index}. {getattr(session, 'id', '')}  {task}")
        try:
            reply = input("Resume session number or id: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not reply:
            return None
        if reply.isdigit():
            index = int(reply) - 1
            if 0 <= index < len(sessions):
                return str(getattr(sessions[index], "id", ""))
        for session in sessions:
            session_id = str(getattr(session, "id", ""))
            if session_id == reply:
                return session_id
        return None

    def prompt_input(self) -> str | None:
        try:
            return input("> ")
        except EOFError:
            return None

    def shutdown(self) -> None:
        return None
