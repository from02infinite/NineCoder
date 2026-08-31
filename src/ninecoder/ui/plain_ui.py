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

    def error(self, message: str) -> None:
        self._emit(f"✗ {message}")

    def prompt_input(self) -> str | None:
        try:
            return input("> ")
        except EOFError:
            return None

    def shutdown(self) -> None:
        return None
