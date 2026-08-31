"""Rich TUI backend.

Colored panels, Markdown rendering, a bottom status bar, and a
``prompt_toolkit``-backed REPL with history and basic multi-line input.

This module imports ``rich`` and ``prompt_toolkit`` eagerly, but it is only
loaded on demand by :func:`ninecoder.ui.make_ui`, so plain / JSON modes never
need those packages.

Dynamic text (tool summaries, model output, error messages) is rendered through
:class:`~rich.text.Text` segments rather than rich markup, so arbitrary content
such as ``[debug]`` or ``pytest [1]`` can never be parsed as style tags.
"""
from __future__ import annotations

import sys
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import radiolist_dialog
from prompt_toolkit.styles import Style

from ninecoder.ui.base import AgentUI, UiContext
from ninecoder.ui.components import (
    permission_summary,
    tool_result_summary,
    tool_summary,
)

_PROMPT = [("class:prompt", "> ")]

# Cap on how often the streaming Live region repaints, in seconds (~20 Hz).
# Keeps redraw cost flat regardless of token burst size; the final render on
# assistant_stream_end always repaints once more.
_STREAM_THROTTLE = 0.05

_STYLE = Style.from_dict(
    {
        "prompt": "bold cyan",
        "bottom-toolbar": "dim",
        "bottom-toolbar.status": "dim",
    }
)


def _stylized(text: str, style: str) -> Text:
    return Text(text, style=style)


class RichUI(AgentUI):
    renders_markdown = True

    def __init__(self, context: UiContext | None = None) -> None:
        super().__init__(context)
        self.console = Console(stderr=True, soft_wrap=True, highlight=False)
        self._stream_buffer: list[str] = []
        self._stream_live: Live | None = None
        self._stream_last_render = 0.0

        key_bindings = KeyBindings()

        @key_bindings.add("escape", "enter")
        def _(event: Any) -> None:
            # Alt+Enter inserts a newline, giving basic multi-line input while
            # plain Enter still submits.
            event.current_buffer.insert_text("\n")

        self._key_bindings = key_bindings
        self._session = PromptSession(
            history=InMemoryHistory(),
            key_bindings=key_bindings,
            style=_STYLE,
        )

    @property
    def _debug(self) -> bool:
        return self.context.debug

    # -- lifecycle ---------------------------------------------------------
    def session_started(self, *, task: str = "", resumed: bool = False) -> None:
        self.console.print(Text("NineCoder", style="bold cyan"))
        self.console.print(f"Workspace:  {self.context.workspace}", markup=False)
        self.console.print(f"Model:      {self.context.model}", markup=False)
        self.console.print(f"Permission: {self.context.permission}", markup=False)
        if self.context.sandbox:
            self.console.print(f"Sandbox:    {self.context.sandbox}", markup=False)
        if self.context.test_cmd:
            self.console.print(f"Test:       {self.context.test_cmd}", markup=False)
        self.console.print()
        if resumed:
            self.console.print(Text("✓ Resumed session", style="green"))
        else:
            self.console.print(Text("✓ New session started", style="green"))

    def session_finished(self, *, summary: str, steps: int, stopped_by: str) -> None:
        self.console.print(Rule(style="dim"))
        if stopped_by == "finished":
            self.console.print(Text("✓ Task completed", style="green"))
        else:
            self.console.print(Text(f"✗ Stopped: {stopped_by}", style="yellow"))
        if summary.strip():
            self.console.print(Markdown(summary.strip()))

    # -- conversation ------------------------------------------------------
    def user_message(self, text: str) -> None:
        self.console.print(Panel(text, border_style="cyan", padding=(0, 1)))

    def assistant_text(self, text: str) -> None:
        if not text.strip():
            return
        self.console.print(Markdown(text))

    def assistant_stream_chunk(self, chunk: str) -> None:
        # Render the accumulated text live instead of buffering until the
        # message finishes. Repainting is throttled (see _STREAM_THROTTLE) so a
        # burst of tokens collapses into a few redraws, and _render_streaming
        # degrades to plain Text while an unclosed fenced code block would
        # otherwise swallow the rest of the Markdown.
        self._stream_buffer.append(chunk)
        if self._stream_live is None:
            self._stream_live = Live(Text(""), console=self.console, auto_refresh=False)
            self._stream_live.start()
        now = time.monotonic()
        if now - self._stream_last_render >= _STREAM_THROTTLE:
            self._stream_last_render = now
            self._stream_live.update(_render_streaming(self._stream_buffer), refresh=True)

    def assistant_stream_end(self) -> None:
        text = "".join(self._stream_buffer)
        self._stream_buffer = []
        if self._stream_live is not None:
            self._stream_live.update(Markdown(text) if text.strip() else Text(""), refresh=True)
            self._stream_live.stop()
            self._stream_live = None
        elif text.strip():
            self.console.print(Markdown(text))

    def session_history(self, messages: list[dict[str, Any]]) -> None:
        visible = _visible_history(messages)
        if not visible:
            return
        self.console.print(Rule("Session history", style="dim"))
        for message in visible:
            role = str(message.get("role", ""))
            content = str(message.get("content", "") or "")
            if role == "user":
                self.console.print(Panel(content, title="You", border_style="cyan", padding=(0, 1)))
            elif role == "assistant":
                if content.strip():
                    self.console.print(Markdown(content))
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    names = [
                        str(call.get("function", {}).get("name", "tool"))
                        for call in tool_calls
                        if isinstance(call, dict)
                    ]
                    self.console.print(Text(f"Tool calls: {', '.join(names)}", style="dim"))
            elif role == "tool":
                name = str(message.get("name", "tool"))
                first = _first_line(content)
                self.console.print(Text(f"{name}: {first}", style="dim"))
        self.console.print(Rule(style="dim"))

    # -- model -------------------------------------------------------------
    def model_started(self) -> None:
        return None

    def model_finished(self, elapsed: float, stop_reason: str) -> None:
        if self._debug:
            self.console.print(
                Text(f"    model request: {elapsed:.2f}s (stop={stop_reason})", style="dim")
            )

    # -- tools -------------------------------------------------------------
    def tool_started(self, name: str, arguments: dict[str, Any]) -> None:
        self.console.print(Text(f"◐ {tool_summary(name, arguments)}", style="dim"))

    def tool_finished(self, name: str, arguments: dict[str, Any], result: Any) -> None:
        detail = tool_result_summary(name, arguments, result)
        label = tool_summary(name, arguments)
        line = Text("✓ ", style="green")
        line.append(label)
        if detail:
            line.append("  └─ ", style="dim")
            line.append(detail, style="dim")
        self.console.print(line)

    def tool_failed(self, name: str, arguments: dict[str, Any], message: str) -> None:
        label = tool_summary(name, arguments)
        first = (message.strip().splitlines() or [""])[0]
        line = Text("✗ ", style="red")
        line.append(label)
        if first:
            line.append("  └─ ", style="dim")
            line.append(first, style="dim")
        self.console.print(line)

    # -- permission --------------------------------------------------------
    def permission_requested(self, name: str, arguments: dict[str, Any], reason: str) -> bool:
        body = Text(permission_summary(name, arguments))
        body.append("\n\n")
        body.append(reason, style="yellow")
        self.console.print(
            Panel(body, title="Permission required", border_style="yellow", padding=(0, 1))
        )
        self.console.print("Allow once? [y/N] ", end="", markup=False)
        try:
            reply = sys.stdin.readline().strip().lower()
        except (EOFError, KeyboardInterrupt):
            reply = ""
        return reply in {"y", "yes"}

    # -- diagnostics -------------------------------------------------------
    def debug(self, message: str) -> None:
        if self._debug:
            self.console.print(Text(f"[debug] {message}", style="dim"))

    def info(self, text: str) -> None:
        for line in text.splitlines():
            self.console.print(line, markup=False)

    def error(self, message: str) -> None:
        for line in message.splitlines():
            self.console.print(Text(f"✗ {line}", style="red"))

    # -- interactive input -------------------------------------------------
    def select_session(self, sessions: list[Any], head_id: str = "") -> str | None:
        if not sessions:
            return None
        values = [
            (
                str(getattr(session, "id", "")),
                _session_label(session, head_id),
            )
            for session in sessions
        ]
        try:
            selected = radiolist_dialog(
                title="Resume session",
                text="Choose a saved session, then press Enter.",
                values=values,
            ).run()
        except (EOFError, KeyboardInterrupt):
            return None
        return str(selected) if selected else None

    def prompt_input(self) -> str | None:
        while True:
            try:
                return self._session.prompt(
                    _PROMPT,
                    bottom_toolbar=self._status_toolbar,
                )
            except KeyboardInterrupt:
                self.console.print(Text("Interrupted (Ctrl-D to exit)", style="dim"))
                continue
            except EOFError:
                return None

    def _status_toolbar(self) -> Any:
        status = (
            f"{self.context.workspace}   "
            f"{self.context.model} · {self.context.permission}"
        )
        # prompt_toolkit wants its own formatted text, not a rich Text object.
        return [("class:bottom-toolbar", status)]

    def shutdown(self) -> None:
        return None


def _session_label(session: Any, head_id: str) -> str:
    session_id = str(getattr(session, "id", ""))
    status = str(getattr(session, "status", "") or "unknown")
    task = str(getattr(session, "task", "") or "(no task)").replace("\n", " ").strip()
    if len(task) > 52:
        task = f"{task[:49]}..."
    marker = "* " if session_id == head_id else ""
    return f"{marker}{session_id}  {status:8}  {task}"


def _visible_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [message for message in messages if message.get("role") != "system"]


def _first_line(text: str, limit: int = 100) -> str:
    first = (text.strip().splitlines() or [""])[0]
    return first if len(first) <= limit else f"{first[: limit - 3]}..."


def _render_streaming(buffer: list[str]) -> Markdown | Text:
    """Render the in-flight stream text, degrading to plain ``Text`` while an
    unclosed fenced code block would make ``Markdown`` swallow the tail."""
    text = "".join(buffer)
    if _in_unclosed_fence(text):
        return Text(text)
    return Markdown(text)


def _in_unclosed_fence(text: str) -> bool:
    """True if ``text`` ends inside an unclosed ```/~~~ fenced code block."""
    fence: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if fence is None:
            if stripped.startswith("```") or stripped.startswith("~~~"):
                fence = stripped[:3]
        elif stripped.startswith(fence):
            fence = None
    return fence is not None
