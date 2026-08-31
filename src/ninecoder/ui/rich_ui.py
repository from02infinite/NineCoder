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
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from ninecoder.ui.base import AgentUI, UiContext
from ninecoder.ui.components import (
    permission_summary,
    tool_result_summary,
    tool_summary,
)

_PROMPT = [("class:prompt", "> ")]

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
    def __init__(self, context: UiContext | None = None) -> None:
        super().__init__(context)
        self.console = Console(stderr=True, soft_wrap=True, highlight=False)

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
            self.console.print(summary.strip(), markup=False)

    # -- conversation ------------------------------------------------------
    def user_message(self, text: str) -> None:
        self.console.print(Panel(text, border_style="cyan", padding=(0, 1)))

    def assistant_text(self, text: str) -> None:
        if not text.strip():
            return
        self.console.print(Markdown(text))

    def assistant_stream_chunk(self, chunk: str) -> None:
        # Reserved: future streaming support will append to the same block.
        self.console.print(chunk, end="", markup=False)

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

    def error(self, message: str) -> None:
        for line in message.splitlines():
            self.console.print(Text(f"✗ {line}", style="red"))

    # -- interactive input -------------------------------------------------
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
