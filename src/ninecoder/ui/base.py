"""Unified UI abstraction for NineCoder.

The agent core never writes to the terminal directly. Instead it emits events by
calling methods on an :class:`AgentUI`. Concrete backends (plain text, rich TUI,
or nothing at all) render those events however they like. This keeps the agent
runtime independent of any particular display, so the same loop can drive a TUI,
a CI log, or a JSON-only report.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UiContext:
    """Static facts about a run that the UI shows but the agent does not know."""

    model: str = ""
    workspace: str = ""
    permission: str = "ask"
    test_cmd: str = ""
    debug: bool = False
    sandbox: str = ""


class AgentUI:
    """Event sink for the agent core.

    The default implementation is intentionally quiet (a no-op for display
    events). It only prompts the terminal when a permission decision requires a
    human, preserving the original interactive behavior for library users who
    construct a :class:`~ninecoder.agent.CodingAgent` directly.
    """

    def __init__(self, context: UiContext | None = None) -> None:
        self.context = context or UiContext()

    # -- lifecycle ---------------------------------------------------------
    def session_started(self, *, task: str = "", resumed: bool = False) -> None:
        """A new session is beginning (header / banner)."""

    def session_finished(self, *, summary: str, steps: int, stopped_by: str) -> None:
        """The agent stopped: finished, max_steps, or a format error."""

    # -- conversation ------------------------------------------------------
    def user_message(self, text: str) -> None:
        """The user's task text, shown as a highlighted block."""

    def assistant_text(self, text: str) -> None:
        """A completed assistant text segment (Markdown-capable)."""

    def assistant_stream_chunk(self, chunk: str) -> None:
        """A streaming token. Reserved for future streaming support."""

    def session_history(self, messages: list[dict[str, Any]]) -> None:
        """Show messages loaded from a resumed session."""

    # -- model -------------------------------------------------------------
    def model_started(self) -> None:
        """A model request is about to be issued."""

    def model_finished(self, elapsed: float, stop_reason: str) -> None:
        """A model request completed."""

    # -- tools -------------------------------------------------------------
    def tool_started(self, name: str, arguments: dict[str, Any]) -> None:
        """A tool call is about to execute."""

    def tool_finished(self, name: str, arguments: dict[str, Any], result: Any) -> None:
        """A tool call completed successfully."""

    def tool_failed(self, name: str, arguments: dict[str, Any], message: str) -> None:
        """A tool call returned an error result."""

    # -- permission --------------------------------------------------------
    def permission_requested(self, name: str, arguments: dict[str, Any], reason: str) -> bool:
        """Ask the user whether to allow a mutating tool call.

        Returns True to allow, False to deny. The default implementation is a
        plain terminal prompt, matching the pre-UI behavior.
        """
        print(f"\nPermission required: {name} ({reason})")
        print(json.dumps(arguments, ensure_ascii=False, indent=2))
        try:
            reply = input("Allow once? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return reply in {"y", "yes"}

    # -- diagnostics -------------------------------------------------------
    def debug(self, message: str) -> None:
        """Internal diagnostic line (only shown in debug mode)."""

    def info(self, text: str) -> None:
        """A user-visible informational line (command output such as /tree)."""

    def error(self, message: str) -> None:
        """A user-visible error."""

    # -- interactive input -------------------------------------------------
    def select_session(self, sessions: list[Any], head_id: str = "") -> str | None:
        """Let the user choose a saved session id."""
        if not sessions:
            return None
        for index, session in enumerate(sessions, 1):
            marker = "* " if getattr(session, "id", "") == head_id else "  "
            task = str(getattr(session, "task", "") or "(no task)").replace("\n", " ")
            print(f"{marker}{index}. {getattr(session, 'id', '')}  {task}")
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
        """Read the next user message. Returns None on EOF (Ctrl-D)."""
        try:
            return input("> ")
        except EOFError:
            return None

    def shutdown(self) -> None:
        """Clean up any terminal state before exiting."""


class NullUI(AgentUI):
    """Silent backend for ``--json`` / ``--quiet``.

    Emits nothing and never blocks: interactive permission requests are denied
    (there is no human to ask), so pair it with ``--permission auto`` or
    ``--permission plan`` for unattended runs.
    """

    def permission_requested(self, name: str, arguments: dict[str, Any], reason: str) -> bool:
        return False

    def prompt_input(self) -> None:
        return None
