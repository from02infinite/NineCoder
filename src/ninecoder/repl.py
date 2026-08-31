"""Pure helpers for the interactive REPL: command parsing and session-tree text.

These functions are deliberately free of terminal and model I/O so they can be
unit-tested in isolation.
"""
from __future__ import annotations

from ninecoder.session import SessionState, build_session_tree

_EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}

REPL_HELP = """\
Commands:
  <message>       Continue the current conversation
  /new            Start a brand-new conversation
  /resume [id]    Resume an existing session (pick one when id is omitted)
  /switch <id>    Branch from an existing session on your next message
  /compact        Compact the current conversation context
  /tree           Show the conversation tree
  /list           List all saved sessions
  /help           Show this help
  exit | quit     Leave the REPL"""


def parse_command(text: str) -> tuple[str, str] | None:
    """Classify a stripped input line.

    Returns ``(verb, arg)`` for a recognized command, ``None`` for a plain
    message, and ``("unknown", verb)`` for an unrecognized slash command.
    """
    if text in _EXIT_COMMANDS:
        return ("quit", "")
    if not text.startswith("/"):
        return None
    verb, _, arg = text.partition(" ")
    arg = arg.strip()
    if verb == "/new":
        return ("new", arg)
    if verb == "/resume":
        return ("resume", arg)
    if verb in ("/switch", "/checkout"):
        return ("switch", arg)
    if verb == "/compact":
        return ("compact", arg)
    if verb in ("/tree", "/branches"):
        return ("tree", arg)
    if verb in ("/list", "/sessions"):
        return ("list", arg)
    if verb in ("/help", "/?"):
        return ("help", arg)
    return ("unknown", verb)


def format_session_tree(sessions: list[SessionState], head_id: str = "") -> str:
    """Render the parent/child session tree, marking the current head."""
    if not sessions:
        return "(no sessions yet)"
    roots, children = build_session_tree(sessions)
    by_id = {session.id: session for session in sessions}
    lines = ["Conversation tree:"]
    for root_id in roots:
        _render_node(root_id, by_id, children, head_id, lines, depth=0)
    return "\n".join(lines)


def _render_node(
    session_id: str,
    by_id: dict[str, SessionState],
    children: dict[str, list[str]],
    head_id: str,
    lines: list[str],
    depth: int,
) -> None:
    session = by_id.get(session_id)
    if session is None:
        return
    marker = "→ " if session_id == head_id else ""
    child_count = len(children.get(session_id, []))
    branch = f"  ({child_count} branch)" if child_count else ""
    lines.append(f"{'    ' * depth}{marker}{_node_label(session)}{branch}")
    for child_id in children.get(session_id, []):
        _render_node(child_id, by_id, children, head_id, lines, depth + 1)


def format_session_list(sessions: list[SessionState], head_id: str = "") -> str:
    """Render a flat, chronological list of sessions, marking the head."""
    if not sessions:
        return "(no sessions yet)"
    lines = ["Sessions:"]
    for session in sessions:
        marker = "→ " if session.id == head_id else "  "
        lines.append(f"{marker}{session.id}  {session.status:8}  {_shorten(session.task)}")
    return "\n".join(lines)


def _node_label(session: SessionState) -> str:
    return f"{session.id}  {_shorten(session.task)}"


def _shorten(text: str, limit: int = 40) -> str:
    text = (text or "").strip().replace("\n", " ")
    if not text:
        return "(no task)"
    return text if len(text) <= limit else f"{text[: limit - 3]}..."
