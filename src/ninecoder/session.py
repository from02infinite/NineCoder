from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ninecoder.persistence import atomic_write_text


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


@dataclass
class SessionState:
    id: str
    task: str
    workspace: str
    permission_mode: str
    parent_id: str = ""
    status: str = "running"
    stopped_by: str = ""
    summary: str = ""
    step: int = 0
    compaction_floor: int = 0
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    messages: list[dict[str, Any]] = field(default_factory=list)
    todos: list[dict[str, str]] = field(default_factory=list)
    task_graph: list[dict[str, Any]] = field(default_factory=list)
    subagent_tasks: list[dict[str, Any]] = field(default_factory=list)


class SessionStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        return self.root / session_id / "session.json"

    def create(
        self,
        task: str,
        workspace: str,
        permission_mode: str,
        messages: list[dict[str, Any]],
        session_id: str | None = None,
        parent_id: str = "",
        compaction_floor: int = 0,
    ) -> SessionState:
        state = SessionState(
            id=session_id or new_session_id(),
            task=task,
            workspace=workspace,
            permission_mode=permission_mode,
            parent_id=parent_id,
            compaction_floor=compaction_floor,
            messages=messages,
        )
        self.save(state)
        return state

    def load(self, session_id: str) -> SessionState:
        path = self.path_for(session_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return SessionState(**data)

    def list(self) -> list[SessionState]:
        """Load every saved session, skipping any that fail to parse."""
        sessions: list[SessionState] = []
        if not self.root.exists():
            return sessions
        for session_dir in sorted(self.root.iterdir()):
            if not session_dir.is_dir():
                continue
            path = session_dir / "session.json"
            if not path.exists():
                continue
            try:
                sessions.append(SessionState(**json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError, TypeError):
                continue
        return sessions

    def save(self, state: SessionState) -> Path:
        state.updated_at = now_iso()
        path = self.path_for(state.id)
        return atomic_write_text(
            path,
            json.dumps(asdict(state), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def rewind_messages(self, session_id: str, message_count: int) -> SessionState:
        state = self.load(session_id)
        if message_count < state.compaction_floor:
            raise ValueError(
                "cannot rewind before compaction_floor="
                f"{state.compaction_floor}"
            )
        state.messages = state.messages[:message_count]
        self.save(state)
        return state


def build_session_tree(
    sessions: list[SessionState],
) -> tuple[list[str], dict[str, list[str]]]:
    """Return (root_ids, children) for a list of sessions.

    ``children`` maps a parent session id to its child ids in insertion order.
    Sessions whose parent is missing are treated as roots so a partial tree
    still renders.
    """
    ids = {session.id for session in sessions}
    children: dict[str, list[str]] = {}
    roots: list[str] = []
    for session in sessions:
        parent = session.parent_id
        if parent and parent in ids:
            children.setdefault(parent, []).append(session.id)
        else:
            roots.append(session.id)
    return roots, children
