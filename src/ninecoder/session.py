from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


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
    status: str = "running"
    stopped_by: str = ""
    summary: str = ""
    step: int = 0
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    messages: list[dict[str, Any]] = field(default_factory=list)
    todos: list[dict[str, str]] = field(default_factory=list)
    task_graph: list[dict[str, Any]] = field(default_factory=list)


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
    ) -> SessionState:
        state = SessionState(
            id=session_id or new_session_id(),
            task=task,
            workspace=workspace,
            permission_mode=permission_mode,
            messages=messages,
        )
        self.save(state)
        return state

    def load(self, session_id: str) -> SessionState:
        path = self.path_for(session_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return SessionState(**data)

    def save(self, state: SessionState) -> Path:
        state.updated_at = now_iso()
        path = self.path_for(state.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(state), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path
