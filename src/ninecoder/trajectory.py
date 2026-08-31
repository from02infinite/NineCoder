from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class Trajectory:
    def __init__(self, root: str | Path, run_name: str | None = None):
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = self.dir / f"{name}.jsonl"

    def write(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "payload": _jsonable(payload),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
