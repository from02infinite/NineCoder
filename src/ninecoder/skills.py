from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    content: str


class SkillLibrary:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else Path(__file__).resolve().parents[2] / "skills"

    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(path.stem for path in self.root.glob("*.md"))

    def load(self, name: str) -> Skill:
        safe = name.strip().replace("/", "-").replace("\\", "-")
        path = self.root / f"{safe}.md"
        if not path.exists():
            available = ", ".join(self.names()) or "(none)"
            raise FileNotFoundError(f"skill not found: {name}. Available: {available}")
        return Skill(safe, path.read_text(encoding="utf-8"))
