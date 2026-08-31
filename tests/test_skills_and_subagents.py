import tempfile
import unittest
from pathlib import Path
from typing import Any

from ninecoder.model_client import ModelResponse
from ninecoder.permissions import PermissionMode
from ninecoder.skills import SkillLibrary
from ninecoder.tools import ToolRegistry
from ninecoder.workspace import Workspace


class TextModel:
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        return ModelResponse("subagent advice")


class SkillAndSubagentTest(unittest.TestCase):
    def test_load_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "skills"
            skill_dir.mkdir()
            (skill_dir / "demo.md").write_text("# Demo\nUse care.", encoding="utf-8")
            registry = ToolRegistry(
                Workspace(root / "work"),
                PermissionMode.AUTO,
                skill_library=SkillLibrary(skill_dir),
            )

            result = registry.execute("load_skill", {"name": "demo"})

            self.assertIn("Loaded skill: demo", result.content)

    def test_spawn_subagent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(
                Workspace(tmp),
                PermissionMode.AUTO,
                model=TextModel(),
            )

            result = registry.execute(
                "spawn_subagent",
                {"role": "reviewer", "prompt": "review this"},
            )

            self.assertEqual(result.content, "subagent advice")


if __name__ == "__main__":
    unittest.main()
