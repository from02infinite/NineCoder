import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from ninecoder.agent import AgentConfig, CodingAgent
from ninecoder.model_client import ModelResponse, ToolCall
from ninecoder.permissions import PermissionMode
from ninecoder.prompts import skills_block
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


class CaptureModel:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.messages = messages
        return ModelResponse("finish", [ToolCall("call_1", "finish", {"summary": "ok"})])


class SkillAndSubagentTest(unittest.TestCase):
    def test_skills_block_lists_names(self) -> None:
        self.assertEqual(skills_block([]), "")
        self.assertIn("- reviewer", skills_block(["reviewer", "python-debugging"]))

    def test_agent_injects_available_skills_into_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "skills"
            skill_dir.mkdir()
            (skill_dir / "demo.md").write_text("# Demo\nUse care.", encoding="utf-8")

            model = CaptureModel()
            CodingAgent(
                model,
                Workspace(root / "work"),
                AgentConfig(permission_mode=PermissionMode.AUTO, memory=False),
                skill_library=SkillLibrary(skill_dir),
            ).run("finish")

            system = model.messages[0]["content"]
            self.assertIn("## Available skills", system)
            self.assertIn("- demo", system)

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

    def test_subagent_task_can_be_started_read_and_listed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(
                Workspace(tmp),
                PermissionMode.AUTO,
                model=TextModel(),
            )

            started = registry.execute(
                "start_subagent_task",
                {"role": "planner", "prompt": "plan"},
            )
            listed = registry.execute("list_subagent_tasks", {})
            read = registry.execute("read_subagent_task", {"task_id": started.metadata["task_id"]})

            self.assertEqual(json.loads(started.content)["status"], "completed")
            self.assertEqual(json.loads(read.content)["result"], "subagent advice")
            self.assertEqual(json.loads(listed.content)[0]["id"], started.metadata["task_id"])


if __name__ == "__main__":
    unittest.main()
