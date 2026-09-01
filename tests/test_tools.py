import tempfile
import unittest
from pathlib import Path
from typing import Any

from ninecoder.permissions import GrantDecision, PermissionMode
from ninecoder.tools import ToolRegistry
from ninecoder.workspace import Workspace


class _GrantUI:
    """A UI stub that returns scripted grant decisions and records prompts."""

    def __init__(self, decisions: list[GrantDecision]) -> None:
        self.decisions = list(decisions)
        self.prompted: list[tuple[str, dict[str, Any]]] = []

    def debug(self, message: str) -> None:
        pass

    def permission_requested(
        self, name: str, arguments: dict[str, Any], reason: str
    ) -> GrantDecision:
        self.prompted.append((name, arguments))
        return self.decisions.pop(0)


class ToolRegistryTest(unittest.TestCase):
    def test_plan_mode_denies_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(Workspace(tmp), PermissionMode.PLAN)

            result = registry.execute("write_file", {"path": "x.txt", "content": "x"})

            self.assertTrue(result.is_error)

    def test_auto_mode_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(Workspace(tmp), PermissionMode.AUTO)

            result = registry.execute("write_file", {"path": "x.txt", "content": "x"})

            self.assertFalse(result.is_error)
            self.assertEqual((Path(tmp) / "x.txt").read_text(encoding="utf-8"), "x")

    def test_ask_allow_always_remembers_grant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui = _GrantUI([GrantDecision.ALLOW_ALWAYS])
            registry = ToolRegistry(Workspace(tmp), PermissionMode.ASK, ui=ui)

            first = registry.execute("write_file", {"path": "a.txt", "content": "1"})
            second = registry.execute("write_file", {"path": "a.txt", "content": "2"})

            self.assertFalse(first.is_error)
            self.assertFalse(second.is_error)
            self.assertEqual(len(ui.prompted), 1)

    def test_ask_allow_once_prompts_every_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui = _GrantUI([GrantDecision.ALLOW_ONCE, GrantDecision.ALLOW_ONCE])
            registry = ToolRegistry(Workspace(tmp), PermissionMode.ASK, ui=ui)

            registry.execute("write_file", {"path": "a.txt", "content": "1"})
            registry.execute("write_file", {"path": "a.txt", "content": "2"})

            self.assertEqual(len(ui.prompted), 2)

    def test_ask_deny_blocks_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui = _GrantUI([GrantDecision.DENY])
            registry = ToolRegistry(Workspace(tmp), PermissionMode.ASK, ui=ui)

            result = registry.execute("write_file", {"path": "a.txt", "content": "1"})

            self.assertTrue(result.is_error)
            self.assertFalse((Path(tmp) / "a.txt").exists())

    def test_grant_key_uses_exact_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(Workspace(tmp), PermissionMode.ASK)

            status = registry._grant_key("run_shell", {"command": "git status"})
            diff = registry._grant_key("run_shell", {"command": "git diff"})

            self.assertEqual(status, "run_shell:git status")
            self.assertEqual(diff, "run_shell:git diff")
            self.assertNotEqual(status, diff)

    def test_grant_key_uses_target_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(Workspace(tmp), PermissionMode.ASK)

            key = registry._grant_key("edit_file", {"path": "src/a.py"})

            self.assertEqual(key, "edit_file:src/a.py")


if __name__ == "__main__":
    unittest.main()
