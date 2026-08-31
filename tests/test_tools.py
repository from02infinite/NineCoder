import tempfile
import unittest
from pathlib import Path

from ninecoder.permissions import PermissionMode
from ninecoder.tools import ToolRegistry
from ninecoder.workspace import Workspace


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


if __name__ == "__main__":
    unittest.main()
