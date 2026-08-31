import tempfile
import unittest
from pathlib import Path

from ninecoder.errors import PermissionDenied, ToolError
from ninecoder.workspace import Workspace


class WorkspaceTest(unittest.TestCase):
    def test_read_file_uses_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            workspace = Workspace(root)

            self.assertIn("2: two", workspace.read_file("a.txt", offset=2, limit=1))

    def test_edit_file_returns_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("print('bad')\n", encoding="utf-8")
            workspace = Workspace(root)

            diff = workspace.edit_file("a.py", "bad", "good")

            self.assertIn("+print('good')", diff)

    def test_edit_file_reports_missing_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("print('hello')\n", encoding="utf-8")
            workspace = Workspace(root)

            with self.assertRaises(ToolError):
                workspace.edit_file("a.py", "missing", "value")

    def test_paths_cannot_escape_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(tmp)

            with self.assertRaises(PermissionDenied):
                workspace.resolve("../outside.txt")


if __name__ == "__main__":
    unittest.main()
