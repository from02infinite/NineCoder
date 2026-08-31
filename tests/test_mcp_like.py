import tempfile
import unittest

from ninecoder.mcp_like import LocalCapabilityRouter
from ninecoder.permissions import PermissionMode
from ninecoder.tools import ToolRegistry
from ninecoder.workspace import Workspace


class McpLikeTest(unittest.TestCase):
    def test_list_and_call_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            router = LocalCapabilityRouter(ToolRegistry(Workspace(tmp), PermissionMode.AUTO))

            listed = router.handle("tools/list")
            called = router.handle("tools/call", {"name": "list_files", "arguments": {"path": "."}})

            self.assertIn("tools", listed)
            self.assertFalse(called["is_error"])

    def test_invalid_arguments_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            router = LocalCapabilityRouter(ToolRegistry(Workspace(tmp), PermissionMode.AUTO))

            called = router.handle("tools/call", {"name": "read_file", "arguments": {}})

            self.assertTrue(called["is_error"])
            self.assertIn("missing required", called["content"])


if __name__ == "__main__":
    unittest.main()
