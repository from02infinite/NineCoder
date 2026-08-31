import tempfile
import unittest

from ninecoder.permissions import PermissionMode
from ninecoder.tools import ToolRegistry
from ninecoder.workspace import Workspace


class TaskGraphTest(unittest.TestCase):
    def test_task_graph_reports_ready_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(Workspace(tmp), PermissionMode.PLAN)

            result = registry.execute(
                "update_task_graph",
                {
                    "tasks": [
                        {"id": "inspect", "content": "Inspect", "status": "done"},
                        {
                            "id": "edit",
                            "content": "Edit",
                            "status": "pending",
                            "depends_on": ["inspect"],
                        },
                    ]
                },
            )

            self.assertIn('"ready": [', result.content)
            self.assertIn('"edit"', result.content)


if __name__ == "__main__":
    unittest.main()
