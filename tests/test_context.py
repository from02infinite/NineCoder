import tempfile
import unittest
from pathlib import Path

from ninecoder.context import ContextManager, summarize_messages


class ContextManagerTest(unittest.TestCase):
    def test_large_tool_result_is_saved_with_model_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = ContextManager(
                tmp,
                "runs",
                "session-1",
                max_tool_chars=20,
            )
            result = manager.store_tool_result("call/1", "run_shell", "x" * 100)

            self.assertIsNotNone(result.path)
            self.assertIn("Full output saved at: runs/context/session-1/tool-results", result.content_for_model)
            self.assertEqual(Path(result.path).read_text(encoding="utf-8"), "x" * 100)

    def test_compaction_writes_summary_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = ContextManager(
                tmp,
                "runs",
                "session-1",
                keep_recent_messages=2,
            )
            messages = [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "older assistant"},
                {"role": "tool", "name": "read_file", "content": "older tool"},
                {"role": "assistant", "content": "recent assistant"},
                {"role": "tool", "name": "finish", "content": "recent tool"},
            ]

            compacted = manager.compact_messages(messages)

            self.assertEqual(len(compacted), 5)
            self.assertIn("summary.md", compacted[2]["content"])
            self.assertIn("older assistant", (Path(tmp) / "runs" / "context" / "session-1" / "summary.md").read_text())

    def test_summarize_messages_mentions_tool_call_count(self) -> None:
        summary = summarize_messages(
            [{"role": "assistant", "content": "thinking", "tool_calls": [{"id": "1"}]}]
        )

        self.assertIn("tool_calls=1", summary)


if __name__ == "__main__":
    unittest.main()
