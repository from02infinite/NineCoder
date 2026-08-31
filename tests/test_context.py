import tempfile
import unittest
from pathlib import Path

from ninecoder.context import ContextManager, compact_messages, summarize_messages


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
                {
                    "role": "assistant",
                    "content": "recent assistant",
                    "tool_calls": [{"id": "call_1"}],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "finish",
                    "content": "recent tool",
                },
            ]

            compacted = manager.compact_messages(messages)

            self.assertEqual(len(compacted), 5)
            self.assertIn("summary.md", compacted[2]["content"])
            self.assertIn("older assistant", (Path(tmp) / "runs" / "context" / "session-1" / "summary.md").read_text())

    def test_compaction_keeps_tool_messages_with_their_tool_calls(self) -> None:
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
                {"role": "assistant", "content": "old"},
                {
                    "role": "assistant",
                    "content": "read",
                    "tool_calls": [{"id": "call_1"}, {"id": "call_2"}],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "read_file",
                    "content": "file",
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "name": "search",
                    "content": "matches",
                },
            ]

            compacted = manager.compact_messages(messages)

        assistant_index = compacted.index(
            next(message for message in compacted if message.get("tool_calls"))
        )
        self.assertEqual(compacted[assistant_index + 1]["role"], "tool")
        self.assertEqual(compacted[assistant_index + 2]["role"], "tool")

    def test_compaction_drops_orphan_tool_messages(self) -> None:
        compacted = compact_messages(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "read_file",
                    "content": "orphan",
                },
            ]
        )

        self.assertNotIn("tool", [message["role"] for message in compacted])

    def test_summarize_messages_mentions_tool_call_count(self) -> None:
        summary = summarize_messages(
            [{"role": "assistant", "content": "thinking", "tool_calls": [{"id": "1"}]}]
        )

        self.assertIn("tool_calls=1", summary)


if __name__ == "__main__":
    unittest.main()
