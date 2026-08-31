import io
import unittest
from dataclasses import dataclass, field
from typing import Any

from ninecoder.tools import ToolResult
from ninecoder.ui import NullUI, PlainUI, UiContext, make_ui
from ninecoder.ui.components import (
    elapsed_human,
    format_args,
    permission_summary,
    tool_result_summary,
    tool_summary,
)


@dataclass(frozen=True)
class FakeResult:
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False
    terminate: bool = False


class ToolSummaryTest(unittest.TestCase):
    def test_read_file(self) -> None:
        self.assertEqual(tool_summary("read_file", {"path": "calculator.py"}), "Read calculator.py")

    def test_run_shell(self) -> None:
        self.assertEqual(tool_summary("run_shell", {"command": "pytest -q"}), "Bash pytest -q")

    def test_search_quotes_pattern(self) -> None:
        self.assertEqual(tool_summary("search", {"pattern": "AgentLoop"}), 'Search "AgentLoop"')

    def test_search_includes_path(self) -> None:
        self.assertEqual(
            tool_summary("search", {"pattern": "AgentLoop", "path": "src"}),
            'Search "AgentLoop" in src',
        )

    def test_unknown_tool_falls_back(self) -> None:
        self.assertEqual(tool_summary("frobnicate", {}), "Frobnicate")

    def test_finish(self) -> None:
        self.assertEqual(tool_summary("finish", {"summary": "all done"}), "Finish all done")


class ToolResultSummaryTest(unittest.TestCase):
    def test_run_shell_success_uses_output_tail(self) -> None:
        result = FakeResult(
            content='{"returncode": 0, "timed_out": false, "output": "4 passed in 0.05s\\n"}',
            metadata={"returncode": 0},
        )
        self.assertEqual(
            tool_result_summary("run_shell", {"command": "pytest"}, result),
            "4 passed in 0.05s",
        )

    def test_run_shell_failure_shows_exit_code(self) -> None:
        result = FakeResult(
            content='{"returncode": 1, "timed_out": false, "output": "1 failed"}',
            metadata={"returncode": 1},
        )
        self.assertEqual(
            tool_result_summary("run_shell", {"command": "pytest"}, result),
            "exit code 1",
        )

    def test_read_file_counts_lines(self) -> None:
        result = FakeResult(content="1: a\n2: b\n3: c\n")
        self.assertEqual(
            tool_result_summary("read_file", {"path": "a.py"}, result),
            "3 lines",
        )

    def test_edit_file_counts_changes(self) -> None:
        result = FakeResult(content="--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n")
        self.assertEqual(
            tool_result_summary("edit_file", {"path": "x.py"}, result),
            "changed 2 lines",
        )

    def test_search_no_matches(self) -> None:
        result = FakeResult(content="(no matches)")
        self.assertEqual(
            tool_result_summary("search", {"pattern": "zzz"}, result),
            "no matches",
        )

    def test_finish_returns_summary(self) -> None:
        result = ToolResult("fixed the bug", terminate=True)
        self.assertEqual(
            tool_result_summary("finish", {"summary": "fixed the bug"}, result),
            "fixed the bug",
        )


class FormatterTest(unittest.TestCase):
    def test_elapsed_human(self) -> None:
        self.assertEqual(elapsed_human(1.42), "1.42s")
        self.assertEqual(elapsed_human(0.003), "3ms")
        self.assertEqual(elapsed_human(0.0009), "900µs")

    def test_format_args(self) -> None:
        self.assertEqual(
            format_args({"path": "calculator.py", "offset": 1}),
            "path=calculator.py, offset=1",
        )
        self.assertEqual(format_args({}), "")

    def test_permission_summary_run_shell(self) -> None:
        self.assertEqual(
            permission_summary("run_shell", {"command": "rm -rf build"}),
            "$ rm -rf build",
        )


class UiFactoryTest(unittest.TestCase):
    def test_make_ui_null(self) -> None:
        self.assertIsInstance(make_ui(mode="null", context=UiContext()), NullUI)

    def test_make_ui_plain(self) -> None:
        self.assertIsInstance(make_ui(mode="plain", context=UiContext()), PlainUI)

    def test_null_ui_denies_permission(self) -> None:
        ui = NullUI()
        self.assertFalse(ui.permission_requested("run_shell", {"command": "rm"}, "changes state"))


class PlainUITest(unittest.TestCase):
    def test_debug_off_emits_nothing(self) -> None:
        stream = io.StringIO()
        ui = PlainUI(UiContext(debug=False), stream=stream)
        ui.debug("iteration=3")
        self.assertEqual(stream.getvalue(), "")

    def test_debug_on_emits_line(self) -> None:
        stream = io.StringIO()
        ui = PlainUI(UiContext(debug=True), stream=stream)
        ui.debug("iteration=3")
        self.assertEqual(stream.getvalue(), "[debug] iteration=3\n")

    def test_info_emits_lines(self) -> None:
        stream = io.StringIO()
        ui = PlainUI(UiContext(), stream=stream)
        ui.info("hello\nworld")
        self.assertEqual(stream.getvalue(), "hello\nworld\n")

    def test_stream_chunk_and_end(self) -> None:
        stream = io.StringIO()
        ui = PlainUI(UiContext(), stream=stream)
        ui.assistant_stream_chunk("Hello")
        ui.assistant_stream_chunk(" world")
        ui.assistant_stream_end()
        self.assertEqual(stream.getvalue(), "Hello world\n")


class InfoTest(unittest.TestCase):
    def test_null_ui_info_is_noop(self) -> None:
        # base AgentUI.info is a no-op; NullUI inherits it, so this must not raise.
        NullUI().info("ignored")


if __name__ == "__main__":
    unittest.main()
