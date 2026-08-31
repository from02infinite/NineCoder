import unittest

from ninecoder.repl import (
    REPL_HELP,
    format_session_list,
    format_session_tree,
    parse_command,
)
from ninecoder.session import SessionState


def _session(session_id: str, task: str, parent_id: str = "") -> SessionState:
    return SessionState(
        id=session_id, task=task, workspace="", permission_mode="", parent_id=parent_id
    )


class ParseCommandTest(unittest.TestCase):
    def test_plain_text_is_not_a_command(self) -> None:
        self.assertIsNone(parse_command("fix the bug"))

    def test_exit_words(self) -> None:
        for word in ("exit", "quit", "/exit", "/quit"):
            self.assertEqual(parse_command(word), ("quit", ""))

    def test_new(self) -> None:
        self.assertEqual(parse_command("/new"), ("new", ""))

    def test_resume(self) -> None:
        self.assertEqual(parse_command("/resume"), ("resume", ""))
        self.assertEqual(parse_command("/resume abc123"), ("resume", "abc123"))

    def test_switch(self) -> None:
        self.assertEqual(parse_command("/switch abc123"), ("switch", "abc123"))

    def test_checkout_is_alias_for_switch(self) -> None:
        self.assertEqual(parse_command("/checkout abc"), ("switch", "abc"))

    def test_tree_list_help(self) -> None:
        self.assertEqual(parse_command("/tree"), ("tree", ""))
        self.assertEqual(parse_command("/branches"), ("tree", ""))
        self.assertEqual(parse_command("/list"), ("list", ""))
        self.assertEqual(parse_command("/sessions"), ("list", ""))
        self.assertEqual(parse_command("/help"), ("help", ""))

    def test_compact(self) -> None:
        self.assertEqual(parse_command("/compact"), ("compact", ""))

    def test_unknown_command(self) -> None:
        self.assertEqual(parse_command("/frobnicate"), ("unknown", "/frobnicate"))

    def test_help_text_mentions_commands(self) -> None:
        for command in ("/new", "/resume", "/switch", "/compact", "/tree", "/list", "/help"):
            self.assertIn(command, REPL_HELP)


class FormatTest(unittest.TestCase):
    def test_empty_tree(self) -> None:
        self.assertEqual(format_session_tree([]), "(no sessions yet)")

    def test_empty_list(self) -> None:
        self.assertEqual(format_session_list([]), "(no sessions yet)")

    def test_tree_marks_head(self) -> None:
        sessions = [_session("root", "task a"), _session("child", "task b", parent_id="root")]
        out = format_session_tree(sessions, head_id="child")
        self.assertIn("Conversation tree:", out)
        self.assertIn("root", out)
        self.assertIn("child", out)
        self.assertIn("→ child", out)

    def test_list_marks_head(self) -> None:
        sessions = [_session("root", "task a"), _session("child", "task b", parent_id="root")]
        out = format_session_list(sessions, head_id="root")
        self.assertIn("Sessions:", out)
        self.assertIn("→ root", out)
        self.assertNotIn("→ child", out)


if __name__ == "__main__":
    unittest.main()
