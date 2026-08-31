import io
import json
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from ninecoder import cli
from ninecoder.config import ModelConfig


@dataclass(frozen=True)
class FakeRun:
    summary: str
    steps: int
    trajectory_path: Path
    session_id: str
    session_path: Path
    stopped_by: str


class FakeAgent:
    def __init__(
        self,
        model: Any,
        workspace: Any,
        config: Any,
        hooks: list[Any] | None = None,
        ui: Any | None = None,
    ) -> None:
        self.workspace = workspace

    def run(self, task: str) -> FakeRun:
        return FakeRun(
            summary=f"handled {task}",
            steps=2,
            trajectory_path=Path(self.workspace.root) / "runs" / "session-1.jsonl",
            session_id="session-1",
            session_path=Path(self.workspace.root) / "runs" / "sessions" / "session-1.json",
            stopped_by="finished",
        )


class CliTest(unittest.TestCase):
    def test_help_groups_options_and_examples(self) -> None:
        parser = cli.build_parser()

        help_text = parser.format_help()

        self.assertIn("run options", help_text)
        self.assertIn("model options", help_text)
        self.assertIn("output options", help_text)
        self.assertIn("examples:", help_text)

    def test_main_prints_clean_human_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patched_cli(), redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = cli.main(["--workspace", tmp, "fix it"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Done", stdout.getvalue())
        self.assertIn("handled fix it", stdout.getvalue())
        self.assertIn("Session: session-1", stdout.getvalue())
        self.assertIn("Workspace:", stderr.getvalue())

    def test_json_mode_suppresses_progress_and_returns_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patched_cli(), redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = cli.main(["--workspace", tmp, "--json", "fix it"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["ok"])
        self.assertEqual(report["session_path"], "runs/sessions/session-1.json")


@contextmanager
def patched_cli() -> Iterator[None]:
    with (
        patch.object(cli.ModelConfig, "from_env", return_value=model_config()),
        patch.object(cli, "ModelClient", return_value=object()),
        patch.object(cli, "CodingAgent", FakeAgent),
    ):
        yield


def model_config() -> ModelConfig:
    return ModelConfig(
        model="test-model",
        base_url="https://example.test/v1",
        api_key="test-key",
        temperature=0.2,
        max_tokens=4096,
    )


if __name__ == "__main__":
    unittest.main()
