from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from ninecoder.agent import AgentConfig, CodingAgent
from ninecoder.config import ModelConfig
from ninecoder.hooks import ModelRequest, ToolRequest
from ninecoder.model_client import ModelClient
from ninecoder.permissions import PermissionMode
from ninecoder.workspace import Workspace


EXAMPLES = """examples:
  ninecoder "Fix the failing tests"
  ninecoder -w demo --permission auto \\
    --test "python -m unittest -q" "Make tests pass"
  ninecoder -w demo --resume 20260831-120000-000000
"""


class NineCoderHelpFormatter(argparse.RawDescriptionHelpFormatter):
    pass


class NineCoderParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\nTry '{self.prog} --help'.\n")


class CliProgressHook:
    def __init__(self, *, stream: TextIO, quiet: bool = False):
        self.stream = stream
        self.quiet = quiet
        self.step = 0

    def before_model(self, request: ModelRequest) -> None:
        if self.quiet:
            return None
        self.step += 1
        print(f"[{self.step}] Asking the model...", file=self.stream)
        return None

    def after_model(self, response: object) -> None:
        if self.quiet:
            return None
        tool_calls = getattr(response, "tool_calls", [])
        content = str(getattr(response, "content", "") or "").strip()
        if content:
            print(f"    {first_line(content)}", file=self.stream)
        if tool_calls:
            names = ", ".join(getattr(call, "name", "tool") for call in tool_calls)
            print(f"    Tools: {names}", file=self.stream)
        return None

    def before_tool(self, request: ToolRequest) -> None:
        if not self.quiet:
            target = tool_target(request.arguments)
            suffix = f" ({target})" if target else ""
            print(f"    Running {request.name}{suffix}", file=self.stream)
        return None

    def after_tool(self, name: str, result: str, is_error: bool) -> None:
        if not self.quiet:
            status = "failed" if is_error else "ok"
            detail = f": {first_line(result)}" if is_error and result.strip() else ""
            print(f"    {name} {status}{detail}", file=self.stream)
        return None

    def on_finish(self, summary: str) -> None:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = NineCoderParser(
        prog="ninecoder",
        description="Run the NineCoder local coding agent.",
        epilog=EXAMPLES,
        formatter_class=NineCoderHelpFormatter,
    )
    parser.add_argument(
        "task",
        nargs="*",
        metavar="TASK",
        help="Natural-language coding task",
    )

    run_options = parser.add_argument_group("run options")
    run_options.add_argument(
        "-w",
        "--workspace",
        default=".",
        help="Workspace directory (default: .)",
    )
    run_options.add_argument(
        "--permission",
        choices=[mode.value for mode in PermissionMode],
        default=PermissionMode.ASK.value,
        metavar="MODE",
        help=(
            "Tool permission mode: plan blocks writes, ask prompts, "
            "auto allows workspace changes (default: ask)"
        ),
    )
    run_options.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum agent loop steps (default: 30)",
    )
    run_options.add_argument(
        "--resume",
        "--resume-session",
        dest="resume_session",
        default="",
        help="Resume a saved session id",
    )
    run_options.add_argument(
        "--test",
        "--test-cmd",
        dest="test_cmd",
        default="",
        help="Verification command the agent should run before finishing",
    )

    model_options = parser.add_argument_group("model options")
    model_options.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model name (env: NINECODER_MODEL)",
    )
    model_options.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL",
    )
    model_options.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    model_options.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum response tokens (default: 4096)",
    )

    output_options = parser.add_argument_group("output options")
    output_options.add_argument(
        "--quiet",
        action="store_true",
        help="Hide live progress output",
    )
    output_options.add_argument(
        "--json",
        action="store_true",
        help="Print the final report as JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.task and not args.resume_session:
        parser.error("task is required unless --resume-session is provided")
    try:
        model_config = ModelConfig.from_env(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        workspace = Workspace(Path(args.workspace))
        agent = CodingAgent(
            ModelClient(model_config),
            workspace,
            AgentConfig(
                max_steps=args.max_steps,
                permission_mode=PermissionMode(args.permission),
                test_cmd=args.test_cmd,
                resume_session=args.resume_session,
            ),
            hooks=[CliProgressHook(stream=sys.stderr, quiet=args.quiet or args.json)],
        )
        if not args.quiet and not args.json:
            print_start(args, model_config.model, workspace, file=sys.stderr)
        result = agent.run(" ".join(args.task))
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(final_report(result, workspace), ensure_ascii=False, indent=2))
    else:
        print_human_report(result, workspace)
    return 0 if result.stopped_by == "finished" else 1


def print_start(args: argparse.Namespace, model: str, workspace: Workspace, *, file: TextIO) -> None:
    task = " ".join(args.task).strip()
    print("NineCoder", file=file)
    print(f"Workspace: {workspace.root}", file=file)
    print(f"Model: {model}", file=file)
    print(f"Permission: {args.permission}", file=file)
    if args.test_cmd:
        print(f"Test: {args.test_cmd}", file=file)
    if args.resume_session:
        print(f"Resuming: {args.resume_session}", file=file)
    elif task:
        print(f"Task: {first_line(task, limit=100)}", file=file)
    print("", file=file)


def print_human_report(result: object, workspace: Workspace) -> None:
    report = final_report(result, workspace)
    title = "Done" if report["status"] == "finished" else f"Stopped: {report['status']}"
    print(f"\n{title}")
    print(report["summary"] or "(no summary)")
    print("")
    print(f"Steps: {report['steps']}")
    print(f"Session: {report['session_id']}")
    print(f"Session state: {report['session_path']}")
    print(f"Trajectory: {report['trajectory_path']}")


def final_report(result: object, workspace: Workspace) -> dict[str, object]:
    return {
        "ok": getattr(result, "stopped_by") == "finished",
        "status": getattr(result, "stopped_by"),
        "summary": str(getattr(result, "summary", "")).strip(),
        "steps": getattr(result, "steps"),
        "session_id": getattr(result, "session_id"),
        "session_path": display_path(getattr(result, "session_path"), workspace),
        "trajectory_path": display_path(getattr(result, "trajectory_path"), workspace),
    }


def display_path(path: object, workspace: Workspace) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve(strict=False).relative_to(workspace.root))
    except ValueError:
        return str(candidate)


def first_line(value: str, *, limit: int = 120) -> str:
    line = value.strip().splitlines()[0] if value.strip() else ""
    if len(line) <= limit:
        return line
    return f"{line[: limit - 3]}..."


def tool_target(arguments: dict[str, object]) -> str:
    for key in ("path", "file_path", "target", "command"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return first_line(value, limit=80)
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
