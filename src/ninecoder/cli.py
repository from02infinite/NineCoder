from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from ninecoder.agent import AgentConfig, CodingAgent
from ninecoder.config import ModelConfig
from ninecoder.model_client import ModelClient
from ninecoder.permissions import PermissionMode
from ninecoder.repl import REPL_HELP, format_session_list, format_session_tree, parse_command
from ninecoder.session import SessionState, SessionStore
from ninecoder.ui import UiContext, make_ui
from ninecoder.workspace import Workspace


EXAMPLES = """examples:
  ninecoder                       # interactive REPL (on a TTY)
  ninecoder "Fix the failing tests"
  ninecoder -w demo --permission auto \\
    --test "python -m unittest -q" "Make tests pass"
  ninecoder -w demo --resume 20260831-120000-000000
  ninecoder --plain "Fix the bug"   # simple print output (CI / pipes)
  ninecoder --debug "Fix the bug"   # verbose internal logging
"""


class NineCoderHelpFormatter(argparse.RawDescriptionHelpFormatter):
    pass


class NineCoderParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\nTry '{self.prog} --help'.\n")


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
    run_options.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable the cross-run MEMORY.md memory",
    )
    run_options.add_argument(
        "--memory-file",
        default="MEMORY.md",
        help="Memory file name in the workspace root (default: MEMORY.md)",
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
    output_options.add_argument(
        "--debug",
        action="store_true",
        help="Show internal debug logging (iterations, timings, decisions)",
    )
    output_style = output_options.add_mutually_exclusive_group()
    output_style.add_argument(
        "--plain",
        action="store_true",
        help="Disable the rich TUI and print plain-text output",
    )
    output_style.add_argument(
        "--rich",
        action="store_true",
        help="Force the rich TUI with Markdown rendering",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    interactive = _is_interactive()
    repl_mode = bool(
        not args.task
        and not args.resume_session
        and interactive
        and not args.json
        and not args.quiet
    )
    if not args.task and not args.resume_session and not repl_mode:
        parser.error("task is required unless --resume-session is provided")

    try:
        model_config = ModelConfig.from_env(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        workspace = Workspace(Path(args.workspace))
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1

    ui = _build_ui(args, model_config, workspace, interactive)

    if repl_mode:
        return _run_repl(ui, model_config, workspace, args)

    ui.session_started(
        task=" ".join(args.task),
        resumed=bool(args.resume_session),
    )
    try:
        result = _run_once(ui, model_config, workspace, args)
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        else:
            ui.error(_error_message(args, exc))
        return 1

    if args.json:
        print(json.dumps(final_report(result, workspace), ensure_ascii=False, indent=2))
    else:
        print_human_report(
            result,
            workspace,
            render_markdown=bool(getattr(ui, "renders_markdown", False)),
        )
    ui.shutdown()
    return 0 if result.stopped_by == "finished" else 1


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty() and sys.stderr.isatty()


def _build_ui(
    args: argparse.Namespace,
    model_config: ModelConfig,
    workspace: Workspace,
    interactive: bool,
) -> object:
    if args.json or args.quiet:
        mode = "null"
    elif args.plain:
        mode = "plain"
    elif args.rich or interactive:
        mode = "rich"
    else:
        mode = "plain"
    context = UiContext(
        model=model_config.model,
        workspace=str(workspace.root),
        permission=args.permission,
        test_cmd=args.test_cmd,
        debug=bool(args.debug),
    )
    return make_ui(mode=mode, context=context)


def _build_agent(
    ui: object,
    model_config: ModelConfig,
    workspace: Workspace,
    args: argparse.Namespace,
    resume_session: str = "",
) -> CodingAgent:
    return CodingAgent(
        ModelClient(model_config),
        workspace,
        AgentConfig(
            max_steps=args.max_steps,
            permission_mode=PermissionMode(args.permission),
            test_cmd=args.test_cmd,
            resume_session=resume_session,
            memory=not args.no_memory,
            memory_file=args.memory_file,
        ),
        ui=ui,
    )


def _run_once(
    ui: object,
    model_config: ModelConfig,
    workspace: Workspace,
    args: argparse.Namespace,
    *,
    task: str | None = None,
    resume_session: str | None = None,
) -> object:
    if resume_session is None:
        resume_session = args.resume_session
    agent = _build_agent(ui, model_config, workspace, args, resume_session)
    return agent.run(task if task is not None else " ".join(args.task))


def _run_repl(
    ui: object,
    model_config: ModelConfig,
    workspace: Workspace,
    args: argparse.Namespace,
) -> int:
    ui.session_started()
    agent: CodingAgent | None = None
    head_id = ""
    fork_from = ""
    try:
        while True:
            text = ui.prompt_input()
            if text is None:
                break
            text = text.strip()
            if not text:
                continue
            command = parse_command(text)
            if command is not None:
                verb, arg = command
                if verb == "quit":
                    break
                if verb == "new":
                    agent = None
                    head_id = ""
                    fork_from = ""
                    ui.info("Starting a new conversation.")
                    continue
                if verb == "help":
                    ui.info(REPL_HELP)
                    continue
                if verb == "resume":
                    sessions = _list_sessions(workspace)
                    target = _resolve_session(sessions, arg) if arg else ui.select_session(
                        sessions, head_id
                    )
                    if target is None:
                        ui.info("No session selected. Try /list or /resume <id>.")
                        continue
                    agent = _build_agent(ui, model_config, workspace, args, target)
                    session = agent.open_session()
                    head_id = session.id
                    fork_from = ""
                    ui.info(f"Resumed session {session.id}. Continue with your next message.")
                    continue
                if verb == "compact":
                    if agent is None or agent.session is None:
                        ui.info("No active conversation to compact.")
                        continue
                    path = agent.compact_context()
                    ui.info(f"Compacted current conversation. Session saved at {path}.")
                    continue
                if verb == "tree":
                    ui.info(format_session_tree(_list_sessions(workspace), head_id))
                    continue
                if verb == "list":
                    ui.info(format_session_list(_list_sessions(workspace), head_id))
                    continue
                if verb == "switch":
                    target = _resolve_session(_list_sessions(workspace), arg)
                    if target is None:
                        ui.info(f"No session matches '{arg}'. Try /list.")
                        continue
                    fork_from = target
                    head_id = target
                    ui.info(f"Will branch from {target}. Your next message forks a new session.")
                    continue
                ui.info(f"Unknown command: {arg}. Try /help.")
                continue
            try:
                if fork_from:
                    if agent is None:
                        agent = _build_agent(ui, model_config, workspace, args)
                    result = agent.fork(fork_from, text)
                elif agent is None:
                    agent = _build_agent(ui, model_config, workspace, args)
                    result = agent.run(text)
                else:
                    result = agent.continue_turn(text)
                head_id = result.session_id
                fork_from = ""
            except KeyboardInterrupt:
                ui.error("Interrupted")
            except Exception as exc:
                ui.error(_error_message(args, exc))
    finally:
        ui.shutdown()
    return 0


def _list_sessions(workspace: Workspace) -> list[SessionState]:
    # ``runs`` matches AgentConfig.runs_dir (the CLI exposes no override).
    return SessionStore(Path(workspace.root) / "runs" / "sessions").list()


def _resolve_session(sessions: list[SessionState], arg: str) -> str | None:
    if not arg:
        return None
    exact = [session.id for session in sessions if session.id == arg]
    if exact:
        return exact[0]
    prefixes = [session.id for session in sessions if session.id.startswith(arg)]
    return prefixes[0] if len(prefixes) == 1 else None


def _error_message(args: argparse.Namespace, exc: BaseException) -> str:
    if getattr(args, "debug", False):
        return traceback.format_exc()
    return str(exc)


def print_human_report(
    result: object,
    workspace: Workspace,
    *,
    render_markdown: bool = False,
) -> None:
    report = final_report(result, workspace)
    title = "Done" if report["status"] == "finished" else f"Stopped: {report['status']}"
    print(f"\n{title}")
    summary = str(report["summary"] or "(no summary)")
    if render_markdown:
        from rich.console import Console
        from rich.markdown import Markdown

        Console(highlight=False).print(Markdown(summary))
    else:
        print(summary)
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


if __name__ == "__main__":
    raise SystemExit(main())
