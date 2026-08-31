from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ninecoder.agent import AgentConfig, CodingAgent
from ninecoder.config import ModelConfig
from ninecoder.model_client import ModelClient
from ninecoder.permissions import PermissionMode
from ninecoder.workspace import Workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ninecoder",
        description="Run the NineCoder local coding agent.",
    )
    parser.add_argument("task", nargs="+", help="Natural-language coding task")
    parser.add_argument("-w", "--workspace", default=".", help="Workspace directory")
    parser.add_argument("-m", "--model", default=None, help="Model name")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument(
        "--permission",
        choices=[mode.value for mode in PermissionMode],
        default=PermissionMode.ASK.value,
        help="Permission mode for mutating tools",
    )
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum agent loop steps")
    parser.add_argument("--test-cmd", default="", help="Verification command to run before finish")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
            ),
        )
        result = agent.run(" ".join(args.task))
    except Exception as exc:
        print(f"NineCoder failed: {exc}", file=sys.stderr)
        return 1
    print("\nNineCoder finished")
    print(f"Status: {result.stopped_by}")
    print(f"Steps: {result.steps}")
    print(f"Trajectory: {result.trajectory_path}")
    print("\nSummary:")
    print(result.summary)
    return 0 if result.stopped_by == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
