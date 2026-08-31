# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NineCoder is a self-contained coding agent built from scratch in pure Python (stdlib only — no third-party runtime dependencies, Python ≥ 3.10). It calls an OpenAI-compatible model, exposes local tools, runs them in a bounded loop, and records every step as JSONL. Default model is `deepseek-v4-flash`, default base URL `https://api.deepseek.com/v1`.

## Commands

Install (editable; exposes the `ninecoder` console script):

```bash
python -m pip install -e .
```

Run the agent (requires `DEEPSEEK_API_KEY` or `NINECODER_API_KEY`):

```bash
ninecoder --workspace demo_workspace "Fix the bug and run tests"
```

Create the demo workspace fixture:

```bash
python scripts/create_demo_workspace.py
```

Run the full test suite:

```bash
pytest                       # preferred; pyproject already sets pythonpath=src
# or, matching the README:
PYTHONPATH=src python -m unittest discover -s tests
```

Run a single test:

```bash
pytest tests/test_context.py::ContextManagerTest::test_large_tool_result_is_saved_with_model_reference
```

Tests are `unittest.TestCase` subclasses (also runnable by pytest). Agent tests use a `FakeModel` implementing the `ChatModel` protocol instead of hitting the network.

## Architecture

Everything funnels through one loop: `CodingAgent.run` in `src/ninecoder/agent.py`. Each step: build model messages → call model → for each returned tool call, execute it → append the tool result → repeat until `finish` (or max_steps / repeated no-tool responses). All other modules support that loop.

Module map (in `src/ninecoder/`):

- `cli.py` — argparse entry point (`ninecoder`), a progress hook, and human/JSON report output.
- `config.py` — `ModelConfig.from_env` resolves model/base_url/api_key from env vars.
- `model_client.py` — `ModelClient` issues the chat-completions HTTP call via stdlib `urllib` (no `requests`), parsing into `ModelResponse`/`ToolCall` dataclasses.
- `agent.py` — the loop, plus `AgentConfig`, `AgentRun`, and session/trajectory wiring.
- `tools.py` — `ToolRegistry` owns `Tool` objects (name, description, JSON schema, handler). Builtin tools are registered in `_register_builtin_tools`; add a tool by registering a `Tool`. `ToolResult.terminate=True` ends the loop (that is what `finish` does).
- `workspace.py` — the only place real filesystem/shell I/O happens. Enforces workspace confinement, sensitive-path denial, shell timeouts, and a dangerous-command blocklist. Tools delegate here.
- `permissions.py` — `plan`/`ask`/`auto` modes; splits tools into read-only vs mutating sets; sensitive-path detection.
- `context.py` — context compaction. `ContextManager` writes a `summary.md` and offloads oversized tool outputs to files under `runs/context/<session>/`; `compact_messages` is the in-memory fallback. Message grouping must keep an assistant `tool_calls` message together with its following `tool` messages (`valid_model_messages`).
- `session.py` — resumable `SessionState`, persisted as JSON at `runs/sessions/<id>/session.json`.
- `trajectory.py` — append-only JSONL run record at `runs/<id>.jsonl`.
- `hooks.py` — `AgentHook` protocol (`before_agent_start`/`before_model`/`after_model`/`before_tool`/`after_tool`/`on_stop`/`on_finish`). Tool hooks may use the new single-arg (`ToolRequest`/`ToolResponse`) or legacy multi-arg signature; `agent._expects_one_argument` disambiguates. `ToolDecision.blocked_result` lets a hook skip execution and return a synthetic tool result.
- `subagents.py` — read-only subagents via `SubagentTaskRunner`; tasks carry id/role/status and are persisted into the session.
- `skills.py` — on-demand markdown skills loaded from `skills/*.md`.
- `mcp_like.py` — `LocalCapabilityRouter` exposes `tools/list` and `tools/call` over the same registry.
- `prompts.py` — `SYSTEM_PROMPT`, `task_prompt`, `no_tool_retry`.
- `errors.py` — `AgentError` → `ModelError`, `ToolError` → `PermissionDenied`.

### Key patterns

- Model access is behind the `ChatModel` Protocol (a `complete(messages, tools)` method), so tests inject a fake model.
- Tool execution path: `ToolRegistry.execute` validates JSON args → evaluates permission → calls the handler → catches exceptions into `ToolResult(is_error=True)`.
- Every startup/model/tool interaction passes through hooks, which may inspect or rewrite the request/response (returning `None` leaves it unchanged). Stop notifications are emitted from `_stop` for both normal and abnormal termination.
