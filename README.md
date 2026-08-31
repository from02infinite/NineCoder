# NineCoder

NineCoder is a lightweight coding agent implemented from scratch. It talks to an
OpenAI-compatible model, exposes local tools, executes them in a bounded loop,
and records every step as JSONL.

Default model: `deepseek-v4-flash`.

## Quick Start

```bash
export DEEPSEEK_API_KEY="..."
python -m pip install -e .
ninecoder --workspace demo_workspace "Fix the bug and run tests"
```

Create a demo workspace:

```bash
python scripts/create_demo_workspace.py
ninecoder \
  --permission auto \
  --workspace demo_workspace \
  --test "python -m unittest -q" \
  "Make divide raise ValueError('division by zero') when b is zero, then run tests"
```

Resume an unfinished run:

```bash
ninecoder --workspace demo_workspace --resume 20260831-120000-000000
```

In the interactive REPL, choose an old session and continue it:

```text
/resume
```

Rich TUI mode opens a keyboard picker: use Up/Down to choose a saved session,
then press Enter. Use `/resume <id>` to jump directly, or `/switch <id>` when
you want to branch a new session from old context instead of continuing it.

Print a script-friendly final report:

```bash
ninecoder --json --workspace demo_workspace "Fix the bug and run tests"
```

Run local tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

Useful environment variables:

- `DEEPSEEK_API_KEY` or `NINECODER_API_KEY`
- `NINECODER_BASE_URL` or `DEEPSEEK_BASE_URL`
- `NINECODER_MODEL`
- `NINECODER_MAX_RETRIES` (transient model-error retries, default `3`)
- `NINECODER_SANDBOX` (sandbox backend, default `auto`)
- `NINECODER_STREAM` (`0` disables streaming model output)

## Safety

NineCoder never stores API keys. File writes are restricted to the selected
workspace, sensitive paths (`.env*`, SSH/AWS/GCP keys and directories) are
denied, shell commands have timeouts, and permission mode can be `plan`, `ask`,
or `auto`.

`run_shell` runs inside an OS sandbox by default (`--sandbox auto`): it auto-
detects `bwrap` (Linux, full-strength) or `sandbox-exec` (macOS, deprecated and
best-effort), and falls back to no sandbox with a banner note when neither is
installed. The sandbox makes the filesystem read-only except the workspace and
`/tmp`, gives the command fresh namespaces, and **blocks network access by
default** — pass `--allow-network` for commands that need it (e.g. `pip install`,
`git clone`). Explicit backends are `--sandbox bwrap` / `--sandbox sandbox-exec`;
`--sandbox off` disables it entirely.

The sandbox is filesystem + process + network isolation, not a full secret
container: a sandboxed command can still *read* `~/.ssh` and friends (it just
cannot send the content out while network is blocked), and `sandbox-exec` is a
deprecated Apple shim whose profile semantics are weaker than bubblewrap. The
sensitive-path denial and `--permission ask` remain the defense for read access.
When the model handles untrusted input, prefer `--permission ask` on top of the
sandbox.

## Implemented Agent Features

- One self-written agent loop
- Tool registry with JSON schema definitions
- Tools: bash, read, write, edit, glob/list, grep/search, todo, finish
- On-demand markdown skill loading
- Read-only subagent tasks with ids, status, independent context, and saved results
- Simple task graph with dependencies
- Resumable session state with messages, todos, task graph, subagent tasks, and status
- Context compaction by recent-window retention, summary files, and stored long tool outputs
- JSONL trajectory persistence
- MCP-like local capability router with `tools/list` and `tools/call`
- Hook points that can inspect or rewrite agent startup, model requests, model responses, tool calls, tool results, and stop events; tool hooks can also block execution with a synthetic result
- Permission governance with `plan`, `ask`, and `auto`
- Pluggable OS sandbox for `run_shell` (bubblewrap / sandbox-exec), network-blocked by default
- Streaming model output (SSE), tokens rendered as they arrive in plain mode (`--no-stream` to disable)
