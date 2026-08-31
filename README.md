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
  --test-cmd "python -m unittest -q" \
  "Make divide raise ValueError('division by zero') when b is zero, then run tests"
```

Run local tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

Useful environment variables:

- `DEEPSEEK_API_KEY` or `NINECODER_API_KEY`
- `NINECODER_BASE_URL` or `DEEPSEEK_BASE_URL`
- `NINECODER_MODEL`

## Safety

NineCoder never stores API keys. File writes are restricted to the selected
workspace, sensitive paths are denied, shell commands have timeouts, and
permission mode can be `plan`, `ask`, or `auto`.

## Implemented Agent Features

- One self-written agent loop
- Tool registry with JSON schema definitions
- Tools: bash, read, write, edit, glob/list, grep/search, todo, finish
- On-demand markdown skill loading
- Lightweight read-only subagent calls
- Simple task graph with dependencies
- Context compaction by recent-window retention and tool-output truncation
- JSONL trajectory persistence
- MCP-like local capability router with `tools/list` and `tools/call`
- Hook points around tool execution and finish
- Permission governance with `plan`, `ask`, and `auto`
