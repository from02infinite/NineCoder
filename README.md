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

Useful environment variables:

- `DEEPSEEK_API_KEY` or `NINECODER_API_KEY`
- `NINECODER_BASE_URL` or `DEEPSEEK_BASE_URL`
- `NINECODER_MODEL`

## Safety

NineCoder never stores API keys. File writes are restricted to the selected
workspace, sensitive paths are denied, shell commands have timeouts, and
permission mode can be `plan`, `ask`, or `auto`.
