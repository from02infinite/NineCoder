from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

MEMORY_FILENAME = "MEMORY.md"
MAX_MEMORY_BLOCKS = 20

MEMORY_HEADER = (
    "# NineCoder Memory\n"
    "\n"
    "Durable facts, decisions, and preferences accumulated across runs.\n"
    "NineCoder appends a block after each run and injects this file into the\n"
    "system prompt of new tasks. Treat its contents as hints, not ground truth.\n"
    "\n"
)

_MEMORY_SYSTEM = (
    "You distill a finished coding-agent run into durable memory. "
    "Output only concise markdown bullets of facts that will help future runs: "
    "project layout, conventions, gotchas, decisions, and user preferences. "
    "Omit transient details such as exact diffs, timestamps, or step counts. "
    "If nothing is worth remembering, output nothing."
)

_BLOCK_SEPARATOR = "## "


def memory_block(memory: str) -> str:
    """Wrap accumulated memory for injection into the system prompt."""
    return (
        "## Project memory\n"
        "\n"
        "Relevant context from previous runs (treat as hints, not ground truth):\n"
        "\n"
        f"{memory.strip()}"
    )


def extract_facts(model: Any, task: str, summary: str) -> str:
    """Distill a finished run into durable facts, with a mechanical fallback."""
    transcript = f"Task: {task}\nSummary: {summary}"
    try:
        response = model.complete(
            [
                {"role": "system", "content": _MEMORY_SYSTEM},
                {"role": "user", "content": f"Run to remember:\n\n{transcript}"},
            ],
            [],
        )
        content = (getattr(response, "content", "") or "").strip()
        if content:
            return content
    except Exception:
        pass
    return f"- {task}: {summary}".strip()


class MemoryStore:
    """A workspace-local markdown file that accumulates cross-run memory."""

    def __init__(self, workspace_root: str | Path, filename: str = MEMORY_FILENAME):
        self.path = Path(workspace_root) / filename

    def read(self) -> str:
        if not self.path.exists():
            return ""
        return self.path.read_text(encoding="utf-8", errors="replace")

    def append(self, task: str, facts: str) -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        label = task.strip().replace("\n", " ")[:80]
        block = f"## {stamp} — {label}\n\n{facts.strip()}\n"
        preamble, blocks = _split(self.read())
        blocks.append(block)
        blocks = blocks[-MAX_MEMORY_BLOCKS:]
        content = MEMORY_HEADER + preamble + "".join(blocks)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(content, encoding="utf-8")
        return self.path


def _split(text: str) -> tuple[str, list[str]]:
    """Split a memory file into a preamble (anything before the first ``## ``)
    and a list of ``## `` blocks, preserving the separator on each block."""
    if text.startswith(MEMORY_HEADER):
        text = text[len(MEMORY_HEADER):]
    if _BLOCK_SEPARATOR not in text:
        return text, []
    first = text.index(_BLOCK_SEPARATOR)
    preamble = text[:first]
    if preamble and not preamble.endswith("\n"):
        preamble += "\n"
    blocks = [
        _BLOCK_SEPARATOR + part
        for part in text[first:].split(_BLOCK_SEPARATOR)
        if part.strip()
    ]
    return preamble, blocks
