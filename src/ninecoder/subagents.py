from __future__ import annotations

from typing import Any, Protocol

from ninecoder.model_client import ModelResponse


class SimpleChatModel(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse: ...


SUBAGENT_SYSTEM = """You are a lightweight read-only subagent inside NineCoder.

Return concise, practical analysis. Do not call tools. Do not claim to have
edited files or executed commands.
"""


def ask_subagent(model: SimpleChatModel, role: str, prompt: str, context: str = "") -> str:
    messages = [
        {"role": "system", "content": SUBAGENT_SYSTEM},
        {
            "role": "user",
            "content": f"Role: {role}\n\nContext:\n{context}\n\nTask:\n{prompt}",
        },
    ]
    response = model.complete(messages, [])
    return response.content.strip() or "(subagent returned no text)"
