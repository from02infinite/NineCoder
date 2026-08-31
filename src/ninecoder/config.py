from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> "ModelConfig":
        resolved_model = model or os.getenv("NINECODER_MODEL") or "deepseek-v4-flash"
        resolved_base_url = (
            base_url
            or os.getenv("NINECODER_BASE_URL")
            or os.getenv("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com/v1"
        ).rstrip("/")
        api_key = os.getenv("NINECODER_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""
        if not api_key:
            raise ValueError(
                "Missing API key. Set DEEPSEEK_API_KEY or NINECODER_API_KEY."
            )
        return cls(
            model=resolved_model,
            base_url=resolved_base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
