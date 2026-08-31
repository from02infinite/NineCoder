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
    max_retries: int = 3
    stream: bool = True

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
        stream: bool | None = None,
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
        resolved_retries = max_retries
        raw_retries = os.getenv("NINECODER_MAX_RETRIES")
        if raw_retries is not None:
            try:
                resolved_retries = int(raw_retries)
            except ValueError:
                resolved_retries = max_retries
        if stream is None:
            raw_stream = os.getenv("NINECODER_STREAM")
            resolved_stream = (
                raw_stream.strip().lower() not in {"0", "false", "no", "off"}
                if raw_stream is not None
                else True
            )
        else:
            resolved_stream = stream
        return cls(
            model=resolved_model,
            base_url=resolved_base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=resolved_retries,
            stream=resolved_stream,
        )
