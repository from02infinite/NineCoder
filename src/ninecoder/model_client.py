from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ninecoder.config import ModelConfig
from ninecoder.errors import ModelError


# Transient provider failures worth retrying with backoff. Everything else
# (4xx client errors, malformed bodies) is surfaced immediately so a coding
# loop fails fast instead of spinning on a bad request.
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT = 120
BACKOFF_BASE = 0.5


class _RetryableModelError(ModelError):
    """A transient failure that may succeed on a later attempt."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class ModelClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        request = self._build_request(payload)
        last_error: BaseException | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                data = self._post_json(request)
            except _RetryableModelError as exc:
                last_error = exc
            else:
                return parse_model_response(data)
            if attempt < self.config.max_retries:
                time.sleep(BACKOFF_BASE * (2 ** attempt))
        raise ModelError(
            f"model request failed after {self.config.max_retries + 1} attempt(s): {last_error}"
        )

    def _build_request(self, payload: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

    def _post_json(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = f"model provider returned HTTP {exc.code}: {body}"
            if exc.code in RETRYABLE_HTTP_CODES:
                raise _RetryableModelError(message) from exc
            raise ModelError(message) from exc
        except OSError as exc:
            raise _RetryableModelError(f"model request failed: {exc}") from exc
        except Exception as exc:
            raise ModelError(f"model request failed: {exc}") from exc


def parse_model_response(data: dict[str, Any]) -> ModelResponse:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelError("model response has no choices")
    first = choices[0]
    message = first.get("message") or {}
    if not isinstance(message, dict):
        raise ModelError("model response message is invalid")
    calls = []
    for raw in message.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") or {}
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = _parse_arguments(function.get("arguments", "{}"))
        calls.append(ToolCall(str(raw.get("id") or f"call_{len(calls)}"), name, arguments))
    return ModelResponse(
        content=message.get("content") or "",
        tool_calls=calls,
        finish_reason=first.get("finish_reason") or "",
        usage=data.get("usage") or {},
        raw=data,
    )


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        # A malformed argument payload is a model output problem, not a reason
        # to abort the whole run. Return empty args and let tool validation
        # report the missing/invalid arguments so the model can retry.
        return {}
    return parsed if isinstance(parsed, dict) else {}
