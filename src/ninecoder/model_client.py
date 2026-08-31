from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

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
    streamed: bool = False


class ModelClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        request = self._build_request(self._payload(messages, tools))
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

    def stream_complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_chunk: Callable[[str], None],
    ) -> ModelResponse:
        """Issue a streaming request, calling ``on_chunk`` per content delta.

        This is an optional capability beyond the :class:`ChatModel` protocol,
        so the agent probes for it with ``getattr`` and falls back to
        :meth:`complete` for plain models. Streaming is a single attempt: a
        stream that fails mid-way cannot be retried without replaying emitted
        chunks, so it surfaces as a :class:`ModelError` immediately.
        """
        if not self.config.stream:
            return self.complete(messages, tools)
        request = self._build_request(self._payload(messages, tools, stream=True))
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                # Yield decoded lines so ``parse_sse_lines`` can emit content
                # deltas live while the HTTP connection is still open.
                def lines() -> Any:
                    for raw in response:
                        yield raw.decode("utf-8", errors="replace")

                return parse_sse_lines(lines(), on_chunk=on_chunk)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ModelError(f"model provider returned HTTP {exc.code}: {body}") from exc
        except OSError as exc:
            raise ModelError(f"model request failed: {exc}") from exc

    def _payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if stream:
            payload["stream"] = True
        return payload

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


def parse_sse_lines(
    lines: Any,
    *,
    on_chunk: Callable[[str], None] | None = None,
) -> ModelResponse:
    """Parse an OpenAI-compatible SSE stream into a :class:`ModelResponse`.

    ``lines`` is an iterable of decoded text lines. Content deltas are passed to
    ``on_chunk`` as they arrive (if given), so a caller feeding a live socket
    can display tokens without waiting for the stream to end. Tool calls arrive
    as fragmented ``delta.tool_calls`` entries and are reassembled by index.
    """
    acc = _SSEAccumulator()
    for line in lines:
        payload = _extract_data_payload(line)
        if payload is None:
            continue
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        delta = acc.feed(data)
        if delta and on_chunk is not None:
            on_chunk(delta)
    return acc.to_response()


def _extract_data_payload(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    return stripped[len("data:"):].strip()


class _SSEAccumulator:
    def __init__(self) -> None:
        self.content_parts: list[str] = []
        self.tool_calls: dict[int, dict[str, str]] = {}
        self.finish_reason = ""
        self.usage: dict[str, Any] = {}

    def feed(self, data: dict[str, Any]) -> str:
        """Ingest one SSE event; return the content delta, if any."""
        delta_text = ""
        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            first = choices[0]
            delta = first.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    self.content_parts.append(content)
                    delta_text = content
                for tool_call in delta.get("tool_calls") or []:
                    self._feed_tool_call(tool_call)
            finish_reason = first.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                self.finish_reason = finish_reason
        usage = data.get("usage")
        if isinstance(usage, dict):
            self.usage = usage
        return delta_text

    def _feed_tool_call(self, tool_call: Any) -> None:
        if not isinstance(tool_call, dict):
            return
        index = tool_call.get("index", 0)
        entry = self.tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
        call_id = tool_call.get("id")
        if isinstance(call_id, str) and call_id:
            entry["id"] = call_id
        function = tool_call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                entry["name"] = name
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments:
                entry["arguments"] += arguments

    def to_response(self) -> ModelResponse:
        calls: list[ToolCall] = []
        for index in sorted(self.tool_calls):
            entry = self.tool_calls[index]
            if not entry["name"]:
                continue
            call_id = entry["id"] or f"call_{index}"
            calls.append(ToolCall(call_id, entry["name"], _parse_arguments(entry["arguments"])))
        return ModelResponse(
            content="".join(self.content_parts),
            tool_calls=calls,
            finish_reason=self.finish_reason,
            usage=self.usage,
            streamed=True,
        )
