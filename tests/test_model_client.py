import io
import json
import unittest
import urllib.error
from unittest import mock

from ninecoder.config import ModelConfig
from ninecoder.errors import ModelError
from ninecoder.model_client import ModelClient, parse_model_response, parse_sse_lines


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _ok_response(content: str = "done") -> bytes:
    return json.dumps(
        {"choices": [{"message": {"content": content}}]}
    ).encode("utf-8")


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.example.com/v1/chat/completions", code, "error", {}, io.BytesIO(b"")
    )


class ModelClientTest(unittest.TestCase):
    def test_parse_tool_call(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "I will inspect files.",
                        "tool_calls": [
                            {
                                "id": "abc",
                                "type": "function",
                                "function": {
                                    "name": "list_files",
                                    "arguments": json.dumps({"path": "."}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1},
        }

        parsed = parse_model_response(response)

        self.assertEqual(parsed.tool_calls[0].name, "list_files")
        self.assertEqual(parsed.tool_calls[0].arguments, {"path": "."})

    def test_parse_malformed_tool_call_arguments_is_graceful(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "x",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{"},
                            },
                            {
                                "id": "y",
                                "type": "function",
                                "function": {"name": "list_files", "arguments": "[1, 2]"},
                            },
                        ]
                    }
                }
            ]
        }

        parsed = parse_model_response(response)

        self.assertEqual(parsed.tool_calls[0].arguments, {})
        self.assertEqual(parsed.tool_calls[1].arguments, {})

    def test_retries_transient_http_error_then_succeeds(self) -> None:
        config = ModelConfig("m", "https://api.example.com/v1", "k", 0.2, 4096, max_retries=2)
        client = ModelClient(config)
        with mock.patch(
            "ninecoder.model_client.urllib.request.urlopen",
            side_effect=[_http_error(503), _http_error(502), _FakeResponse(_ok_response())],
        ) as urlopen, mock.patch("ninecoder.model_client.time.sleep") as sleep:
            response = client.complete([{"role": "user", "content": "hi"}], [])

        self.assertEqual(response.content, "done")
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_does_not_retry_client_error(self) -> None:
        config = ModelConfig("m", "https://api.example.com/v1", "k", 0.2, 4096, max_retries=3)
        client = ModelClient(config)
        with mock.patch(
            "ninecoder.model_client.urllib.request.urlopen",
            side_effect=[_http_error(400)],
        ) as urlopen, mock.patch("ninecoder.model_client.time.sleep") as sleep:
            with self.assertRaises(ModelError):
                client.complete([{"role": "user", "content": "hi"}], [])

        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(sleep.call_count, 0)

    def test_gives_up_after_max_retries(self) -> None:
        config = ModelConfig("m", "https://api.example.com/v1", "k", 0.2, 4096, max_retries=2)
        client = ModelClient(config)
        with mock.patch(
            "ninecoder.model_client.urllib.request.urlopen",
            side_effect=[_http_error(500), _http_error(500), _http_error(500)],
        ) as urlopen, mock.patch("ninecoder.model_client.time.sleep"):
            with self.assertRaises(ModelError):
                client.complete([{"role": "user", "content": "hi"}], [])

        self.assertEqual(urlopen.call_count, 3)


class StreamParseTest(unittest.TestCase):
    def test_parse_sse_accumulates_content_and_finish_reason(self) -> None:
        lines = [
            'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}',
            'data: {"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}',
            "data: [DONE]",
        ]

        response = parse_sse_lines(lines)

        self.assertEqual(response.content, "Hello world")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage, {"prompt_tokens": 5, "completion_tokens": 2})
        self.assertTrue(response.streamed)
        self.assertEqual(response.tool_calls, [])

    def test_parse_sse_emits_chunks_in_order(self) -> None:
        lines = [
            'data: {"choices":[{"index":0,"delta":{"content":"a"},"finish_reason":null}]}',
            'data: {"choices":[{"index":0,"delta":{"content":"b"},"finish_reason":null}]}',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        chunks: list[str] = []

        response = parse_sse_lines(lines, on_chunk=chunks.append)

        self.assertEqual(chunks, ["a", "b"])
        self.assertEqual(response.content, "ab")

    def test_parse_sse_reassembles_fragmented_tool_calls(self) -> None:
        lines = [
            'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"read_file","arguments":""}}]},"finish_reason":null}]}',
            'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\": "}}]},"finish_reason":null}]}',
            'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"a.py\\"}"}}]},"finish_reason":null}]}',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]

        response = parse_sse_lines(lines)

        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(response.tool_calls[0].id, "c1")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "a.py"})
        self.assertEqual(response.finish_reason, "tool_calls")

    def test_parse_sse_ignores_comments_and_keepalive(self) -> None:
        lines = [
            ": keep-alive",
            'data: {"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}',
            "data: [DONE]",
        ]

        response = parse_sse_lines(lines)

        self.assertEqual(response.content, "x")


class StreamClientTest(unittest.TestCase):
    def _streaming_response(self) -> object:
        class _StreamResponse:
            def __init__(self) -> None:
                self._data = [
                    b'data: {"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
                    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                    b"data: [DONE]\n\n",
                ]
                self._index = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._index >= len(self._data):
                    raise StopIteration
                item = self._data[self._index]
                self._index += 1
                return item

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return _StreamResponse()

    def test_stream_complete_emits_chunks_and_returns_response(self) -> None:
        config = ModelConfig("m", "https://api.example.com/v1", "k", 0.2, 4096)
        client = ModelClient(config)
        chunks: list[str] = []
        with mock.patch(
            "ninecoder.model_client.urllib.request.urlopen",
            return_value=self._streaming_response(),
        ):
            response = client.stream_complete(
                [{"role": "user", "content": "hi"}], [], chunks.append
            )

        self.assertEqual(chunks, ["Hello"])
        self.assertEqual(response.content, "Hello")
        self.assertEqual(response.finish_reason, "stop")
        self.assertTrue(response.streamed)

    def test_stream_complete_falls_back_when_disabled(self) -> None:
        config = ModelConfig("m", "https://api.example.com/v1", "k", 0.2, 4096, stream=False)
        client = ModelClient(config)
        chunks: list[str] = []
        with mock.patch(
            "ninecoder.model_client.urllib.request.urlopen",
            return_value=_FakeResponse(_ok_response("done")),
        ) as urlopen:
            response = client.stream_complete(
                [{"role": "user", "content": "hi"}], [], chunks.append
            )

        self.assertEqual(response.content, "done")
        self.assertFalse(response.streamed)
        self.assertEqual(chunks, [])
