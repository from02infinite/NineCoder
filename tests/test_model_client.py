import io
import json
import unittest
import urllib.error
from unittest import mock

from ninecoder.config import ModelConfig
from ninecoder.errors import ModelError
from ninecoder.model_client import ModelClient, parse_model_response


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
