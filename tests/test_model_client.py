import json
import unittest

from ninecoder.model_client import parse_model_response


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
