import tempfile
import unittest
from pathlib import Path

from ninecoder import memory
from ninecoder.agent import AgentConfig, CodingAgent
from ninecoder.memory import MemoryStore, extract_facts
from ninecoder.model_client import ModelResponse, ToolCall
from ninecoder.permissions import PermissionMode
from ninecoder.workspace import Workspace


class _DistillModel:
    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        return ModelResponse(self.text)


class _FailingModel:
    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        raise RuntimeError("boom")


class _MemoryRunModel:
    """Answers the agent loop with a finish, and memory extraction with a fact."""

    def __init__(self, fact: str = "- Project uses pytest") -> None:
        self.fact = fact

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        if tools:
            return ModelResponse(
                "done", [ToolCall("call_1", "finish", {"summary": "fixed tests"})]
            )
        return ModelResponse(self.fact)


class ExtractFactsTest(unittest.TestCase):
    def test_returns_model_text(self) -> None:
        self.assertEqual(
            extract_facts(_DistillModel("durable facts"), "task", "summary"),
            "durable facts",
        )

    def test_falls_back_on_empty_content(self) -> None:
        self.assertEqual(
            extract_facts(_DistillModel(""), "task", "summary"),
            "- task: summary",
        )

    def test_falls_back_on_error(self) -> None:
        self.assertEqual(
            extract_facts(_FailingModel(), "task", "summary"),
            "- task: summary",
        )


class MemoryStoreTest(unittest.TestCase):
    def test_read_missing_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(MemoryStore(tmp).read(), "")

    def test_append_writes_dated_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            store.append("do the thing", "- fact one")
            content = store.read()
            self.assertIn("## ", content)
            self.assertIn("do the thing", content)
            self.assertIn("- fact one", content)

    def test_append_preserves_existing_blocks_and_trims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(tmp)
            original = memory.MAX_MEMORY_BLOCKS
            memory.MAX_MEMORY_BLOCKS = 3
            try:
                for index in range(5):
                    store.append(f"task {index}", f"- fact {index}")
                content = store.read()
            finally:
                memory.MAX_MEMORY_BLOCKS = original
            self.assertEqual(content.count("## "), 3)
            self.assertNotIn("task 0", content)
            self.assertNotIn("task 1", content)
            self.assertIn("task 4", content)


class AgentMemoryTest(unittest.TestCase):
    def test_run_writes_memory_then_next_run_injects_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(tmp)
            first = CodingAgent(
                _MemoryRunModel(),
                workspace,
                AgentConfig(max_steps=5, permission_mode=PermissionMode.AUTO),
            )
            first.run("fix the tests")

            mem_path = Path(tmp) / "MEMORY.md"
            self.assertTrue(mem_path.exists())
            self.assertIn("Project uses pytest", mem_path.read_text(encoding="utf-8"))

            second = CodingAgent(
                _MemoryRunModel(),
                workspace,
                AgentConfig(max_steps=5, permission_mode=PermissionMode.AUTO),
            )
            second.run("another task")

            system = second.messages[0]["content"]
            self.assertIn("Project memory", system)
            self.assertIn("Project uses pytest", system)

    def test_memory_disabled_skips_write_and_inject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(tmp)
            agent = CodingAgent(
                _MemoryRunModel(),
                workspace,
                AgentConfig(
                    max_steps=5,
                    permission_mode=PermissionMode.AUTO,
                    memory=False,
                ),
            )
            agent.run("fix the tests")

            self.assertFalse((Path(tmp) / "MEMORY.md").exists())
            self.assertNotIn("Project memory", agent.messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
