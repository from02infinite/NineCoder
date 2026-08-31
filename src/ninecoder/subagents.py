from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
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


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_task_id(count: int) -> str:
    return f"subagent-{count + 1}"


@dataclass
class SubagentTask:
    id: str
    role: str
    prompt: str
    context: str = ""
    status: str = "pending"
    result: str = ""
    error: str = ""
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    messages: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SubagentTaskRunner:
    def __init__(self, model: SimpleChatModel):
        self.model = model
        self.tasks: dict[str, SubagentTask] = {}

    def start(self, role: str, prompt: str, context: str = "") -> SubagentTask:
        task = SubagentTask(new_task_id(len(self.tasks)), role, prompt, context)
        task.status = "running"
        task.messages = subagent_messages(role, prompt, context)
        task.updated_at = now_iso()
        self.tasks[task.id] = task
        try:
            response = self.model.complete(task.messages, [])
            task.result = response.content.strip() or "(subagent returned no text)"
            task.status = "completed"
        except Exception as exc:
            task.error = f"{type(exc).__name__}: {exc}"
            task.status = "failed"
        task.updated_at = now_iso()
        return task

    def get(self, task_id: str) -> SubagentTask | None:
        return self.tasks.get(task_id)

    def list(self) -> list[SubagentTask]:
        return list(self.tasks.values())

    def export_tasks(self) -> list[dict[str, Any]]:
        return [task.to_dict() for task in self.list()]

    def import_tasks(self, tasks: list[dict[str, Any]]) -> None:
        self.tasks = {}
        for item in tasks:
            task = SubagentTask(**item)
            self.tasks[task.id] = task


def ask_subagent(model: SimpleChatModel, role: str, prompt: str, context: str = "") -> str:
    task = SubagentTaskRunner(model).start(role, prompt, context)
    if task.status == "failed":
        return f"subagent failed: {task.error}"
    return task.result


def subagent_messages(role: str, prompt: str, context: str = "") -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": SUBAGENT_SYSTEM},
        {
            "role": "user",
            "content": f"Role: {role}\n\nContext:\n{context}\n\nTask:\n{prompt}",
        },
    ]
    return messages
