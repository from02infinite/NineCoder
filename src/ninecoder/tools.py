from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from ninecoder.errors import PermissionDenied
from ninecoder.permissions import Decision, PermissionMode, evaluate_permission
from ninecoder.skills import SkillLibrary
from ninecoder.subagents import SubagentTaskRunner
from ninecoder.workspace import Workspace

if TYPE_CHECKING:
    from ninecoder.ui.base import AgentUI


@dataclass(frozen=True)
class ToolResult:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False
    terminate: bool = False


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], ToolResult]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(
        self,
        workspace: Workspace,
        mode: PermissionMode,
        *,
        model: Any | None = None,
        skill_library: SkillLibrary | None = None,
        ui: "AgentUI | None" = None,
    ):
        from ninecoder.ui.base import AgentUI

        self.workspace = workspace
        self.mode = mode
        self.model = model
        self.ui = ui if ui is not None else AgentUI()
        self.skill_library = skill_library or SkillLibrary()
        self.subagent_runner = SubagentTaskRunner(model) if model is not None else None
        self.todos: list[dict[str, str]] = []
        self.task_graph: list[dict[str, Any]] = []
        self._tools: dict[str, Tool] = {}
        self._register_builtin_tools()

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema() for name in self.names]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                f"Unknown tool: {name}. Available tools: {', '.join(self.names)}",
                is_error=True,
            )
        validation_error = validate_arguments(tool.parameters, arguments)
        if validation_error:
            return ToolResult(f"Invalid arguments for {name}: {validation_error}", is_error=True)
        permission = evaluate_permission(self.mode, name, self._target_argument(arguments))
        self.ui.debug(
            f"permission {permission.decision.value} for {name}"
            + (f" ({permission.reason})" if permission.reason else "")
        )
        if permission.decision is Decision.DENY:
            return ToolResult(f"Permission denied: {permission.reason}", is_error=True)
        if permission.decision is Decision.ASK and not self._confirm(name, arguments, permission.reason):
            return ToolResult(f"Permission denied by user: {permission.reason}", is_error=True)
        try:
            return tool.handler(arguments)
        except PermissionDenied as exc:
            return ToolResult(f"Permission denied: {exc}", is_error=True)
        except Exception as exc:
            return ToolResult(f"Tool error in {name}: {type(exc).__name__}: {exc}", is_error=True)

    def _register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def _register_builtin_tools(self) -> None:
        self._register(
            Tool(
                "list_files",
                "List files in a workspace directory.",
                object_schema({"path": string("Directory path, default '.'")}),
                lambda args: ToolResult(self.workspace.list_files(args.get("path", "."))),
            )
        )
        self._register(
            Tool(
                "read_file",
                "Read a text file with optional line offset and limit.",
                object_schema(
                    {
                        "path": string("File path to read"),
                        "offset": integer("1-indexed start line", default=1),
                        "limit": integer("Maximum lines to read", default=200),
                    },
                    required=["path"],
                ),
                lambda args: ToolResult(
                    self.workspace.read_file(
                        args["path"],
                        int(args.get("offset", 1)),
                        int(args.get("limit", 200)),
                    )
                ),
            )
        )
        self._register(
            Tool(
                "search",
                "Search file contents using ripgrep when available.",
                object_schema(
                    {
                        "pattern": string("Search pattern"),
                        "path": string("Directory or file path, default '.'"),
                        "include": string("Optional glob include pattern"),
                    },
                    required=["pattern"],
                ),
                lambda args: ToolResult(
                    self.workspace.search(
                        args["pattern"],
                        args.get("path", "."),
                        args.get("include", ""),
                    )
                ),
            )
        )
        self._register(
            Tool(
                "edit_file",
                "Edit one file with exact search/replace text and return a unified diff.",
                object_schema(
                    {
                        "path": string("File path to edit"),
                        "old_text": string("Exact text to replace"),
                        "new_text": string("Replacement text"),
                        "replace_all": boolean("Replace every occurrence", default=False),
                    },
                    required=["path", "old_text", "new_text"],
                ),
                lambda args: ToolResult(
                    self.workspace.edit_file(
                        args["path"],
                        args["old_text"],
                        args["new_text"],
                        bool(args.get("replace_all", False)),
                    )
                ),
            )
        )
        self._register(
            Tool(
                "write_file",
                "Create or overwrite a file inside the workspace and return a diff.",
                object_schema(
                    {
                        "path": string("File path to write"),
                        "content": string("Complete file content"),
                    },
                    required=["path", "content"],
                ),
                lambda args: ToolResult(self.workspace.write_file(args["path"], args["content"])),
            )
        )
        self._register(
            Tool(
                "run_shell",
                "Run a shell command in the workspace with timeout and output capture.",
                object_schema(
                    {
                        "command": string("Shell command to run"),
                        "timeout": integer("Timeout in seconds", default=30),
                    },
                    required=["command"],
                ),
                self._run_shell,
            )
        )
        self._register(
            Tool(
                "update_todo",
                "Replace the visible task checklist maintained by the agent.",
                object_schema(
                    {
                        "todos": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "doing", "done"],
                                    },
                                },
                                "required": ["content", "status"],
                            },
                        }
                    },
                    required=["todos"],
                ),
                self._update_todo,
            )
        )
        self._register(
            Tool(
                "update_task_graph",
                "Replace the task graph, including simple dependencies between task ids.",
                object_schema(
                    {
                        "tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "content": {"type": "string"},
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "doing", "done"],
                                    },
                                    "depends_on": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["id", "content", "status"],
                            },
                        }
                    },
                    required=["tasks"],
                ),
                self._update_task_graph,
            )
        )
        self._register(
            Tool(
                "load_skill",
                "Load an on-demand markdown skill by name.",
                object_schema({"name": string("Skill name")}, required=["name"]),
                self._load_skill,
            )
        )
        self._register(
            Tool(
                "spawn_subagent",
                "Ask a lightweight read-only subagent and return its completed result.",
                object_schema(
                    {
                        "role": string("Subagent role, such as planner or reviewer"),
                        "prompt": string("Question or task for the subagent"),
                        "context": string("Optional compact context for the subagent"),
                    },
                    required=["role", "prompt"],
                ),
                self._spawn_subagent,
            )
        )
        self._register(
            Tool(
                "start_subagent_task",
                "Run a read-only subagent task and keep its task record.",
                object_schema(
                    {
                        "role": string("Subagent role, such as planner or reviewer"),
                        "prompt": string("Question or task for the subagent"),
                        "context": string("Optional compact context for the subagent"),
                    },
                    required=["role", "prompt"],
                ),
                self._start_subagent_task,
            )
        )
        self._register(
            Tool(
                "read_subagent_task",
                "Read a saved subagent task by id.",
                object_schema({"task_id": string("Subagent task id")}, required=["task_id"]),
                self._read_subagent_task,
            )
        )
        self._register(
            Tool(
                "list_subagent_tasks",
                "List saved subagent tasks and statuses.",
                object_schema({}),
                self._list_subagent_tasks,
            )
        )
        self._register(
            Tool(
                "finish",
                "Finish the task with a concise summary and evidence.",
                object_schema({"summary": string("Final task summary")}, required=["summary"]),
                lambda args: ToolResult(args["summary"], terminate=True),
            )
        )

    def _run_shell(self, args: dict[str, Any]) -> ToolResult:
        result = self.workspace.run_shell(args["command"], int(args.get("timeout", 30)))
        content = json.dumps(
            {
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "output": result.output,
            },
            ensure_ascii=False,
            indent=2,
        )
        return ToolResult(content, metadata={"returncode": result.returncode})

    def _update_todo(self, args: dict[str, Any]) -> ToolResult:
        self.todos = list(args.get("todos", []))
        return ToolResult(json.dumps(self.todos, ensure_ascii=False, indent=2))

    def _update_task_graph(self, args: dict[str, Any]) -> ToolResult:
        self.task_graph = list(args.get("tasks", []))
        completed = {
            task["id"]
            for task in self.task_graph
            if task.get("status") == "done" and isinstance(task.get("id"), str)
        }
        ready = [
            task["id"]
            for task in self.task_graph
            if task.get("status") == "pending"
            and all(dep in completed for dep in task.get("depends_on", []))
        ]
        return ToolResult(
            json.dumps(
                {"tasks": self.task_graph, "ready": ready},
                ensure_ascii=False,
                indent=2,
            ),
            metadata={"ready": ready},
        )

    def _load_skill(self, args: dict[str, Any]) -> ToolResult:
        skill = self.skill_library.load(args["name"])
        return ToolResult(
            f"Loaded skill: {skill.name}\n\n{skill.content}",
            metadata={"skill": skill.name},
        )

    def _spawn_subagent(self, args: dict[str, Any]) -> ToolResult:
        if self.subagent_runner is None:
            return ToolResult("No model is attached for subagent calls.", is_error=True)
        task = self.subagent_runner.start(
            args["role"],
            args["prompt"],
            args.get("context", ""),
        )
        return ToolResult(
            task.result if task.status != "failed" else f"subagent failed: {task.error}",
            metadata={"role": args["role"], "task_id": task.id, "status": task.status},
            is_error=task.status == "failed",
        )

    def _start_subagent_task(self, args: dict[str, Any]) -> ToolResult:
        if self.subagent_runner is None:
            return ToolResult("No model is attached for subagent calls.", is_error=True)
        task = self.subagent_runner.start(
            args["role"],
            args["prompt"],
            args.get("context", ""),
        )
        return ToolResult(
            json.dumps(task.to_dict(), ensure_ascii=False, indent=2),
            metadata={"task_id": task.id, "status": task.status},
            is_error=task.status == "failed",
        )

    def _read_subagent_task(self, args: dict[str, Any]) -> ToolResult:
        if self.subagent_runner is None:
            return ToolResult("No subagent task runner is available.", is_error=True)
        task = self.subagent_runner.get(args["task_id"])
        if task is None:
            return ToolResult(f"Unknown subagent task: {args['task_id']}", is_error=True)
        return ToolResult(
            json.dumps(task.to_dict(), ensure_ascii=False, indent=2),
            metadata={"task_id": task.id, "status": task.status},
            is_error=task.status == "failed",
        )

    def _list_subagent_tasks(self, args: dict[str, Any]) -> ToolResult:
        if self.subagent_runner is None:
            return ToolResult("No subagent task runner is available.", is_error=True)
        tasks = [
            {
                "id": task.id,
                "role": task.role,
                "status": task.status,
                "updated_at": task.updated_at,
            }
            for task in self.subagent_runner.list()
        ]
        return ToolResult(json.dumps(tasks, ensure_ascii=False, indent=2))

    def _target_argument(self, arguments: dict[str, Any]) -> str:
        for key in ("path", "file_path", "target", "command"):
            value = arguments.get(key)
            if isinstance(value, str):
                return value
        return ""

    def _confirm(self, name: str, arguments: dict[str, Any], reason: str) -> bool:
        return self.ui.permission_requested(name, arguments, reason)


def string(description: str, default: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "description": description}
    if default is not None:
        schema["default"] = default
    return schema


def integer(description: str, default: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "integer", "description": description}
    if default is not None:
        schema["default"] = default
    return schema


def boolean(description: str, default: bool | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "boolean", "description": description}
    if default is not None:
        schema["default"] = default
    return schema


def object_schema(
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def validate_arguments(schema: dict[str, Any], value: dict[str, Any]) -> str:
    if not isinstance(value, dict):
        return "arguments must be a JSON object"
    allowed = set(schema.get("properties", {}))
    extra = sorted(set(value) - allowed)
    if extra and schema.get("additionalProperties") is False:
        return f"unexpected argument(s): {', '.join(extra)}"
    missing = [key for key in schema.get("required", []) if key not in value]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    for key, item in value.items():
        if key in schema.get("properties", {}):
            error = _validate_value(schema["properties"][key], item, key)
            if error:
                return error
    return ""


def _validate_value(schema: dict[str, Any], value: Any, label: str) -> str:
    expected = schema.get("type")
    if expected == "string" and not isinstance(value, str):
        return f"{label} must be a string"
    if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        return f"{label} must be an integer"
    if expected == "boolean" and not isinstance(value, bool):
        return f"{label} must be a boolean"
    if expected == "array":
        if not isinstance(value, list):
            return f"{label} must be an array"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                error = _validate_value(item_schema, item, f"{label}[{index}]")
                if error:
                    return error
    if expected == "object":
        if not isinstance(value, dict):
            return f"{label} must be an object"
        nested = validate_arguments(schema, value)
        return f"{label}.{nested}" if nested else ""
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        return f"{label} must be one of {enum}"
    return ""
