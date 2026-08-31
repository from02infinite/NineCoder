from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ninecoder.errors import PermissionDenied
from ninecoder.permissions import Decision, PermissionMode, evaluate_permission
from ninecoder.workspace import Workspace


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
    def __init__(self, workspace: Workspace, mode: PermissionMode):
        self.workspace = workspace
        self.mode = mode
        self.todos: list[dict[str, str]] = []
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
        permission = evaluate_permission(self.mode, name, self._target_argument(arguments))
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

    def _target_argument(self, arguments: dict[str, Any]) -> str:
        for key in ("path", "file_path", "target", "command"):
            value = arguments.get(key)
            if isinstance(value, str):
                return value
        return ""

    def _confirm(self, name: str, arguments: dict[str, Any], reason: str) -> bool:
        print(f"\nPermission required: {name} ({reason})")
        print(json.dumps(arguments, ensure_ascii=False, indent=2))
        reply = input("Allow once? [y/N] ").strip().lower()
        return reply in {"y", "yes"}


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
