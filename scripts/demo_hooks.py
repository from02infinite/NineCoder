"""演示 NineCoder 的 hook 扩展点。

用假模型（不联网、不需要 API key）真实走一遍 agent 循环，
把每个 hook 的触发过程打印出来。运行：

    python scripts/demo_hooks.py
"""
from __future__ import annotations

from typing import Any

from ninecoder.agent import AgentConfig, CodingAgent
from ninecoder.hooks import (
    AgentStartRequest,
    ModelRequest,
    StopEvent,
    ToolDecision,
    ToolRequest,
    ToolResponse,
)
from ninecoder.model_client import ModelResponse, ToolCall
from ninecoder.permissions import PermissionMode
from ninecoder.workspace import Workspace


# 假模型：第一步让 agent 调用 write_file，之后调用 finish。
class DemoModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, tools) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                "write",
                [ToolCall("call_1", "write_file", {"path": "hello.txt", "content": "original text"})],
            )
        return ModelResponse("finish", [ToolCall("call_2", "finish", {"summary": "完成了"})])


# 1. 启动时追加一段策略到 system prompt
class StartupHook:
    def before_agent_start(self, request: AgentStartRequest) -> AgentStartRequest:
        print("[hook] before_agent_start -> 追加启动策略")
        return AgentStartRequest(
            request.task, request.workspace, request.test_cmd,
            request.system_prompt + "\n\n# Hook 注入的启动策略。",
        )


# 2. 每次调模型前打印消息条数
class ModelHook:
    def before_model(self, request: ModelRequest) -> ModelRequest:
        print(f"[hook] before_model -> 当前 {len(request.messages)} 条消息")
        return request


# 3. 改写 write_file 的参数（content 换成大写）
class RewriteToolHook:
    def before_tool(self, request: ToolRequest) -> ToolRequest:
        if request.name == "write_file":
            new_args = request.arguments | {"content": request.arguments["content"].upper()}
            print(f"[hook] before_tool -> 改写 write_file 参数: {new_args!r}")
            return ToolRequest(request.name, new_args)
        return request


# 4. 阻断 delete_file（演示 ToolDecision 阻断执行）
class BlockDeleteHook:
    def before_tool(self, request: ToolRequest) -> ToolDecision | None:
        if request.name == "delete_file":
            print("[hook] before_tool -> 阻断 delete_file")
            return ToolDecision(blocked_result=ToolResponse("hook 策略禁止删除", is_error=True))
        return None


# 5. 改写 finish 的最终摘要
class ResultHook:
    def after_tool(self, response: ToolResponse) -> ToolResponse:
        if response.terminate:
            print(f"[hook] after_tool -> 改写最终摘要（原: {response.content!r}）")
            return ToolResponse("由 hook 改写后的摘要 ✨", terminate=True)
        print(f"[hook] after_tool -> 结果 {len(response.content)} 字符")
        return response


# 6. 记录停止事件
class StopHook:
    def on_stop(self, event: StopEvent) -> None:
        print(f"[hook] on_stop -> 步数={event.steps} 原因={event.stopped_by}")


def main() -> None:
    hooks = [
        StartupHook(),
        ModelHook(),
        RewriteToolHook(),
        BlockDeleteHook(),
        ResultHook(),
        StopHook(),
    ]
    ws = Workspace("/tmp/demo_hooks_ws")
    result = CodingAgent(
        DemoModel(),
        ws,
        AgentConfig(max_steps=5, permission_mode=PermissionMode.AUTO, memory=False),
        hooks=hooks,
    ).run("写一个 hello.txt")

    print("\n===== 结果 =====")
    print(f"最终摘要: {result.summary}")
    print(f"停止原因: {result.stopped_by}")
    print(f"文件内容: {(ws.root / 'hello.txt').read_text(encoding='utf-8')!r}")


if __name__ == "__main__":
    main()
