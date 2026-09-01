NineCoder 是一个从零实现的轻量级 coding agent，纯 Python 标准库（Python 3.10 以上，无第三方运行时依赖），调用 OpenAI 兼容接口，默认模型 deepseek-v4-flash。模型只判断下一步动作，agent 循环、工具、文件读写、命令执行、错误反馈与轨迹保存均由项目自身实现。

Git 仓库地址：
https://github.com/from02infinite/NineCoder

如何运行：
1. 设置 API key：
   export DEEPSEEK_API_KEY="你的 key"
2. 安装项目：
   python -m pip install -e .
3. 创建演示工作区：
   python scripts/create_demo_workspace.py
4. 运行一次 agent：
   ninecoder --workspace demo_workspace "修复 bug 并运行测试"
5. 自动权限并指定验证命令：
   ninecoder --permission auto --workspace demo_workspace \
     --test "python -m unittest -q" "让 divide 在 b 为零时抛出 ValueError"
6. 恢复旧会话：
   ninecoder --workspace demo_workspace --resume 会话ID
7. 输出便于脚本处理的 JSON 结果：
   ninecoder --json --workspace demo_workspace "修复 bug 并运行测试"
8. 运行本地测试：
   pytest
   （或 PYTHONPATH=src python -m unittest discover -s tests）

交互模式（REPL）：
在终端直接运行 ninecoder（不带任务参数）即进入交互 REPL，可持续对话。常用命令：
   <消息>        继续当前对话
   /new          开启全新对话
   /resume [id]  恢复某个会话（省略 id 时从列表选择）
   /switch <id>  从旧会话分叉出新对话
   /compact      压缩当前上下文
   /tree         显示会话树
   /list         列出所有已保存会话
   /help         查看帮助
   exit / quit   退出 REPL

特色功能说明：
一、Agent 循环。从零手写 ReAct 执行环，模型只选下一步动作，调度、工具派发、错误反馈、终止判断与总结由项目自控；max_steps 限制迭代，格式错误进入修正提示；每次运行在 runs/*.jsonl 记录 run_start、model_response、tool_result、run_end 等事件。

二、工具系统。统一 ToolRegistry，每个工具带 JSON schema；内置 list_files、read_file、search、edit_file、write_file、run_shell、update_todo、update_task_graph、load_skill、subagent、finish 等。edit_file 精确替换并返回 unified diff；工具异常包装为 ToolResult 回传，模型据此自我修正。

三、权限与安全。plan 只读、ask 写前确认、auto 自动三种模式；敏感路径（.env、SSH 私钥、密钥）默认拒绝；shell 支持超时、危险命令拦截、输出截断；可插拔沙箱（bwrap、sandbox-exec）默认阻止网络，可用 --allow-network 开启。

四、会话持久化。SessionState 完整落盘（消息、todo、任务图、子 agent、状态、摘要），支持恢复、分叉、压缩，记录 compaction_floor 防回退。

五、上下文压缩与记忆。保留最近消息、旧消息摘要化、超长工具输出落盘引用；摘要缓存到 runs/context/；跨运行维护 MEMORY.md 提炼稳定事实注入系统提示，可 --no-memory 关闭。

六、Skills 与子 agent。Markdown Skill 按需加载，启动只注入清单；支持 spawn_subagent 等只读委派；update_todo 维护清单，update_task_graph 维护依赖任务图。

七、Hook 与本地能力。before_agent_start、before_model、after_model、before_tool、after_tool、on_stop、on_finish 可改写上下文、拦截工具；before_tool 返回 ToolDecision 阻断危险操作；LocalCapabilityRouter 提供 tools/list 与 tools/call。

八、终端体验。Rich TUI 与纯文本（--plain）双模式；流式输出（--no-stream 关闭）；--json 输出摘要、步数、轨迹路径、会话 id、停止原因等。

安全：API key 只读环境变量；文件写入限制在 workspace 内；shell 默认沙箱断网；处理不可信输入建议加 --permission ask。
