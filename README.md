# NineCoder

NineCoder 是一个从零实现的轻量级 coding agent。它会与兼容 OpenAI
接口的模型对话，暴露本地工具，在有边界的循环中执行工具，并将每一步记录为
JSONL。

默认模型：`deepseek-v4-flash`。

## 快速开始

```bash
export DEEPSEEK_API_KEY="..."
python -m pip install -e .
ninecoder --workspace demo_workspace "修复 bug 并运行测试"
```

创建演示工作区：

```bash
python scripts/create_demo_workspace.py
ninecoder \
  --permission auto \
  --workspace demo_workspace \
  --test "python -m unittest -q" \
  "当 b 为零时，让 divide 抛出 ValueError('division by zero')，然后运行测试"
```

继续一个未完成的运行：

```bash
ninecoder --workspace demo_workspace --resume 20260831-120000-000000
```

在交互式 REPL 中选择旧会话并继续：

```text
/resume
```

Rich TUI 模式会打开一个键盘选择器：使用 Up/Down 选择已保存的会话，然后按
Enter。使用 `/resume <id>` 可以直接跳转；如果想从旧上下文分叉出一个新会话，
而不是继续原会话，可以使用 `/switch <id>`。

输出便于脚本处理的最终报告：

```bash
ninecoder --json --workspace demo_workspace "修复 bug 并运行测试"
```

运行本地测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

常用环境变量：

- `DEEPSEEK_API_KEY` 或 `NINECODER_API_KEY`
- `NINECODER_BASE_URL` 或 `DEEPSEEK_BASE_URL`
- `NINECODER_MODEL`
- `NINECODER_MAX_RETRIES`（临时模型错误的重试次数，默认 `3`）
- `NINECODER_SANDBOX`（沙箱后端，默认 `auto`）
- `NINECODER_STREAM`（设为 `0` 可关闭模型输出流式显示）

## 安全性

NineCoder 不会存储 API key。文件写入被限制在所选工作区内，敏感路径
（`.env*`、SSH/AWS/GCP 密钥和目录）会被拒绝访问，shell 命令有超时限制，
权限模式可以设置为 `plan`、`ask` 或 `auto`。

`run_shell` 默认在操作系统沙箱中运行（`--sandbox auto`）：它会自动检测
`bwrap`（Linux，隔离能力完整）或 `sandbox-exec`（macOS，已废弃且仅尽力而为），
如果两者都未安装，则会回退到无沙箱模式并显示提示。沙箱会让工作区和 `/tmp`
之外的文件系统只读，为命令提供新的命名空间，并且**默认阻止网络访问**。需要
联网的命令（例如 `pip install`、`git clone`）可以传入 `--allow-network`。
显式后端包括 `--sandbox bwrap` 和 `--sandbox sandbox-exec`；`--sandbox off`
会完全关闭沙箱。

沙箱提供的是文件系统、进程和网络隔离，并不是完整的密钥保险箱：沙箱中的命令
仍然可以*读取* `~/.ssh` 等路径（只是在网络被阻止时无法把内容发送出去），而
`sandbox-exec` 是 Apple 已废弃的兼容层，它的 profile 语义弱于 bubblewrap。
敏感路径拒绝策略和 `--permission ask` 仍然是读取访问的防线。当模型处理不可信
输入时，建议在沙箱之外再使用 `--permission ask`。

## 已实现的 Agent 功能

- 自写的 agent 循环
- 带 JSON schema 定义的工具注册表
- 工具：bash、read、write、edit、glob/list、grep/search、todo、finish
- 按需加载 Markdown skill
- 只读子 agent 任务，支持 id、状态、独立上下文和已保存结果
- 简单的带依赖任务图
- 可恢复的会话状态，包含消息、todos、任务图、子 agent 任务和状态
- 通过近期窗口保留、摘要文件和长工具输出存储实现上下文压缩
- JSONL 轨迹持久化
- 类 MCP 的本地能力路由，支持 `tools/list` 和 `tools/call`
- Hook 扩展点，可检查或改写 agent 启动、模型请求、模型响应、工具调用、工具结果和停止事件；工具 hook 也可以阻断执行并返回合成结果
- `plan`、`ask` 和 `auto` 权限治理
- `run_shell` 可插拔操作系统沙箱（bubblewrap / sandbox-exec），默认阻止网络访问
- 模型输出流式显示（SSE），plain 模式会实时渲染 token（可用 `--no-stream` 关闭）
