Git 仓库地址：https://github.com/from02infinite/NineCoder

NineCoder 是一个从零实现的轻量 coding agent，默认使用 deepseek-v4-flash，并兼容 OpenAI 风格 Chat Completions API。模型只负责决定下一步动作；对话历史、工具定义、本地读写、命令执行、错误反馈、循环终止和轨迹保存均由项目自行实现。

运行方式：
1. 设置环境变量：export DEEPSEEK_API_KEY="你的 key"
2. 安装：python -m pip install -e .
3. 运行：ninecoder --workspace demo_workspace "修复 bug 并运行测试"

特色功能：支持 read/list/search/edit/write/bash/finish/todo/skill/subagent 等工具；edit 使用精确 search/replace 并返回 diff；shell 命令有超时和危险命令拦截；支持 plan/ask/auto 权限模式；每一步写入 runs/*.jsonl，便于复盘和视频展示；可按需加载 skills，可调用 Planner/Reviewer 风格的轻量子 agent 辅助分析，并提供 before_tool/after_tool/on_finish hook 扩展点。

安全说明：API key 只从环境变量读取，不写入仓库；写文件限制在工作目录内，敏感路径如 .env、.ssh、私钥等默认拒绝。
