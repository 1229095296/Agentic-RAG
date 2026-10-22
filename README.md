# Agentic-RAG：企业知识检索原型与 StateFlow-GRPO

这个仓库将两部分放在一起：**可以本地体验的离线知识检索演示**，以及**原始 StateFlow-GRPO 多跳检索训练代码**。前者补齐文档入库与检索服务，后者提供模型检索动作、证据状态和成功状态回放机制。

> **请先区分：离线演示不含大模型，不是完整的自主问答 Agent；原训练项目需要外部模型服务、Slime 和数据集。两者通过 `/retrieve` 契约对接，本仓库尚未完成真实模型端到端训练联调。**

## 从这里开始

| 你的目标 | 阅读入口 |
|---|---|
| 第一次运行，完整演示 | [使用与演示教程](docs/01-使用与演示教程.md) |
| 理解每个目录、类和函数 | [代码结构与实现导读](docs/02-代码结构与实现导读.md) |
| 调用接口、追踪数据结构 | [接口与数据流](docs/03-接口与数据流.md) |
| 理解 GRPO、Gate、回放与如何对接 | [StateFlow训练与融合](docs/04-StateFlow训练与融合.md) |
| 测试、故障排查和生产升级 | [测试排障与企业化路线](docs/05-测试排障与企业化路线.md) |
| 查看真实验证记录 | [验证记录](docs/06-验证记录.md) |

## 仓库结构

```text
Agentic-RAG/
├── enterprise-demo/             # 新增：可运行离线知识检索原型
│   ├── knowledge.py            # 文件解析、分片、持久化、索引与检索
│   ├── runtime.py              # LangGraph 状态流
│   ├── stateflow_gate.py       # 从原项目复制的严格证据校验
│   ├── app.py                  # FastAPI 与兼容 /retrieve 接口
│   ├── index.html              # 文件上传与检索页面
│   ├── launcher.py             # 本地前台启动
│   ├── test_demo.py            # 逻辑和接口测试
│   ├── smoke_server.py         # 短时真实服务测试，结束后停止
│   ├── requirements*.txt       # 依赖约束与本次实装版本
│   └── 启动演示.command         # macOS 双击入口
├── stateflow-musique-release/   # 已有：原始 StateFlow-GRPO，保持原样
├── docs/                       # 中文教程与验证记录
├── 分析与融合方案.md             # 初次代码检查与架构分析（保留当时检查状态）
└── THIRD_PARTY_NOTICES.md       # 依赖来源与原代码授权边界
```

## 五分钟快速开始

需要 Python 3.12。首次安装依赖需联网；安装后演示运行不需要模型服务或模型下载。

```bash
git clone https://github.com/1229095296/Agentic-RAG.git
cd Agentic-RAG/enterprise-demo
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python launcher.py
```

Windows 使用 `.venv\Scripts\python.exe` 替代 `.venv/bin/python`，其他系统见教程；目前实际验证环境为 macOS arm64 / Python 3.12。

浏览器访问 http://127.0.0.1:8765 ，点击「导入示例制度」，检索「差旅报销需要什么材料？」。可查看原文、文件来源、证据状态与 Gate。保持启动窗口打开，按 Control+C 停止。macOS 完成安装后也可双击「启动演示.command」。

## 当前能力与边界

| 能力 | 状态 |
|---|---|
| PDF 文字、DOCX、Markdown、TXT 解析 | 已实现；扫描件 OCR 不支持 |
| Chunk 分片、Span 标识、文档版本、同名替换 | 已实现 |
| 本地 Qdrant + 字符 TF-IDF 的 RRF 融合 | 已实现；不是语义 Embedding |
| 证据来源、原文匹配、schema 校验 | 已实现；不保证事实或答案正确 |
| 页面、上传与兼容检索接口 | 已实现，仅本地单用户 |
| 模型自主多轮检索、答案综合 | 离线演示未实现；原训练代码提供相应动作循环 |
| 完整 Slime 训练部署 | 本仓库不提供启动器、模型、数据与集群配置 |
| 企业 SSO、ACL、增量同步、删除、审计治理 | 待建设 |

没有下载依赖、虚拟环境、用户文档数据库、模型权重或凭据随代码提交。示例制度是虚构资料。

## 测试

```bash
cd enterprise-demo
.venv/bin/python -m pytest -q
.venv/bin/python smoke_server.py
```

验证结果及覆盖边界见 [验证记录](docs/06-验证记录.md)。浏览器视觉和真实模型训练不在本次验证范围。

## 开源组件

- [LlamaIndex Core](https://github.com/run-llama/llama_index)：分片。
- [Qdrant Client](https://github.com/qdrant/qdrant-client)：本地向量索引。
- [LangGraph](https://github.com/langchain-ai/langgraph)：显式状态图。
- [FastAPI](https://github.com/fastapi/fastapi)：HTTP 服务。

原 StateFlow 代码的许可未在源目录声明。本次不擅自为整个仓库新增统一开源许可证，使用与再分发前请确认权利，见 [来源说明](THIRD_PARTY_NOTICES.md)。
