# 企业知识检索 · 离线原型

## 如何体验

双击本文件夹中的「启动演示.command」，保持窗口打开。浏览器会打开 http://127.0.0.1:8765 。关闭启动窗口或按 Control+C 停止。端口占用时请关闭之前启动的演示，不要重复启动。

1. 点击「导入示例制度」，或选择本地 PDF、DOCX、MD、TXT 文件。
2. 点击「导入文件」完成解析、分片、版本化与索引。
3. 输入关键词，例如「差旅报销需要什么材料？」。
4. 查看原文摘录、来源、证据标识与 Gate 记录。

当前没有持续后台服务；已用真实 HTTP 服务完成短时启动验证并自动关闭。正式体验请双击启动入口。

## 已实现

- PDF 文字提取（页码来源）、Word 段落与表格、UTF-8 Markdown/TXT。
- LlamaIndex SentenceSplitter：500 字符规模，70 字符重叠；明确使用字符 tokenizer 而非模型 token。
- SQLite 文档目录、稳定文档标识、内容与管线版本、同名更新、重复导入去重。
- Qdrant 本地持久化索引；启动时根据 SQLite 重建，可恢复向量写入中断。
- 字符 hashing 向量召回与字符 TF-IDF 检索，使用 RRF 融合。
- LangGraph 执行 retrieve → build_evidence → gate → extract。
- 原项目 direct_evidence_state.py 的 Gate 原样复制为 stateflow_gate.py，原项目未修改。
- 与原项目 HTTP 契约兼容的 POST /retrieve，返回 result[query_index]、document、spans。
- 本地文件上传页面、检索与证据展示，不使用 CDN 或云模型。

## 非目标与限制（重要）

这是完整入库检索链路的本地原型，不是完成的企业生产系统，也不是真正模型自主决策的 Agent。

- 未接大模型，没有多跳自主查询、语义规划、答案综合、语义支持验证或 GRPO 训练。离线图只走一次检索。
- Hashing 向量是词面字符特征，不是训练过的语义 Embedding；同义词跨表达召回较弱。Qdrant 的使用不等于具备语义理解。
- 证据状态保留 search 与未解问题，不能据检索命中宣称 answer-ready。摘录节点仅展示原文，不执行自动回答。
- Gate 检查摘录与可见原文、schema，不检查事实真实性或证据充分性。
- 原 Gate 的中文子串摘录词数规则未修改；演示选择完整 span 避免该特定限制，不代表已修复。
- 不含 OCR、旧 DOC、复杂 PDF 版面、图片与复杂表格结构恢复。文件上限 10 MB。
- DOCX 表格保留单元格文字和阅读顺序，但大表格切分不保证重复表头。
- 不保存原始文件副本，只保存提取的文本、来源和版本。metadata 有位置但尚无精确字符区间。
- 每次入库与启动重建全部本地向量，适合小规模演示，不适合海量资料。
- 单用户、单进程，只绑定 127.0.0.1，无身份认证或 ACL。禁止直接暴露公网或当作企业权限系统。
- 本地目录包含提取的文档正文，应按敏感数据管理；未实现删除接口、保留策略或加密。
- 没有线上 checkpoint、分布式一致性或训练池对接；sentence_utils.py 保留为原项目参考，实际中文切句在 knowledge.py。

## 与现有 StateFlow 代码结合

1. 启动本演示。
2. 将原项目 MUSIQUE_SEARCH_URL 配置为 http://127.0.0.1:8765/retrieve。
3. 原项目的 _retrieve 可读取本演示返回的 result 和 spans。
4. 原项目的模型生成服务及 Slime 训练环境仍需要另行部署。本次没有启动训练或运行真实模型 rollout。
5. 原项目 MUSIQUE_CORPUS_VERSION 应对应实际知识库版本；本原型只提供逐文档版本，没有实现跨知识库快照认证。

## 模型升级接口边界

后续应增加：

- EmbeddingProvider：与 query/index 同时使用，变更模型时重建索引；可配置企业本地模型。
- PolicyProvider：统一输出 search、evidence_state、answer，并接入预算受限的循环边。
- AnswerVerifier：引用匹配与事实支持，不以 Gate 代替语义验证。
- 文档源连接器、更新/删除同步、身份与 ACL，在召回之前限制可访问文档。
- 企业服务：独立 Qdrant 服务、持久检查点、审计与评测。

当前没有实现以上模型适配器，不需要填写密钥。离线运行不会把企业文档发送到外部模型。

## 开源仓库选型（查询日期：2026-10-22）

| 仓库 | 链接 | 角色 | 许可与说明 |
|---|---|---|---|
| LangGraph | https://github.com/langchain-ai/langgraph | 状态流 | MIT；本次实际使用 StateGraph |
| LlamaIndex | https://github.com/run-llama/llama_index | 数据处理、分片 | MIT；本次只使用 core，不使用云解析 |
| Qdrant Client | https://github.com/qdrant/qdrant-client | 本地向量库 | Apache-2.0；官方明确本地模式面向原型与测试 |
| RAGFlow | https://github.com/infiniflow/ragflow | 可选的一体化知识库底座 | Apache-2.0；本次未安装。当前官方说明 macOS 部署不受支持，且依赖较重 |

这是组件组合，不是把整个第三方仓库复制到项目。官方 README 为选型信息来源；不要将商业云功能等同于开源框架功能。以上许可证只指列出的开源组件，不能据此推断原 StateFlow 代码的授权；对外分发前需确认原项目许可。

## 开发验证

使用 Python 3.12、Apple Silicon。依赖已经由 offline_packages 离线安装；requirements.lock.txt 记录本次实装版本。

- 单元／集成测试：`.venv/bin/python -m pytest -q`
- 有界真实 HTTP 启动测试：`.venv/bin/python smoke_server.py`（自动停止）
- 前台启动：`.venv/bin/python launcher.py`

已通过：6 组测试，包括离线网络禁用下入库检索、持久化重载、版本替换、ID 唯一性、分片、DOCX 表格解析、伪造证据拒绝、HTTP 上传／检索／状态链路。
真实 HTTP 启动测试通过：health、示例入库、ask、retrieve；随后服务正常停止。
浏览器视觉交互尚未自动化验证；PDF 支持由代码提供，但当前回归测试没有真实 PDF 样本。

## 文件说明

- knowledge.py：解析、chunk、索引和检索。
- runtime.py：LangGraph 离线状态流。
- app.py：本地接口。
- index.html：操作页面。
- stateflow_gate.py：原始 Gate 副本。
- test_demo.py / smoke_server.py：测试。
- sample.md：明确标记为虚构的制度演示材料。
- data/：首次启动后创建；保存文本与索引，不宜公开。

## GitHub 整理与完整教程（2026-10-22）

本目录在完整仓库中名为 enterprise-demo。重新克隆仓库不会包含安装环境，必须先按 [使用教程](../docs/01-使用与演示教程.md) 安装依赖；不能直接双击启动期待环境自动出现。

完整入口见 [仓库首页](../README.md)，包括代码结构、API 数据流、训练融合、排障和验证记录。本次整理后重新验证演示 6 项测试、真实 HTTP 服务，以及原 StateFlow 的 22 项逻辑测试；没有运行真实模型训练。
