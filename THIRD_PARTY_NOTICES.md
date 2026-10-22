# 来源与授权说明

## 原始 StateFlow 代码

`stateflow-musique-release/` 已存在于本 GitHub 仓库，本次完整保留，未修改算法。`enterprise-demo/stateflow_gate.py` 为其中 `slime_integration/direct_evidence_state.py` 的原样副本；`sentence_utils.py` 为同名参考副本。

收到的原目录没有 LICENSE 文件，本次不替原作者声明开源授权，也不为整个仓库添加统一许可证。公开可见不等于自动授予复制、修改和再分发权利；发布者及后续使用者需自行确认原代码权利。

## 通过依赖安装的开源组件

| 组件 | 上游 | 常见上游许可证 |
|---|---|---|
| LlamaIndex Core | https://github.com/run-llama/llama_index | MIT |
| LangGraph | https://github.com/langchain-ai/langgraph | MIT |
| Qdrant Client | https://github.com/qdrant/qdrant-client | Apache-2.0 |
| FastAPI | https://github.com/fastapi/fastapi | MIT |
| Uvicorn | https://github.com/encode/uvicorn | BSD-3-Clause |
| scikit-learn | https://github.com/scikit-learn/scikit-learn | BSD-3-Clause |
| pypdf | https://github.com/py-pdf/pypdf | BSD-3-Clause |
| python-docx | https://github.com/python-openxml/python-docx | MIT |

本表用于来源指引，不代替每个已安装版本及其传递依赖的 LICENSE/NOTICE。发布二进制、镜像或集成第三方源码时需按实际版本单独审查许可证义务。本仓库提交 Python 源码及依赖清单，不捆绑 wheel、模型或第三方完整仓库。

## 示例与数据

sample.md 为虚构演示资料。用户上传的文档、索引、训练池和轨迹不属于随代码发布的样例，不得默认提交到 Git。
