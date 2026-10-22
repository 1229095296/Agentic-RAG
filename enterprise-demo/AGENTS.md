# 项目说明

## 项目概述
本地离线知识入库、检索与 StateFlow 证据协议演示；不是生产企业系统或模型自主 Agent。

## 技术栈
Python 3.12；LlamaIndex SentenceSplitter；Qdrant 本地模式；SQLite；scikit-learn 字符特征；LangGraph StateGraph；FastAPI；无外部前端依赖的本地页面。

## 实装与验证
依赖从 offline_packages 安装到独立 .venv，requirements.lock.txt 为实际版本。
6 组 pytest 测试已通过；smoke_server.py 完成真实 HTTP 启动、请求和关闭。浏览器视觉及真实 PDF 样本未验证。

## 注意事项
保持完全离线，不调用云模型，不添加遥测。不修改原始 sfgrpo-main 项目。
stateflow_gate.py 是原项目校验文件的未修改副本，sentence_utils.py 为参考副本。未经确认不得将原项目代码视作具有第三方组件同样的开源许可。
运行时只能输出原文摘录，不要宣传语义理解、模型回答或自动多跳。证据状态 search 不代表当前图会循环：离线图没有策略模型，始终返回待人工判断。
服务仅绑定回环接口，单进程；未实现 ACL，勿暴露网络。
开发测试用临时数据库；真实用户数据位于 data，不得默认删除。
后续模型策略和 Embedding 都是待实现扩展点，不是现成功能。
用户通过「启动演示.command」前台运行服务；当前会话没有持续托管进程。

## 仓库文档整理（2026-10-22）
完整教程位于 ../docs，根 README 是用户入口。本目录在 Git 中为 enterprise-demo；不含 .venv、离线包或数据。克隆后必须先安装依赖。
待提交副本重新通过 6 个演示测试和真实 HTTP smoke；原项目 22 个逻辑测试通过。不要将此解读为模型训练验证。
