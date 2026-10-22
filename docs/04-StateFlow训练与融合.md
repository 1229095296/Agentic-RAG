# 04｜StateFlow 训练与融合

## 1. 原项目不是知识库构建器

原 `stateflow-musique-release` 实现模型如何提出查询、读取外部材料、输出证据状态和接受训练。它没有文件解析入库、Chunk 索引或检索服务实现。

新增 `enterprise-demo` 补齐最小本地检索底座。二者的实际接缝是 HTTP `/retrieve`，并非共享一个 Python KnowledgeBase 实例。

## 2. 完整起点的真实模型循环

阅读 `generate_stateflow_direct.py::generate`：

1. 使用外部 Slime 的 GenerateState 获取 tokenizer，构建模型服务地址。
2. 构建问题 prompt，初始 phase 为 search。
3. 请求模型生成一个完整动作。
4. 解析 `<search>...</search>`，调用 `_retrieve`。
5. 将结果转成可见 spans，包装 `<information>` 并追加上下文。
6. phase 变为 evidence，要求模型输出完整 `<evidence_state>`。
7. Gate 接受后转换成池状态，保存轮次；可引用集合替换为已保留证据。
8. decision 为 search 则再次检索，为 answer 则只允许最终答案动作。
9. 记录 response tokens、logprobs、loss_mask 与 trace，交给外部训练系统。

Gate 拒绝直接终止，属于 completed 零奖励负样本，不是基础设施 abort。检索服务故障目前会被转换为空结果，这一设计应在企业化时改为显式异常类型。

模型 token 的 loss mask=1，检索文本／环境指令 mask=0；这避免把服务端返回文本当成模型要学习生成的目标，但模型仍会看到它作为上下文。

## 3. 三种状态不要混淆

| 状态 | schema | 目的 |
|---|---|---|
| DirectEvidenceState | evidence / open_questions / decision | 模型可见、模型输出的严格状态 |
| EvidenceState | E / U / H / B + 版本与轨迹字段 | 成功状态持久化与训练回放 |
| RunState | question / hits / gate / citations 等 | 新演示的框架运行状态 |

池里的 E 是证据，U 是未解约束，H 为历史字段，B 为预算字段。Direct 转池时，部分字段可为空；不要假定存在完整可恢复历史。

`evidence_state.py` 中旧 Canonicalizer 会做规则规范化，但不是 direct 主入口的执行路径。当前算法强调不自动修复模型状态。

## 4. GRPO 与中间状态回放

### full-start

从同一问题复制 G 个样本，生成完整轨迹作为同一 GRPO 组。

### harvest

`harvest_states.process` 选择已完成、达到奖励阈值、非 replay 的轨迹，检查来源与标记泄漏后将中间状态写到 JSONL 池。replay 产生的状态不回写主池，避免从不可真实到达的状态不断繁殖。

### replay-start

`StateFlowDataSource` 从池选择 `(question, S_k)`，序列化起点并 fan-out 为另一组后缀轨迹。full 和 replay 使用不同组，不能混合难度后归一化。

注意：具体 GRPO 优势计算和参数更新在外部 Slime 中，本仓库只提供采样与 rollout 接口。

### 自适应课程

启用相应配置且 total rollout 信息有效时，前 50 步全 full，50–499 步混合并逐渐降低 replay，500 步后回到全 full。未启用时采用固定比例。

锚点后的成功率高，向更早、更难的位置移动；低则向更晚、更容易的位置移动。EMA 统计按问题和 turn_index，不是每条精确证据链。

## 5. 奖励实际用了什么

`generate_stateflow_direct.reward_func` 在训练时调用 `compute_score(include_evidence=False)`。因此当前 direct 版本不是在优化所有证据/path/retrieval 分项，核心是：

- 答案质量：EM 正确为 1，否则使用部分 token F1。
- 可执行动作格式。
- 查询去重及相对 hop 的轮次效率。
- 硬协议失败归零。

评测返回二值 EM，并另记 F1。效率项不是实际货币／延迟成本，也不是检索越少越好。

## 6. 接上本地检索服务

第一终端启动本地演示并导入资料。然后在原训练环境启动前配置：

```bash
export MUSIQUE_SEARCH_URL=http://127.0.0.1:8765/retrieve
export MUSIQUE_TOPK=3
```

先验证契约：

```bash
curl -X POST "$MUSIQUE_SEARCH_URL" \
  -H 'Content-Type: application/json' \
  -d '{"queries":["差旅报销"],"topk":3,"return_scores":false}'
```

原 `_retrieve` 会读取 result[0]，并使用返回 spans。已有 spans 时不会走英文 fallback 分句。

**若训练在另一台机器或容器中，127.0.0.1 指向训练进程所在环境，而不是当前电脑。** 正式远程部署必须另建经过认证的检索服务，不要为了连通直接暴露当前无认证演示。

MuSiQue 问题不一定能在企业示例文档中找到答案。连通协议并不等于训练数据和检索语料匹配。训练集要与目标知识库一致，gold 仅供奖励，不进入策略输入。

## 7. 原项目 Slime 钩子

以下是原 README 提供的训练器参数，不是独立可运行命令：

```text
--custom-generate-function-path slime_integration.generate_stateflow_direct.generate
--custom-rm-path slime_integration.generate_stateflow_direct.reward_func
--data-source-path slime_integration.stateflow_data_source.StateFlowDataSource
--rollout-all-samples-process-path slime_integration.harvest_states.process
```

还必须自行提供：兼容 Slime 安装、模型与 tokenizer、SGLang 服务、优化训练配置、数据路径、计算资源、checkpoint 和输出目录。没有这些信息，不能承诺一条命令就能训练。

数据转换入口是 `musique_to_slime.py`。先检查它接受的 id/query/reference_answer 等字段，别把任意原版数据直接套入转换器。配置文件 `.env` 不会自动读取，需由启动器加载或显式导入环境变量。

## 8. 保留原样的已知风险

本次以文档和演示补全为目标，没有暗改原研究算法。训练前至少处理：

1. `_em_scalar` 将标量满分当成功，可能把正确非满分 replay 计成失败；应读取 reward_detail.em。
2. replay 搜索计数从 0 开始，未恢复 anchor 剩余预算。
3. replay decision 阶段允许绕过未解问题直接回答。
4. fallback d0/d1 文档标识可能跨轮碰撞；新增服务返回稳定 spans 可避免这条路径，但没修原 fallback。
5. pool 读取采样缺当前版本过滤、去重签名偏粗、多进程缓存不一致。
6. 采集钩子缺明确 eval 排除；生产训练必须隔离 split、运行和版本。
7. 全历史仍拼进模型上下文，没有真实压缩。
8. Gate 不验证答案语义支持；防标记泄漏不是安全认证或 ACL。

进一步证据和设计建议见根目录 [分析与融合方案](../分析与融合方案.md)。该分析保留了最初检查时测试依赖缺失的历史情况；最新验证状态以验证记录为准。

## 9. 训练和生产推理的正确边界

线上请求不需要 GRPO 分组或标准答案；保留检索、证据状态、Gate、预算和恢复即可。带 gold 的训练池不能给线上 Agent 当长期记忆。

后续最合理的重构是共享 Runtime 状态转移，分别提供在线模型适配器与 Slime 训练记录器。当前离线 Runtime 只是规则演示，尚未完成这项重构。
