"""证据状态与规范化（S3.1）。

对应 STATEFLOW_DESIGN.md 决策 A：抽取式证据状态——模型选 span 编号做语义决策，规则回填原文做可信记账。
本模块是**纯规则**部分（不调 LLM、不依赖 GPU/检索/torch），可在 CPU 单测。

包含：
- EvidenceCapsule / EvidenceState 数据结构（(E,U,H,B)）。
- parse_state_update：解析模型输出的 <state_update>{...}</state_update> JSON（Δ_t）。
- Canonicalizer.canonicalize(S_t, O_t, Δ_t) -> S_{t+1}：校验 span 存在且来自本轮可见结果 → 回填原文/来源
  → 去重 → 限长 → 更新 U/H/B → 拒绝非法字段 → 固定顺序。
- render_state(S) -> str：固定模板渲染（模型下一轮看到的文本，不是 JSON、不是自由摘要）。
- serialize_for_replay(S) -> str：把中间状态 S_k 序列化成 replay 起点 prompt，严格白名单防泄漏。
- assert_no_leakage：CI 断言序列化不含 gold answer / 未来动作。

因果约束（设计文档 §5.5）：S_{t+1} 只用 S_t + 当前动作 + 当前检索结果；不得使用后续结果/最终答案/gold。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

# ---------------- V1 初始限制（STATEFLOW_DESIGN §4.3） ----------------
MAX_EVIDENCE = 6            # 每状态最多证据条数
MAX_SPANS_PER_EVIDENCE = 2  # 每条证据最多连续句子数
MAX_EVIDENCE_CHARS = 600    # 每条证据回填文本字符上限（~80-120 token 的宽松字符近似）
MAX_OPEN_CONSTRAINTS = 4    # 最多未解约束
MAX_FAILED_DIRECTIONS = 3   # H 中最多记录的失败/历史搜索方向
VALID_ROLES = {"direct", "bridge"}
VALID_SEMANTIC_STATUS = {"selected", "tentative", "conflicting", "superseded"}


@dataclass
class EvidenceCapsule:
    evidence_id: str
    span_ids: list                # 模型选择（校验：存在且来自本轮可见结果）
    text: str                     # 规则回填的原文（模型不可改）
    source: dict                  # 规则填：document_id/file_name/sentence_ids/corpus_version
    supports: list                # 模型判断：支持哪些约束 id
    role: str                     # 模型判断：direct | bridge
    entities: list = field(default_factory=list)   # 模型提议，规则只查格式
    time: str | None = None       # 模型提议
    provenance_valid: bool = True # 规则确定：span 确来自来源
    semantic_status: str = "selected"  # 模型判断（不用 confirmed）
    parents: list = field(default_factory=list)  # 证据级依赖：本条由哪些既有证据暴露的实体/线索引出（模型提议，规则校验须为已存在 evidence_id）


@dataclass
class EvidenceState:
    question_id: str
    turn_index: int
    E: list = field(default_factory=list)   # list[EvidenceCapsule]
    U: list = field(default_factory=list)   # list[{constraint_id, text, depends_on}]
    H: list = field(default_factory=list)   # list[{query, target_constraint, result}]
    B: dict = field(default_factory=dict)   # {search_rounds_left, context_tokens_left}
    # 元信息（存池/去重/淘汰用，不进 prompt / 不进 loss）
    evidence_chain_signature: str = ""
    corpus_index_version: str = ""
    trajectory_id: str = ""
    full_rollout_verifier_result: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "EvidenceState":
        caps = [EvidenceCapsule(**c) for c in d.get("E", [])]
        s = EvidenceState(
            question_id=d.get("question_id", ""),
            turn_index=d.get("turn_index", 0),
            E=caps,
            U=list(d.get("U", [])),
            H=list(d.get("H", [])),
            B=dict(d.get("B", {})),
            evidence_chain_signature=d.get("evidence_chain_signature", ""),
            corpus_index_version=d.get("corpus_index_version", ""),
            trajectory_id=d.get("trajectory_id", ""),
            full_rollout_verifier_result=d.get("full_rollout_verifier_result", 0.0),
        )
        return s


# ---------------- 解析模型的 STATE_UPDATE (Δ_t) ----------------
_STATE_UPDATE_RE = re.compile(r"<state_update>(.*?)</state_update>", re.DOTALL)


def parse_state_update(text: str) -> dict | None:
    """从模型输出中抽取 <state_update>{...}</state_update> 里的 JSON。解析失败返回 None。

    期望结构:
      {"keep":[{"span_ids":[...],"supports":[...],"role":"direct|bridge",
                "entities":[...],"time":"...","semantic_status":"selected"}...],
       "resolve":["c1",...], "open_constraints":[{"text":"...","depends_on":[...]}...],
       "decision":"continue|answer"}
    """
    m = _STATE_UPDATE_RE.search(text or "")
    if not m:
        # 兜底：有开标签无闭标签（引擎在别处截断）时，取开标签之后的内容尝试解析
        if text and "<state_update>" in text:
            body = text.split("<state_update>", 1)[1]
        else:
            return None
    else:
        body = m.group(1).strip()
    # 容错：允许 ```json 包裹
    if "```" in body:
        body = body.split("```")[1] if body.count("```") >= 2 else body
        body = body[4:].strip() if body.lower().startswith("json") else body.strip()
    try:
        obj = json.loads(body)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, IndexError):
        return None


class Canonicalizer:
    """规则规范化：把模型的 Δ_t（选 span）+ 本轮检索结果 O_t 落成新的可信状态 S_{t+1}。

    O_t（本轮可见检索结果）以 span_id -> {text, source} 提供；只有出现在 O_t（或 S_t 已有）中的
    span 才允许被 keep（防止模型引用不存在/不可见的 span）。
    """

    def __init__(self, corpus_index_version: str = ""):
        self.corpus_index_version = corpus_index_version

    @staticmethod
    def build_visible_spans(retrieval_docs: list) -> dict:
        """把一轮 /retrieve 的 docs（含 spans）整理成 span_id -> {text, source} 映射。"""
        visible = {}
        for doc in retrieval_docs or []:
            d = doc.get("document", doc) if isinstance(doc, dict) else {}
            file_name = d.get("file_name") or d.get("title") or ""
            doc_id = d.get("doc_id") or d.get("id") or ""
            for sp in d.get("spans", []) or []:
                sid = sp.get("span_id")
                if not sid:
                    continue
                visible[sid] = {
                    "text": sp.get("text", ""),
                    "source": {
                        "document_id": doc_id,
                        "file_name": file_name,
                        "sentence_ids": [sid],
                    },
                }
        return visible

    def canonicalize(self, S_t: EvidenceState, visible_spans: dict, delta: dict) -> EvidenceState:
        """核心：S_{t+1} = Canonicalize(S_t, O_t, Δ_t)。只做可验证操作，不生成新事实。

        对模型畸形输出保持健壮：delta 非 dict 则视为空更新；keep 项非 dict 则跳过。
        """
        if not isinstance(delta, dict):
            delta = {}
        # 可用 span 池 = 本轮可见 + S_t 中已保留证据的 span（允许跨轮引用旧证据）
        prior_spans = {}
        for cap in S_t.E:
            for i, sid in enumerate(cap.span_ids):
                prior_spans[sid] = {
                    "text": cap.text if len(cap.span_ids) == 1 else None,  # 多 span 拼接文本无法反切，标注 None
                    "source": cap.source,
                }
        available = dict(prior_spans)
        available.update(visible_spans)  # 本轮可见优先

        # 先保留 S_t 已有证据（延续），再并入本轮新 keep
        new_E: list[EvidenceCapsule] = list(S_t.E)
        seen_span_sets = {tuple(sorted(c.span_ids)) for c in new_E}
        eid_counter = len(new_E)

        for item in (delta.get("keep") or []):
            if not isinstance(item, dict):
                continue  # 跳过畸形 keep 项（模型可能吐 bare string 而非 {span_ids:...}）
            span_ids = [s for s in (item.get("span_ids") or []) if isinstance(s, str)]
            span_ids = span_ids[:MAX_SPANS_PER_EVIDENCE]
            if not span_ids:
                continue
            # 校验：所有 span 必须可见/可用
            if not all(sid in available for sid in span_ids):
                continue  # 拒绝不可见/幻觉 span
            key = tuple(sorted(span_ids))
            if key in seen_span_sets:
                continue  # 去重（完全相同的 span 集合）
            # 回填原文（规则）：优先用本轮可见 span 的文本拼接
            texts = []
            source_sids = []
            file_name = ""
            document_id = ""
            for sid in span_ids:
                info = available.get(sid) or {}
                if info.get("text"):
                    texts.append(info["text"])
                src = info.get("source") or {}
                source_sids.append(sid)
                file_name = file_name or src.get("file_name", "")
                document_id = document_id or src.get("document_id", "")
            backfilled = " ".join(texts).strip()[:MAX_EVIDENCE_CHARS]
            if not backfilled:
                continue  # 无法回填（如跨轮多 span 旧证据）则跳过，宁缺毋滥
            role = item.get("role") if item.get("role") in VALID_ROLES else "direct"
            sem = item.get("semantic_status")
            sem = sem if sem in VALID_SEMANTIC_STATUS else "selected"
            supports = [c for c in (item.get("supports") or []) if isinstance(c, str)]
            entities = [e for e in (item.get("entities") or []) if isinstance(e, str)][:8]
            # parents（模型提议、规则校验）：只保留指向"已存在证据"的依赖，杜绝悬挂/幻觉引用
            existing_eids = {c.evidence_id for c in new_E}
            parents = [p for p in (item.get("parents") or [])
                       if isinstance(p, str) and p in existing_eids]
            # time：接受模型给的字符串或结构化 dict（如 {"value":"June 1982","type":"signing_date"}）
            t = item.get("time")
            time_val = t if isinstance(t, (str, dict)) else None
            eid_counter += 1
            new_E.append(EvidenceCapsule(
                evidence_id=f"e{eid_counter}",
                span_ids=span_ids,
                text=backfilled,
                source={
                    "document_id": document_id,
                    "file_name": file_name,
                    "sentence_ids": source_sids,
                    "corpus_version": self.corpus_index_version,
                },
                supports=supports,
                role=role,
                entities=entities,
                time=time_val,
                provenance_valid=True,  # 已校验 span 存在于来源
                semantic_status=sem,
                parents=parents,
            ))
            seen_span_sets.add(key)

        # 限证据总数（保留最新的 MAX_EVIDENCE 条）
        if len(new_E) > MAX_EVIDENCE:
            new_E = new_E[-MAX_EVIDENCE:]

        # ---- 更新未解约束 U ----
        resolved = set(delta.get("resolve") or [])
        new_U = [u for u in S_t.U if u.get("constraint_id") not in resolved]
        # 追加模型新提出的 open_constraints（分配稳定 id）
        cid_counter = _max_constraint_id(S_t.U)
        for oc in (delta.get("open_constraints") or []):
            txt = oc.get("text") if isinstance(oc, dict) else None
            if not txt:
                continue
            cid_counter += 1
            dep = [d for d in (oc.get("depends_on") or []) if isinstance(d, str)]
            if not dep:
                # 缺省：新约束依赖本轮刚 resolve 的约束（桥接解出实体后才能实例化下一跳）
                dep = [c for c in (delta.get("resolve") or []) if isinstance(c, str)]
            new_U.append({
                "constraint_id": f"c{cid_counter}",
                "text": str(txt)[:200],
                "depends_on": dep,
            })
        new_U = new_U[:MAX_OPEN_CONSTRAINTS]

        # ---- 更新历史 H（本轮 query 结果），限长 ----
        new_H = list(S_t.H)
        q = delta.get("query") or (S_t.B.get("_last_query") if isinstance(S_t.B, dict) else None)
        if q:
            tgt = delta.get("target_constraint")
            if not tgt:
                # 缺省：本次检索针对的约束 = 本轮 resolve 的第一个（无则空）
                res = [c for c in (delta.get("resolve") or []) if isinstance(c, str)]
                tgt = res[0] if res else ""
            new_H.append({"query": str(q)[:120],
                          "target_constraint": tgt,
                          "result": "useful" if delta.get("keep") else "no_new_evidence"})
        new_H = new_H[-MAX_FAILED_DIRECTIONS:]

        # ---- 更新预算 B ----
        new_B = dict(S_t.B)
        new_B["search_rounds_left"] = max(0, int(new_B.get("search_rounds_left", 0)) - 1)

        S_next = EvidenceState(
            question_id=S_t.question_id,
            turn_index=S_t.turn_index + 1,
            E=new_E, U=new_U, H=new_H, B=new_B,
            corpus_index_version=self.corpus_index_version,
            trajectory_id=S_t.trajectory_id,
        )
        S_next.evidence_chain_signature = _chain_signature(S_next)
        return S_next


def _max_constraint_id(U: list) -> int:
    mx = 0
    for u in U or []:
        cid = u.get("constraint_id", "")
        m = re.match(r"c(\d+)", str(cid))
        if m:
            mx = max(mx, int(m.group(1)))
    return mx


def _chain_signature(S: EvidenceState) -> str:
    """证据链签名：体现证据间的依赖（parents），用于去重/淘汰。

    每条证据用其 span 集合表示；若有 parents，则前缀父证据的 span 集合并以 '->' 连接，
    使签名反映依赖链（如 d3426:s6->d3429:s1），而非仅"两 span 同时存在"。
    """
    id2spans = {c.evidence_id: "|".join(sorted(c.span_ids)) for c in S.E}
    toks = []
    for c in S.E:
        me = "|".join(sorted(c.span_ids))
        if c.parents:
            par = "+".join(id2spans.get(p, p) for p in c.parents)
            toks.append(f"{par}->{me}")
        else:
            toks.append(me)
    return "::".join(toks)


def init_state(question_id: str, search_rounds: int, question_constraints: list | None = None,
               corpus_index_version: str = "", trajectory_id: str = "") -> EvidenceState:
    """S0：初始状态（无证据，带初始约束=原问题，预算=最大搜索轮数）。"""
    U = question_constraints or [{"constraint_id": "c1", "text": "answer the question", "depends_on": []}]
    return EvidenceState(
        question_id=question_id, turn_index=0, E=[], U=U, H=[],
        B={"search_rounds_left": search_rounds},
        corpus_index_version=corpus_index_version, trajectory_id=trajectory_id,
    )


# ---------------- 固定模板渲染（模型看到的自然语言） ----------------
def render_state(S: EvidenceState, question_text: str) -> str:
    """把结构化状态按固定模板渲染成模型可读文本（非 JSON、非自由摘要）。"""
    lines = ["原始问题：", question_text.strip(), "", "已知证据："]
    if S.E:
        for cap in S.E:
            sup = ",".join(cap.supports) if cap.supports else "-"
            src = cap.source.get("file_name") or cap.source.get("document_id") or "?"
            sid = ",".join(cap.span_ids)
            lines.append(f"[{cap.evidence_id} | 支持 {sup} | 来源 {src}#{sid}]")
            lines.append(f"“{cap.text}”")
    else:
        lines.append("（暂无）")
    lines.append("")
    lines.append("尚未解决：")
    if S.U:
        for u in S.U:
            lines.append(f"[{u['constraint_id']}] {u['text']}")
    else:
        lines.append("（无）")
    if S.H:
        lines.append("")
        lines.append("已经尝试：")
        for h in S.H:
            lines.append(f"- “{h['query']}”：{h.get('result','')}")
    lines.append("")
    lines.append(f"剩余预算：搜索轮数 {S.B.get('search_rounds_left', 0)}")
    return "\n".join(lines)


# ---------------- replay 起点序列化（防泄漏白名单，设计文档 §5.3） ----------------
# search-r1 指令尾（与 musique_to_slime.PROMPT_TEMPLATE 的动作协议一致）
_REPLAY_INSTRUCTION = (
    "\n\nYou may continue: reason inside <think></think>, call search with "
    "<search> query </search>, and when ready answer with <answer> ... </answer>.\n"
)


def serialize_for_replay(S: EvidenceState, question_text: str,
                         forbidden_answers: list | None = None, strict: bool = True) -> str:
    """把中间状态 S_k 序列化成 replay 起点 prompt。

    白名单：只放 已知什么/还缺什么（问题 + E 带来源 + U + 必要 H + B），末尾接续跑指令。
    绝不放：S_k 之前的完整 reasoning、原始 observation 全文、S_k 之后动作、最终答案、gold facts。
    渲染直接复用 render_state（其内容已是白名单字段）。

    防泄漏在**状态内容**上校验（render_state 输出），不校验后接的固定指令（指令合法包含 <answer> 说明）。
    """
    state_text = render_state(S, question_text)
    assert_no_leakage(state_text, forbidden_answers or [], strict=strict)  # 只守护状态内容
    return state_text + _REPLAY_INSTRUCTION


def assert_no_leakage(state_text: str, forbidden_answers: list, forbidden_markers: list | None = None,
                      strict: bool = True):
    """防泄漏断言：**状态内容**（render_state 输出，不含后接指令）不得含动作标记或 gold 标记。

    合法的证据状态只由证据文本/约束/历史/预算组成，绝不该出现 <search>/<answer>/gold 标记；
    一旦出现即说明未来动作/最终答案被写进了起点。注意：不要把 serialize_for_replay 后接的
    固定指令（合法引用 <answer> 标签）传进来。

    对 gold answer **文本**（如证据里恰好出现答案词，属正常）只返回告警，不硬失败。
    """
    markers = forbidden_markers or ["<answer>", "</answer>", "<search>", "</search>",
                                    "<state_update>", "gold_", "supporting_fact"]
    for mk in markers:
        if mk in state_text:
            raise AssertionError(f"state leakage: contains forbidden marker {mk!r}")
    warnings = []
    for ans in forbidden_answers or []:
        if ans and ans.strip() and ans.strip().lower() in state_text.lower():
            warnings.append(ans)
    return warnings  # 答案词出现在证据里属正常，返回供调用方记录
