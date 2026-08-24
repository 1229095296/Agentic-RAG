"""成功状态池（S3.3）——按 question_id 分文件的 append-only JSONL 池，配进程内缓存。

对应 STATEFLOW_DESIGN.md 决策 B。存放"从 S0 出发的完整成功轨迹"的 turn-boundary 证据状态，
供训练时从中间状态 S_k replay 起点。可审计（JSONL 可人工看）、跨重启不丢、DSW/DLC 共享。

一条池记录（每行）:
  {"question_id","turn_index","state": <EvidenceState.to_dict>, "trajectory_id",
   "evidence_chain_signature","corpus_index_version","verifier_result","state_age"}

职责：
- add_states：写入成功轨迹的各 turn 状态（带去重：同 question 同 chain_signature 只留一条）。
- sample_anchor：随机/按策略取一个 (question_id, turn_index, state) 作为 replay 起点。
- 维护每 anchor 的 suffix success rate（ρ_k，EMA）供第四步自适应课程用。
- index-version 校验、旧状态淘汰接口（V1 简化）。

纯逻辑 + 文件 IO，无 GPU/torch 依赖，可 CPU 单测。
"""
from __future__ import annotations

import json
import os
import random
import threading


class StatePool:
    def __init__(self, pool_dir: str, ema_alpha: float = 0.3,
                 rho_low: float = 0.25, rho_high: float = 0.75,
                 adaptive_k: bool = False, min_obs_for_move: int = 2):
        self.pool_dir = pool_dir
        os.makedirs(pool_dir, exist_ok=True)
        self.ema_alpha = ema_alpha
        # 第四步自适应课程参数（STATEFLOW_DESIGN §8）
        self.rho_low = rho_low
        self.rho_high = rho_high
        self.adaptive_k = adaptive_k
        self.min_obs_for_move = min_obs_for_move
        self._lock = threading.Lock()
        # 进程内缓存: question_id -> list[record]
        self._cache: dict[str, list] = {}
        # anchor 统计: (question_id, turn_index) -> {"rho": float, "n": int}
        self._rho: dict[tuple, dict] = {}
        # 每问题当前锚点难度位置 k（turn_index）；随 ρ_k 前移/后移
        self._kcur: dict[str, int] = {}
        # 课程状态（ρ_k + kcur）持久化到磁盘，跨 step/重启不丢、可审计
        self._curriculum_path = os.path.join(pool_dir, "_curriculum.json")
        self._loaded = False

    # ---------- 文件路径 ----------
    def _path(self, question_id: str) -> str:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(question_id))[:120]
        return os.path.join(self.pool_dir, f"{safe}.jsonl")

    # ---------- 加载 ----------
    def load(self):
        """把 pool_dir 下所有 jsonl 读进缓存（进程启动时调用一次）。"""
        with self._lock:
            self._cache.clear()
            if not os.path.isdir(self.pool_dir):
                self._loaded = True
                return
            for fn in os.listdir(self.pool_dir):
                if not fn.endswith(".jsonl"):
                    continue
                path = os.path.join(self.pool_dir, fn)
                recs = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                recs.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                if recs:
                    self._cache[recs[0]["question_id"]] = recs
            self._load_curriculum()
            self._loaded = True

    def _load_curriculum(self):
        """从 _curriculum.json 载入 ρ_k 与每问题当前锚点 k（须在持锁上下文内调用）。"""
        if not os.path.exists(self._curriculum_path):
            return
        try:
            with open(self._curriculum_path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        self._rho = {}
        for e in d.get("rho", []):
            try:
                self._rho[(str(e["qid"]), int(e["k"]))] = {
                    "rho": float(e["rho"]), "n": int(e.get("n", 1))}
            except (KeyError, ValueError, TypeError):
                continue
        self._kcur = {str(q): int(k) for q, k in (d.get("kcur") or {}).items()}

    def _save_curriculum(self):
        """把 ρ_k 与 kcur 原子落盘（须在持锁上下文内调用）。"""
        d = {
            "rho": [{"qid": q, "k": k, "rho": v["rho"], "n": v["n"]}
                    for (q, k), v in self._rho.items()],
            "kcur": self._kcur,
        }
        tmp = self._curriculum_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
            os.replace(tmp, self._curriculum_path)
        except OSError:
            pass

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    # ---------- 写入 ----------
    def add_states(self, question_id: str, states: list, trajectory_id: str,
                   verifier_result: float, corpus_index_version: str = "",
                   index_version_current: str | None = None,
                   label: dict | None = None, question_text: str = "") -> int:
        """把一条成功轨迹的 turn-boundary 状态写入池。返回新增条数。

        - 去重：同 question 下 evidence_chain_signature 已存在则跳过。
        - index-version 校验：若给了 index_version_current 且与状态记录不符，拒绝写入。
        - label/question_text 随记录存下（replay 时算 reward 需要 ground_truth；不进 replay prompt）。
        - 只应传入"从 S0 出发的完整成功轨迹"的状态（由 harvest 侧保证；见 harvest_states.py）。
        """
        self._ensure_loaded()
        with self._lock:
            existing = self._cache.setdefault(question_id, [])
            existing_sigs = {r.get("evidence_chain_signature") for r in existing}
            path = self._path(question_id)
            added = 0
            with open(path, "a", encoding="utf-8") as f:
                for st in states:
                    sig = st.get("evidence_chain_signature", "")
                    turn_index = st.get("turn_index", 0)
                    # turn_index=0 的空初始状态无 replay 价值，跳过
                    if turn_index == 0 and not st.get("E"):
                        continue
                    if index_version_current is not None and \
                       st.get("corpus_index_version", corpus_index_version) != index_version_current:
                        continue
                    if sig and sig in existing_sigs:
                        continue
                    # 写入成功池前回填轨迹终局 verifier 结果（避免状态里残留中途默认 0.0）
                    st = dict(st)
                    st["full_rollout_verifier_result"] = verifier_result
                    rec = {
                        "question_id": question_id,
                        "turn_index": turn_index,
                        "state": st,
                        "trajectory_id": trajectory_id,
                        "evidence_chain_signature": sig,
                        "corpus_index_version": st.get("corpus_index_version", corpus_index_version),
                        "verifier_result": verifier_result,
                        "label": label,               # replay 算 reward 用（含 ground_truth），不进 prompt
                        "question_text": question_text,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    existing.append(rec)
                    existing_sigs.add(sig)
                    added += 1
            return added

    # ---------- 采样 ----------
    def is_empty(self) -> bool:
        self._ensure_loaded()
        return not any(self._cache.values())

    def num_states(self) -> int:
        self._ensure_loaded()
        return sum(len(v) for v in self._cache.values())

    def sample_anchor(self, rng: random.Random | None = None,
                      prefer_turn_index: int | None = None) -> dict | None:
        """取一个 anchor 记录 {question_id, turn_index, state, ...}。

        - adaptive_k 关：均匀随机取一个 question，再随机取状态（可指定 prefer_turn_index）——V1 行为。
        - adaptive_k 开：随机取 question 后，按 ρ_k 自适应挑该问题的锚点难度位置 k（_select_k）。
        """
        self._ensure_loaded()
        rng = rng or random
        qids = [q for q, v in self._cache.items() if v]
        if not qids:
            return None
        qid = rng.choice(qids)
        recs = self._cache[qid]
        if self.adaptive_k and prefer_turn_index is None:
            with self._lock:
                k = self._select_k(qid)
                self._save_curriculum()
            cands = [r for r in recs if r["turn_index"] == k] or recs
        elif prefer_turn_index is not None:
            cands = [r for r in recs if r["turn_index"] == prefer_turn_index] or recs
        else:
            allowed = set(self._anchor_turns(qid))
            cands = [r for r in recs if r["turn_index"] in allowed] or recs
        return rng.choice(cands)

    def _available_turns(self, qid: str) -> list:
        """该问题池里可用的 turn_index（升序去重）。小=早期/难，大=接近终局/易。"""
        return sorted({r["turn_index"] for r in self._cache.get(qid, [])})

    def _anchor_turns(self, qid: str) -> list:
        """优先用作 replay 锚点的 turn_index：仍有未解约束（U 非空）的中间态。

        answer-ready 终局态（U 空，只剩作答）replay 价值低，降为兜底：仅当没有任何
        informative 中间态时才回退到全部 turn。"""
        recs = self._cache.get(qid, [])
        informative = sorted({r["turn_index"] for r in recs
                              if (r.get("state") or {}).get("U")})
        return informative or self._available_turns(qid)

    def _select_k(self, qid: str) -> int | None:
        """按 ρ_k 自适应选锚点位置 k（须持锁调用）。

        规则（STATEFLOW_DESIGN §8）：ρ_k>ρ_high → k 前移（更早、更难，逼多做早期搜索）；
        ρ_k<ρ_low → k 后移（更晚、更易，多给已知证据）；否则保持。初始落在中等难度。
        观测数 < min_obs_for_move 时不移动，避免抖动。
        """
        turns = self._anchor_turns(qid)
        if not turns:
            return None
        # 初始：中等难度（可用状态的中位）
        k = self._kcur.get(qid)
        if k is None or k not in turns:
            k = turns[len(turns) // 2]
        if self.adaptive_k:
            stat = self._rho.get((qid, k))
            if stat and stat["n"] >= self.min_obs_for_move:
                idx = turns.index(k)
                if stat["rho"] > self.rho_high and idx > 0:
                    k = turns[idx - 1]            # 太简单 → 前移（更早、更难）
                elif stat["rho"] < self.rho_low and idx < len(turns) - 1:
                    k = turns[idx + 1]            # 太难 → 后移（更晚、更易）
        self._kcur[qid] = k
        return k

    # ---------- 第四步：suffix success rate（EMA） ----------
    def update_suffix_success(self, question_id: str, turn_index: int, success_rate: float):
        with self._lock:
            key = (question_id, turn_index)
            cur = self._rho.get(key)
            if cur is None:
                self._rho[key] = {"rho": success_rate, "n": 1}
            else:
                a = self.ema_alpha
                cur["rho"] = (1 - a) * cur["rho"] + a * success_rate
                cur["n"] += 1
            self._save_curriculum()

    def get_suffix_success(self, question_id: str, turn_index: int) -> float | None:
        v = self._rho.get((question_id, turn_index))
        return v["rho"] if v else None

    def rho_snapshot(self) -> list:
        """当前 ρ_k 统计快照（供日志/指标）：[{qid,k,rho,n,kcur}...]，按 qid,k 排序。"""
        out = []
        for (q, k), v in sorted(self._rho.items()):
            out.append({"qid": q, "k": k, "rho": round(v["rho"], 3),
                        "n": v["n"], "kcur": self._kcur.get(q)})
        return out


# 进程内单例（供 data source 与 harvest 钩子共享；同一 RolloutManager 进程内有效）
_GLOBAL_POOL: StatePool | None = None


def get_pool(pool_dir: str | None = None) -> StatePool:
    global _GLOBAL_POOL
    if _GLOBAL_POOL is None:
        d = pool_dir or os.environ.get("STATEFLOW_POOL_DIR", "state_pool")
        _GLOBAL_POOL = StatePool(
            d,
            ema_alpha=float(os.environ.get("STATEFLOW_EMA_ALPHA", "0.3")),
            rho_low=float(os.environ.get("STATEFLOW_RHO_LOW", "0.25")),
            rho_high=float(os.environ.get("STATEFLOW_RHO_HIGH", "0.75")),
            adaptive_k=os.environ.get("STATEFLOW_ADAPTIVE_K", "0") == "1",
        )
        _GLOBAL_POOL.load()
    return _GLOBAL_POOL
