"""StateFlow 数据源（S3.4）——自定义 --data-source-path，混采 full-start 与 replay-start。

对应 STATEFLOW_DESIGN.md 决策 C + §6。继承 slime 的 RolloutDataSourceWithBuffer：
- full-start：从 MuSiQue 数据集取问题，fan-out n_samples_per_prompt 份 → 一个 GRPO 组（=第二步行为）。
- replay-start：从成功状态池取 anchor (q, S_k)，把 S_k 序列化成 prompt 注入，fan-out → **独立** GRPO 组。
- full 组与 replay 组是不同的内层 list → 天然独立归一化（满足周报 §3.4）。

replay Sample 的 metadata 带 is_replay=True + anchor_state（供 generate_stateflow 从该状态续跑）+ qid + question_text；
label 取自池记录（含 ground_truth，reward 用）。replay prompt 由 serialize_for_replay 白名单守护，不泄漏答案。

混比 p_full：默认取固定值 STATEFLOW_P_FULL（V1/smoke）；开启 STATEFLOW_ADAPTIVE_PFULL=1 则按 rollout step
调度：前若干步全 full-start，随后短窗口内逐步减少 replay。anchor turn_index：默认随机；开启 STATEFLOW_ADAPTIVE_K=1 则
由 state_pool 按 ρ_k 自适应挑难度（前移/后移）。ρ_k 由 harvest_states 在每轮 rollout 后回写。

接入：--data-source-path slime_integration.stateflow_data_source.StateFlowDataSource
"""
from __future__ import annotations

import copy
import os
import random

from slime.rollout.data_source import RolloutDataSource, RolloutDataSourceWithBuffer
from slime.utils.types import Sample

from slime_integration.state_pool import get_pool
from slime_integration.evidence_state import EvidenceState, serialize_for_replay
from slime_integration.prompt_utils import extract_question_id, extract_question_text


class StateFlowDataSource(RolloutDataSourceWithBuffer):
    def __init__(self, args):
        super().__init__(args)
        # 固定混比（V1 / smoke）；开启 STATEFLOW_ADAPTIVE_PFULL 则用分阶段调度覆盖
        self.p_full_fixed = float(os.environ.get("STATEFLOW_P_FULL", "0.5"))
        self.adaptive_pfull = os.environ.get("STATEFLOW_ADAPTIVE_PFULL", "0") == "1"
        self.pool = get_pool(os.environ.get("STATEFLOW_POOL_DIR", "state_pool"))
        self._rng = random.Random(getattr(args, "seed", 42) or 42)
        # 分阶段调度用：当前 rollout step 与总步数
        self._step = 0
        self._total_steps = int(getattr(args, "num_rollout", 0) or 0)
        self.replay_start_step = int(os.environ.get("STATEFLOW_REPLAY_START_STEP", "50"))
        self.replay_end_step = int(os.environ.get("STATEFLOW_REPLAY_END_STEP", "500"))
        self.replay_start_p_full = float(os.environ.get("STATEFLOW_REPLAY_START_P_FULL", "0.5"))
        # Smoke/debug only: guarantee that low replay ratios are observable in
        # short runs. Default 0 preserves the stochastic production schedule.
        self.min_replay_groups = int(os.environ.get("STATEFLOW_MIN_REPLAY_GROUPS", "0"))
        prefer_turn = os.environ.get("STATEFLOW_REPLAY_PREFER_TURN_INDEX", "").strip()
        self.prefer_turn_index = int(prefer_turn) if prefer_turn else None
        # 统计
        self._n_full = 0
        self._n_replay = 0

    def _p_full_for_step(self) -> float:
        """Step-based full-start比例；replay% = 1 - p_full.

        Default schedule:
        - step < 50: p_full=1.0, replay=0%, harvest valid states from full trajectories.
        - 50 <= step < 500: p_full linearly rises from 0.5 to 1.0.
        - step >= 500: p_full=1.0, replay=0%, full-task hardening.
        """
        if not self.adaptive_pfull or self._total_steps <= 0:
            return self.p_full_fixed
        if self._step < self.replay_start_step:
            return 1.0
        if self._step >= self.replay_end_step:
            return 1.0
        span = max(1, self.replay_end_step - self.replay_start_step)
        progress = (self._step - self.replay_start_step) / span
        p_full = self.replay_start_p_full + (1.0 - self.replay_start_p_full) * progress
        return max(0.0, min(1.0, p_full))

    # ---------- 组装工具 ----------
    def _fanout(self, base_sample: Sample) -> list:
        """把一个起点样本复制成 n_samples_per_prompt 份，赋 group/index（与父类一致）。"""
        group = []
        for _ in range(self.args.n_samples_per_prompt):
            s = copy.deepcopy(base_sample)
            s.group_index = self.sample_group_index
            s.index = self.sample_index
            self.sample_index += 1
            group.append(s)
        self.sample_group_index += 1
        return group

    def _full_group(self) -> list:
        """从基础数据集取一个 MuSiQue 问题，标注 metadata 后 fan-out。"""
        # 直接走数据集路径（buffer 已在 get_samples 顶部处理，避免二次 drain）
        base = RolloutDataSource.get_samples(self, num_samples=1)  # [[G 份...]]
        group = base[0]
        # base 已 fan-out；补 metadata（full-start）
        label = getattr(group[0], "label", None)
        meta0 = group[0].metadata if isinstance(group[0].metadata, dict) else {}
        qid = extract_question_id(meta0, label if isinstance(label, dict) else None, default="")
        question_text = extract_question_text(
            meta0.get("question_text") or meta0.get("question") or group[0].prompt
        )
        for s in group:
            meta = dict(s.metadata) if isinstance(s.metadata, dict) else {}
            meta.setdefault("is_replay", False)
            meta["qid"] = meta.get("qid") or qid
            meta["question_text"] = extract_question_text(
                meta.get("question_text") or meta.get("question") or question_text or s.prompt
            )
            meta.setdefault("trajectory_id", f"full_{s.index}")
            s.metadata = meta
        self._n_full += 1
        return group

    def _replay_group(self) -> list | None:
        """从状态池取 anchor，序列化 S_k 为 prompt，fan-out 成独立 replay 组。"""
        rec = self.pool.sample_anchor(rng=self._rng, prefer_turn_index=self.prefer_turn_index)
        if rec is None:
            return None
        state_dict = rec["state"]
        question_text = extract_question_text(rec.get("question_text", ""))
        label = rec.get("label")
        try:
            S = EvidenceState.from_dict(state_dict)
            replay_prompt = serialize_for_replay(S, question_text)  # 白名单守护，raise 则该 anchor 作废
        except AssertionError:
            return None
        base = Sample(prompt=replay_prompt, label=label)
        base.metadata = {
            "is_replay": True,
            "anchor_state": state_dict,
            "anchor_turn_index": rec.get("turn_index", 0),
            "qid": rec.get("question_id", ""),
            "question_text": question_text,
            "trajectory_id": f"replay_{rec.get('question_id','')}_{rec.get('turn_index',0)}",
        }
        self._n_replay += 1
        return self._fanout(base)

    # ---------- 覆盖 get_samples ----------
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """返回 num_samples 个 prompt 组，每组 full 或 replay（各自独立 GRPO 组）。"""
        # 先用 buffer（如 partial rollout 回收），与父类一致
        groups = self._get_samples_from_buffer(num_samples)
        remaining = num_samples - len(groups)

        p_full = self._p_full_for_step()
        pool_ready = not self.pool.is_empty()
        forced_replay_positions = set()
        if pool_ready and p_full < 1.0 and self.min_replay_groups > 0 and remaining > 0:
            k = min(remaining, self.min_replay_groups)
            forced_replay_positions = set(self._rng.sample(range(remaining), k))

        for i in range(remaining):
            use_replay = pool_ready and (
                i in forced_replay_positions or self._rng.random() >= p_full
            )
            g = self._replay_group() if use_replay else None
            if g is None:
                g = self._full_group()
            groups.append(g)

        print(f"[stateflow_ds] step={self._step} p_full={p_full:.2f} "
              f"(adaptive={self.adaptive_pfull}, min_replay={self.min_replay_groups}) "
              f"cum_full={self._n_full} cum_replay={self._n_replay}",
              flush=True)
        self._step += 1
        return groups
