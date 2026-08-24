"""收割成功轨迹状态入池（S3.3）——接 slime 的 --rollout-all-samples-process-path。

一轮 rollout 后被 slime 调用（签名 (args, samples) -> None）。职责：
- 筛出**从 S0 出发的完整成功轨迹**：reward≥阈值、status==COMPLETED、且 metadata.is_replay 非 True。
- 取其 metadata["turn_states"]，逐个过 leakage/provenance 校验（防止未来动作/gold 泄漏进状态）。
- 通过校验的状态写入 state_pool（按 question_id 分文件 JSONL）。

关键（STATEFLOW_DESIGN §5.2 / 周报 §3.1）：**replay continuation 产生的状态不回写主池**，
避免"当前策略无法从真实初始状态到达的状态"自增殖——靠 metadata.is_replay 区分。
"""
from __future__ import annotations

import os
import re

from slime_integration.state_pool import get_pool
from slime_integration.evidence_state import EvidenceState, render_state, assert_no_leakage
from slime_integration.prompt_utils import extract_question_id, extract_question_text

REWARD_THRESHOLD = float(os.environ.get("STATEFLOW_HARVEST_REWARD_THRESHOLD", "0.8"))


def _reward_scalar(sample) -> float:
    r = getattr(sample, "reward", None)
    if isinstance(r, dict):
        return float(r.get("score", r.get("reward", 0.0)) or 0.0)
    try:
        return float(r)
    except (TypeError, ValueError):
        return 0.0


def _em_scalar(sample) -> float:
    """答案是否正确（1.0/0.0）——ρ_k 的 Verifier 信号。优先取 reward['em']，否则 score>=1.0。"""
    r = getattr(sample, "reward", None)
    if isinstance(r, dict):
        if "em" in r:
            return 1.0 if r["em"] else 0.0
        return 1.0 if float(r.get("score", 0.0) or 0.0) >= 1.0 else 0.0
    try:
        return 1.0 if float(r) >= 1.0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _as_groups(samples) -> list:
    """还原 GRPO 组：samples 为嵌套 list 时直接用；否则按 sample.group_index 归组。"""
    if samples and all(isinstance(s, list) for s in samples):
        return [g for g in samples if g]
    groups: dict = {}
    for s in samples:
        gi = getattr(s, "group_index", id(s))
        groups.setdefault(gi, []).append(s)
    return list(groups.values())


_SPAN_ID_RE = re.compile(r"^[^:\s]+:s\d+$")


def _cap_has_rule_provenance(cap) -> bool:
    """接受显式 source 元数据，或 wiki-18 这类 doc:sN 形式的规则 span_id。"""
    source = cap.source if isinstance(cap.source, dict) else {}
    if source.get("file_name") or source.get("document_id"):
        return True
    span_ids = [sid for sid in (cap.span_ids or []) if isinstance(sid, str)]
    return bool(span_ids) and all(_SPAN_ID_RE.match(sid) for sid in span_ids)


def _passes_leakage_and_provenance(state_dict: dict, question_text: str, forbidden_answers: list) -> bool:
    """校验单个状态：provenance 有效 + 序列化后无动作/gold 标记泄漏。"""
    try:
        S = EvidenceState.from_dict(state_dict)
    except Exception:  # noqa: BLE001
        return False
    # provenance：优先要求显式 source；wiki-18/e5 返回源字段为空时，用规则生成的 span_id 兜底。
    for cap in S.E:
        if not cap.provenance_valid:
            return False
        if not _cap_has_rule_provenance(cap):
            return False
    # leakage：状态渲染文本不得含动作标记（answer/search/state_update/gold）
    try:
        assert_no_leakage(render_state(S, extract_question_text(question_text)), forbidden_answers or [])
    except AssertionError:
        return False
    return True


def process(args, samples, data_source=None, **kwargs):
    """slime --rollout-all-samples-process-path 入口。

    slime 调用签名为 process_func(args, all_samples, data_source)（3 个位置参数）。
    就地处理，不返回。data_source 暂未使用（状态池经 get_pool 单例共享）。
    """
    pool = get_pool()
    index_version = os.environ.get("MUSIQUE_CORPUS_VERSION", "musique_v1")

    # samples 可能是嵌套 list（分组）或平铺；统一拍平
    flat = []
    for s in samples:
        if isinstance(s, list):
            flat.extend(s)
        else:
            flat.append(s)

    n_harvested = 0
    n_traj = 0
    for sample in flat:
        meta = getattr(sample, "metadata", None)
        if not isinstance(meta, dict):
            continue
        if meta.get("is_replay"):
            continue  # replay continuation 不回写主池
        status = getattr(sample, "status", None)
        status_ok = (status is None) or (getattr(status, "name", str(status)) == "COMPLETED")
        if not status_ok:
            continue
        if _reward_scalar(sample) < REWARD_THRESHOLD:
            continue
        turn_states = meta.get("turn_states") or []
        if not turn_states:
            continue

        # 目标答案（用于泄漏告警，不作硬判）
        label = getattr(sample, "label", None)
        forbidden = []
        if isinstance(label, dict):
            gt = label.get("ground_truth", {})
            if isinstance(gt, dict):
                tgt = gt.get("target")
                forbidden = tgt if isinstance(tgt, list) else ([tgt] if tgt else [])
        question_text = extract_question_text(meta.get("question_text") or meta.get("question") or getattr(sample, "prompt", ""))
        qid = extract_question_id(meta, label if isinstance(label, dict) else None)
        trajectory_id = meta.get("trajectory_id", str(getattr(sample, "index", "")) )

        good_states = [st for st in turn_states
                       if _passes_leakage_and_provenance(st, question_text, forbidden)]
        if not good_states:
            continue
        added = pool.add_states(
            question_id=str(qid), states=good_states, trajectory_id=str(trajectory_id),
            verifier_result=_reward_scalar(sample), corpus_index_version=index_version,
            index_version_current=index_version,
            label=label if isinstance(label, dict) else None,
            question_text=question_text,
        )
        n_harvested += added
        n_traj += 1

    if n_harvested:
        print(f"[harvest_states] harvested {n_harvested} states from {n_traj} successful trajectories; "
              f"pool now has {pool.num_states()} states", flush=True)

    # ---- 第四步：ρ_k 反馈 + 组级指标（all-zero/all-one/mixed）----
    groups = _as_groups(samples)
    n_all0 = n_all1 = n_mixed = 0
    n_replay_groups = 0
    for g in groups:
        ems = [_em_scalar(s) for s in g]
        if not ems:
            continue
        m = sum(ems) / len(ems)
        if m == 0.0:
            n_all0 += 1
        elif m == 1.0:
            n_all1 += 1
        else:
            n_mixed += 1
        # replay 组：把该 anchor 的组内成功率喂给 ρ_k（EMA）
        meta0 = getattr(g[0], "metadata", None)
        if isinstance(meta0, dict) and meta0.get("is_replay"):
            qid = str(meta0.get("qid", ""))
            try:
                k = int(meta0.get("anchor_turn_index", 0))
            except (TypeError, ValueError):
                k = 0
            if qid:
                pool.update_suffix_success(qid, k, m)
                n_replay_groups += 1

    if groups:
        print(f"[harvest_states] groups={len(groups)} (replay={n_replay_groups}) "
              f"| all-zero={n_all0} all-one={n_all1} mixed={n_mixed} "
              f"| rho_k={pool.rho_snapshot()}", flush=True)

    # ---- 第四步追加：GRPO 组内统计（reward 方差、EM 方差、排序一致率）----
    groups = _as_groups(samples)
    if groups and len(groups) > 0:
        import math as _math
        r_stds, e_stds = [], []
        n_rank_violation = 0   # 错误轨迹最高 reward 超过正确轨迹最低 reward 的组数
        n_valid_groups = 0
        for g in groups:
            rewards = [_reward_scalar(s) for s in g]
            ems = [_em_scalar(s) for s in g]
            if len(rewards) < 2:
                continue
            n_valid_groups += 1
            # reward/EM 标准差
            r_mean = sum(rewards) / len(rewards)
            r_stds.append(_math.sqrt(sum((r - r_mean)**2 for r in rewards) / len(rewards)))
            e_mean = sum(ems) / len(ems)
            e_stds.append(_math.sqrt(sum((e - e_mean)**2 for e in ems) / len(ems)))
            # 排序检查：错误轨迹最高 reward 是否超过正确轨迹最低 reward
            correct_rewards = [r for r, e in zip(rewards, ems) if e >= 1.0]
            wrong_rewards = [r for r, e in zip(rewards, ems) if e < 1.0]
            if correct_rewards and wrong_rewards:
                if max(wrong_rewards) > min(correct_rewards):
                    n_rank_violation += 1
        if n_valid_groups > 0:
            avg_r_std = sum(r_stds) / len(r_stds) if r_stds else 0.0
            avg_e_std = sum(e_stds) / len(e_stds) if e_stds else 0.0
            violation_ratio = n_rank_violation / n_valid_groups if n_valid_groups else 0.0
            print(f"[harvest_stats] groups={n_valid_groups} "
                  f"| avg_reward_std={avg_r_std:.4f} avg_em_std={avg_e_std:.4f} "
                  f"| rank_violation={n_rank_violation}/{n_valid_groups} ({violation_ratio:.3f}) "
                  f"| {'⚠️ 奖励排序异常!' if violation_ratio > 0.05 else '✓ 奖励排序正常'}", flush=True)
