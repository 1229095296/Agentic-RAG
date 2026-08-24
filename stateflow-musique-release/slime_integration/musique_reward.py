"""Rule-based rewards for MuSiQue/Search-R1 and StateFlow-GRPO.

The non-evidence pieces are shared by the Search-R1 baseline and StateFlow:
final-answer quality, executable search format, and simple search efficiency.
StateFlow adds rule-scored evidence/path/retrieval components computed from the
canonical evidence state and rollout trace. No LLM-as-a-judge is used.
"""
from __future__ import annotations

import math
import os
import re
import string
from collections import Counter
from typing import Any


# ---------------- Text normalization and answer matching ----------------
def normalize_answer(s: str) -> str:
    """SQuAD-style normalization."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc((s or "").lower())))


def em_check(prediction: str, golds: list[str]) -> int:
    npred = normalize_answer(prediction)
    return int(any(npred == normalize_answer(g) for g in golds))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_f1(prediction: str, golds: list[str]) -> float:
    return max((f1_score(prediction, g) for g in golds), default=0.0)


# ---------------- Structured trace parsing ----------------
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_INFO_RE = re.compile(r"<information>(.*?)</information>", re.DOTALL)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_STATE_UPDATE_RE = re.compile(r"<state_update>(.*?)</state_update>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_DANGLING_THINK_RE = re.compile(r"<think>.*$", re.DOTALL)
_CHAT_CONTROL_RE = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>(?:assistant|user|system)?")


def baseline_reward_view(text: str) -> str:
    """Return the executable Search-R1 transcript for baseline reward parsing.

    Qwen3.5 may quote prompt examples such as "<answer> Beijing </answer>" inside
    long reasoning. The baseline reward should grade only actions that survived
    the environment loop, not meta-format prose inside <think>.
    """
    text = _CHAT_CONTROL_RE.sub("", text or "")
    text = _THINK_RE.sub("\n", text)
    text = _DANGLING_THINK_RE.sub("", text)
    return text


def extract_solution(text: str):
    """Return the last non-prompt <answer> block; None if absent."""
    matches = _ANSWER_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1].strip()


def extract_search_queries(text: str) -> list[str]:
    return [q.strip() for q in _SEARCH_RE.findall(text or "") if q.strip()]


def extract_information_blocks(text: str) -> list[str]:
    return [b.strip() for b in _INFO_RE.findall(text or "") if b.strip()]


def has_valid_final_answer(text: str) -> int:
    answer = extract_solution(text)
    return int(answer is not None and bool(answer.strip()))


def is_valid_format(text: str, require_state_update: bool = False) -> bool:
    """Executable protocol check.

    This intentionally allows environment-injected cue/state text between tags,
    because StateFlow appends such text during rollout. It still blocks the old
    no-search bare-answer loophole by requiring search followed by information.
    """
    text = text or ""
    if not has_valid_final_answer(text):
        return False
    tags = ("think", "search", "information", "answer", "state_update")
    for tag in tags:
        if text.count(f"<{tag}>") != text.count(f"</{tag}>"):
            return False

    searches = list(_SEARCH_RE.finditer(text))
    infos = list(_INFO_RE.finditer(text))
    if not searches or not infos:
        return False
    if not any(info.start() >= searches[0].end() for info in infos):
        return False

    ans = list(_ANSWER_RE.finditer(text))
    if not ans or ans[-1].start() < infos[0].end():
        return False

    if require_state_update:
        state_updates = list(_STATE_UPDATE_RE.finditer(text))
        if not state_updates:
            return False
        ordered_events = []
        for tag, regex in (
            ("search", _SEARCH_RE),
            ("information", _INFO_RE),
            ("state_update", _STATE_UPDATE_RE),
            ("answer", _ANSWER_RE),
        ):
            ordered_events.extend((m.start(), m.end(), tag) for m in regex.finditer(text))
        ordered_events.sort(key=lambda x: x[0])
        for info in infos:
            next_action = next(
                (ev for ev in ordered_events if ev[0] >= info.end() and ev[2] in {"search", "state_update", "answer"}),
                None,
            )
            if next_action is None or next_action[2] != "state_update":
                return False
        if ans[-1].start() < state_updates[-1].end():
            return False
    return True


def soft_format(text: str, require_state_update: bool = False) -> float:
    """Buffered format reward in [0, 1]. Bare answers get only a tiny signal."""
    text = text or ""
    if is_valid_format(text, require_state_update=require_state_update):
        return 1.0
    if "<information>" in text and "<search>" in text:
        return 0.5
    if has_valid_final_answer(text):
        return 0.2
    return 0.0


def gate_valid_sequence(text: str, require_state_update: bool = False) -> float:
    return 1.0 if is_valid_format(text, require_state_update=require_state_update) else 0.0


def is_valid_trace_sequence(turns: list[dict] | None, require_state_update: bool = False) -> bool:
    """Protocol check from environment-parsed actions.

    StateFlow rollouts contain environment-injected text and model chat-template
    boundary tokens. The executable protocol should therefore be judged from the
    parsed trace whenever it is available, not from raw concatenated text.
    """
    if not turns:
        return False
    types = [t.get("type") for t in turns if isinstance(t, dict)]
    if "search" not in types or "answer" not in types:
        return False
    answer_idx = max(i for i, typ in enumerate(types) if typ == "answer")
    if not require_state_update:
        return any(i < answer_idx for i, typ in enumerate(types) if typ == "search")

    state_indices = [i for i, typ in enumerate(types) if typ == "state_update"]
    if not state_indices or answer_idx < state_indices[-1]:
        return False

    for i, typ in enumerate(types):
        if typ != "search":
            continue
        next_action = next(
            (later for later in types[i + 1:] if later in {"search", "state_update", "answer"}),
            None,
        )
        if next_action != "state_update":
            return False
    return True


def trace_soft_format(turns: list[dict] | None, require_state_update: bool = False) -> float:
    if is_valid_trace_sequence(turns, require_state_update=require_state_update):
        return 1.0
    if not turns:
        return 0.0
    types = [t.get("type") for t in turns if isinstance(t, dict)]
    if "search" in types and (not require_state_update or "state_update" in types):
        return 0.5
    if "answer" in types:
        return 0.2
    return 0.0


# ---------------- Label extraction ----------------
def _to_str_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple, set)):
        return [str(e) for e in x if e is not None and str(e)]
    return [str(x)]


def _extract_label(sample_or_label) -> dict:
    if isinstance(sample_or_label, dict):
        return sample_or_label
    label = getattr(sample_or_label, "label", None)
    return label if isinstance(label, dict) else {}


def _extract_targets(sample_or_label) -> list[str]:
    label = _extract_label(sample_or_label)
    if label:
        gt = label.get("ground_truth", {})
        if isinstance(gt, dict) and "target" in gt:
            return _to_str_list(gt["target"])
        if label.get("target") is not None:
            return _to_str_list(label["target"])
    if isinstance(sample_or_label, str):
        return [sample_or_label]
    label = getattr(sample_or_label, "label", None)
    if isinstance(label, str):
        return [label]
    if label is not None and not isinstance(label, dict):
        return _to_str_list(label)
    return []


def _extract_gold_evidence_names(sample_or_label) -> list[str]:
    label = _extract_label(sample_or_label)
    names = []
    for key in ("evidence_file_names", "support_file_names", "gold_file_names"):
        names.extend(_to_str_list(label.get(key)))
    gt = label.get("ground_truth", {})
    if isinstance(gt, dict):
        for key in ("evidence_file_names", "support_file_names", "gold_file_names"):
            names.extend(_to_str_list(gt.get(key)))
    seen = set()
    out = []
    for n in names:
        nn = _norm_evidence_name(n)
        if nn and nn not in seen:
            seen.add(nn)
            out.append(n)
    return out


def _extract_hop(sample_or_label) -> int | None:
    label = _extract_label(sample_or_label)
    hop = label.get("hop")
    if hop is None and isinstance(label.get("ground_truth"), dict):
        hop = label["ground_truth"].get("hop")
    try:
        return int(hop) if hop is not None else None
    except (TypeError, ValueError):
        return None


# ---------------- Shared non-evidence rewards ----------------
def compute_outcome_reward(answer: str | None, targets: list[str]) -> tuple[float, int, float]:
    if answer is None or not answer.strip():
        return 0.0, 0, 0.0
    em_val = em_check(answer, targets)
    f1_val = max_f1(answer, targets)
    # Keep EM dominant but allow useful gradients for near-miss answers.
    outcome = 1.0 if em_val else 0.5 * f1_val
    return max(0.0, min(1.0, outcome)), int(em_val), float(f1_val)


def compute_efficiency_reward(text: str, hop: int | None = None) -> float:
    queries = extract_search_queries(text)
    if not queries:
        return 0.0
    norm_queries = [normalize_answer(q) for q in queries if normalize_answer(q)]
    if not norm_queries:
        return 0.0
    unique_ratio = len(set(norm_queries)) / len(norm_queries)
    info_ratio = min(len(extract_information_blocks(text)), len(norm_queries)) / len(norm_queries)

    expected = hop if hop and hop > 0 else min(2, len(norm_queries))
    expected = max(1, min(expected, int(os.environ.get("MUSIQUE_MAX_TURNS", "5"))))
    if len(norm_queries) < expected:
        turn_score = len(norm_queries) / expected
    elif len(norm_queries) <= expected + 1:
        turn_score = 1.0
    else:
        turn_score = (expected + 1) / len(norm_queries)

    return max(0.0, min(1.0, unique_ratio * info_ratio * turn_score))


def is_retrieval_correct(text: str, golds: list[str]) -> bool:
    infos = " ".join(extract_information_blocks(text))
    if not infos:
        return False
    ninfo = normalize_answer(infos)
    return any(normalize_answer(g) and normalize_answer(g) in ninfo for g in (golds or []))


# ---------------- Evidence/path/retrieval rule rewards ----------------
def _norm_evidence_name(name: Any) -> str:
    s = str(name or "")
    s = re.sub(r"\s+", " ", s).strip()
    return normalize_answer(s)


def _evidence_match_score(pred_name: str, gold_names: list[str]) -> float:
    npred = _norm_evidence_name(pred_name)
    if not npred:
        return 0.0
    best = 0.0
    for gold in gold_names:
        ngold = _norm_evidence_name(gold)
        if not ngold:
            continue
        if npred == ngold:
            return 1.0
        if npred in ngold or ngold in npred:
            best = max(best, 0.75)
    return best


def _f1_from_counts(tp: float, pred_n: int, gold_n: int) -> tuple[float, float, float]:
    if pred_n <= 0 or gold_n <= 0 or tp <= 0:
        return 0.0, 0.0, 0.0
    precision = min(1.0, tp / pred_n)
    recall = min(1.0, tp / gold_n)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _state_caps_from_sample(sample) -> list[dict]:
    meta = getattr(sample, "metadata", None)
    if not isinstance(meta, dict):
        return []
    turn_states = meta.get("turn_states", []) or []
    final_state = turn_states[-1] if turn_states and isinstance(turn_states[-1], dict) else {}
    return [c for c in (final_state.get("E", []) or []) if isinstance(c, dict)]


def _kept_source_names(caps: list[dict]) -> list[str]:
    names = []
    for cap in caps:
        source = cap.get("source") or {}
        name = source.get("file_name") or source.get("document_id") or ""
        if name:
            names.append(str(name))
    return names


def _turn_source_sets(sample) -> list[set[str]]:
    meta = getattr(sample, "metadata", None)
    if not isinstance(meta, dict):
        return []
    sets = []
    for event in ((meta.get("trace") or {}).get("turns") or []):
        if not isinstance(event, dict):
            continue
        names = set()
        if event.get("type") == "search":
            for doc in event.get("retrieved", []) or []:
                if isinstance(doc, dict):
                    name = doc.get("source") or doc.get("file_name") or doc.get("document_id")
                    if name:
                        names.add(str(name))
        elif event.get("type") == "state_update":
            for ev in event.get("evidence_kept", []) or []:
                if isinstance(ev, dict) and ev.get("source"):
                    names.add(str(ev["source"]))
        if names:
            sets.append(names)
    return sets


def _search_source_names(sample) -> list[str]:
    meta = getattr(sample, "metadata", None)
    if not isinstance(meta, dict):
        return []
    names = []
    for event in ((meta.get("trace") or {}).get("turns") or []):
        if not isinstance(event, dict) or event.get("type") != "search":
            continue
        for doc in event.get("retrieved", []) or []:
            if isinstance(doc, dict):
                name = doc.get("source") or doc.get("file_name") or doc.get("document_id")
                if name:
                    names.append(str(name))
    return names


def _kept_turn_source_sets(sample) -> list[set[str]]:
    meta = getattr(sample, "metadata", None)
    if not isinstance(meta, dict):
        return []
    sets = []
    for event in ((meta.get("trace") or {}).get("turns") or []):
        if not isinstance(event, dict) or event.get("type") != "state_update":
            continue
        names = set()
        for ev in event.get("evidence_kept", []) or []:
            if isinstance(ev, dict) and ev.get("source"):
                names.add(str(ev["source"]))
        if names:
            sets.append(names)
    return sets


def _coverage_count(pred_names: list[str], gold_names: list[str]) -> float:
    if not pred_names or not gold_names:
        return 0.0
    matched_gold = set()
    score = 0.0
    for pred in pred_names:
        best_i = None
        best_s = 0.0
        for i, gold in enumerate(gold_names):
            if i in matched_gold:
                continue
            s = _evidence_match_score(pred, [gold])
            if s > best_s:
                best_i = i
                best_s = s
        if best_i is not None and best_s > 0:
            matched_gold.add(best_i)
            score += best_s
    return score


def compute_rule_evidence_metrics(sample, targets: list[str] | None = None) -> dict[str, float]:
    """Compute StateFlow-only evidence metrics from rule-visible fields.

    Preferred signal: MuSiQue gold evidence file/title names vs canonical state
    source names. Fallback: old answer-in-kept-evidence heuristic so smoke tests
    without support metadata still get a weak but non-crashing signal.
    """
    targets = targets or _extract_targets(sample)
    gold_names = _extract_gold_evidence_names(sample)
    hop = _extract_hop(sample)
    caps = _state_caps_from_sample(sample)
    kept_names = _kept_source_names(caps)
    state_precision = state_recall = state_f1 = 0.0

    if gold_names:
        tp = _coverage_count(kept_names, gold_names)
        state_precision, state_recall, state_f1 = _f1_from_counts(tp, len(kept_names), len(gold_names))
    else:
        ev_text = " ".join((cap.get("text") or "") for cap in caps)
        ninfo = normalize_answer(ev_text)
        answer_hit = any(normalize_answer(g) and normalize_answer(g) in ninfo for g in (targets or []))
        state_precision = state_recall = state_f1 = 1.0 if answer_hit else (0.3 if caps else 0.0)

    # Retrieval reward: did the search observations expose gold support docs?
    retrieved_names = _search_source_names(sample)
    if gold_names:
        retrieval = min(1.0, _coverage_count(retrieved_names, gold_names) / max(1, len(gold_names)))
    else:
        retrieval = 1.0 if is_retrieval_correct(getattr(sample, "response", "") or "", targets) else 0.0

    # Path reward: order-agnostic coverage of distinct gold support docs that
    # survived canonical state updates. Raw retrieval exposure is scored above.
    if gold_names:
        kept_turn_sets = _kept_turn_source_sets(sample)
        covered = set()
        for turn in kept_turn_sets:
            for i, gold in enumerate(gold_names):
                if i not in covered and any(_evidence_match_score(n, [gold]) > 0 for n in turn):
                    covered.add(i)
        denom = min(len(gold_names), hop) if hop else len(gold_names)
        path = min(1.0, len(covered) / max(1, denom))
    else:
        path = state_recall

    return {
        "state": max(0.0, min(1.0, state_f1)),
        "state_precision": float(state_precision),
        "state_recall": float(state_recall),
        "path": max(0.0, min(1.0, path)),
        "retrieval": max(0.0, min(1.0, retrieval)),
    }


# ---------------- Final score composition ----------------
DEFAULT_WEIGHTS = {
    "outcome": float(os.environ.get("MUSIQUE_R_OUTCOME", "0.45")),
    "state": float(os.environ.get("MUSIQUE_R_STATE", "0.25")),
    "path": float(os.environ.get("MUSIQUE_R_PATH", "0.15")),
    "retrieval": float(os.environ.get("MUSIQUE_R_RETRIEVAL", "0.10")),
    "efficiency": float(os.environ.get("MUSIQUE_R_EFFICIENCY", "0.05")),
    # Outside the gate, capped at 0.05; bare answers therefore get <=0.01.
    "soft_format": float(os.environ.get("MUSIQUE_R_SOFT_FORMAT", "0.05")),
}


def _clamp01(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, v))


def compute_score(
    solution_str: str,
    targets: list[str],
    *,
    label: dict | None = None,
    state_reward: float | None = None,
    evidence_metrics: dict[str, float] | None = None,
    trace_turns: list[dict] | None = None,
    include_evidence: bool = True,
    require_state_update: bool = False,
) -> dict:
    """Compute the unified reward.

    Baseline calls this with include_evidence=False. StateFlow passes rule-scored
    evidence_metrics and include_evidence=True.
    """
    text = solution_str or ""
    label = label or {}
    answer = extract_solution(text)
    outcome, em_val, f1_val = compute_outcome_reward(answer, targets)
    if trace_turns is not None:
        fmt = trace_soft_format(trace_turns, require_state_update=require_state_update)
        gate = 1.0 if is_valid_trace_sequence(trace_turns, require_state_update=require_state_update) else 0.0
    else:
        fmt = soft_format(text, require_state_update=require_state_update)
        gate = gate_valid_sequence(text, require_state_update=require_state_update)
    hop = _extract_hop(label)
    efficiency = compute_efficiency_reward(text, hop=hop)

    ev = dict(evidence_metrics or {})
    if state_reward is not None and "state" not in ev:
        ev["state"] = state_reward
    state = _clamp01(ev.get("state", 0.0)) if include_evidence else 0.0
    path = _clamp01(ev.get("path", 0.0)) if include_evidence else 0.0
    retrieval = _clamp01(ev.get("retrieval", 0.0)) if include_evidence else 0.0

    w = DEFAULT_WEIGHTS
    gated = (
        w["outcome"] * outcome
        + w["state"] * state
        + w["path"] * path
        + w["retrieval"] * retrieval
        + w["efficiency"] * efficiency
    )
    if not include_evidence:
        shared_weight = w["outcome"] + w["efficiency"]
        target_weight = max(0.0, 1.0 - w["soft_format"])
        gated = target_weight * gated / shared_weight if shared_weight > 0 else 0.0
    reward = w["soft_format"] * fmt + gate * gated
    reward = max(0.0, min(1.0, reward))

    return {
        "score": float(reward),
        "em": int(em_val),
        "f1": float(f1_val),
        "outcome": float(outcome),
        "format": int(gate if trace_turns is not None else is_valid_format(text, require_state_update=require_state_update)),
        "format_soft": float(fmt),
        "gate": float(gate),
        "has_answer": has_valid_final_answer(text),
        "efficiency": float(efficiency),
        "state": float(state),
        "state_precision": float(_clamp01(ev.get("state_precision", state))),
        "state_recall": float(_clamp01(ev.get("state_recall", state))),
        "path": float(path),
        "retrieval": float(retrieval),
        "process": float(0.25 * state + 0.15 * path + 0.10 * retrieval + 0.05 * efficiency),
    }


def compute_eval_em_score(solution_str: str, targets: list[str]) -> dict:
    """Evaluation-only metric: exact-match 0/1, with no shaped reward terms."""
    answer = extract_solution(solution_str or "")
    em_val = em_check(answer or "", targets) if answer is not None else 0
    f1_val = max_f1(answer or "", targets) if answer is not None else 0.0
    return {
        "score": float(em_val),
        "em": int(em_val),
        "f1": float(f1_val),
        "outcome": float(em_val),
        "format": int(answer is not None and bool(str(answer).strip())),
        "format_soft": 0.0,
        "gate": 0.0,
        "has_answer": int(answer is not None and bool(str(answer).strip())),
        "efficiency": 0.0,
        "state": 0.0,
        "state_precision": 0.0,
        "state_recall": 0.0,
        "path": 0.0,
        "retrieval": 0.0,
        "process": 0.0,
        "eval_0_1": 1,
    }


async def reward_func(args, sample, **kwargs):
    """Baseline-compatible reward entrypoint.

    This intentionally excludes evidence/state/path/retrieval components so the
    Search-R1 GRPO baseline shares the non-evidence reward base with StateFlow.
    """
    response = getattr(sample, "response", "") or ""
    targets = _extract_targets(sample)
    label = _extract_label(sample)
    meta = getattr(sample, "metadata", None)
    trace_turns = None
    if isinstance(meta, dict):
        trace_turns = (meta.get("trace") or {}).get("turns")
    if kwargs.get("evaluation") or (isinstance(meta, dict) and meta.get("_is_eval")):
        result = compute_eval_em_score(baseline_reward_view(response), targets)
        if isinstance(meta, dict):
            meta["reward_detail"] = result
            _write_baseline_trace(sample, result, targets)
        return result["score"]
    result = compute_score(
        baseline_reward_view(response),
        targets,
        label=label,
        trace_turns=trace_turns,
        include_evidence=False,
        require_state_update=False,
    )
    if isinstance(meta, dict):
        meta["reward_detail"] = result
        _write_baseline_trace(sample, result, targets)
    return result["score"]


def _write_baseline_trace(sample, result, targets):
    trace_dir = os.environ.get("MUSIQUE_TRACE_DIR")
    if not trace_dir:
        return
    try:
        import json as _json

        os.makedirs(trace_dir, exist_ok=True)
        meta = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
        trace = dict(meta.get("trace", {}) or {})
        trace["targets"] = targets
        trace["em"] = result.get("em", 0)
        trace["f1"] = result.get("f1", 0.0)
        trace["reward"] = result.get("score", 0.0)
        trace["outcome"] = result.get("outcome", 0.0)
        trace["gate"] = result.get("gate", 0.0)
        trace["efficiency"] = result.get("efficiency", 0.0)
        trace["format_soft"] = result.get("format_soft", 0.0)
        trace["status"] = getattr(getattr(sample, "status", None), "name", "")
        trace["final_answer"] = extract_solution(baseline_reward_view(getattr(sample, "response", "") or ""))
        with open(os.path.join(trace_dir, "traces.jsonl"), "a", encoding="utf-8") as f:
            f.write(_json.dumps(trace, ensure_ascii=False) + "\n")
        if result.get("em"):
            with open(os.path.join(trace_dir, "correct_traces.jsonl"), "a", encoding="utf-8") as f:
                f.write(_json.dumps(trace, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"[musique_reward] write baseline trace error: {e}", flush=True)
