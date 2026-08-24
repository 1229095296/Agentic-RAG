"""StateFlow rollout with a strict, model-authored evidence state.

Unlike the legacy StateFlow generator, this implementation never repairs,
filters, summarizes, or backfills a model state. After every retrieval the
model must emit the complete evidence state. A deterministic Gate either
accepts that state verbatim or terminates the rollout with reward zero.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import aiohttp

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from slime_integration.direct_evidence_state import (
    DirectEvidenceState,
    parse_and_validate_state,
    state_from_anchor,
)
from slime_integration.evidence_state import EvidenceCapsule, EvidenceState, init_state
from slime_integration.musique_reward import (
    _extract_targets,
    baseline_reward_view,
    compute_eval_em_score,
    compute_score,
    extract_solution,
)
from slime_integration.prompt_utils import extract_question_id, extract_question_text
from slime_integration.sentence_utils import make_spans


CONFIG = {
    "max_turns": int(os.environ.get("MUSIQUE_MAX_TURNS", "3")),
    "max_steps": int(os.environ.get("STATEFLOW_MAX_STEPS", "10")),
    "topk": int(os.environ.get("MUSIQUE_TOPK", "3")),
    "search_concurrency": int(os.environ.get("MUSIQUE_SEARCH_CONCURRENCY", "256")),
    "search_url": os.environ.get("MUSIQUE_SEARCH_URL", "http://127.0.0.1:8000/retrieve"),
    "search_timeout": int(os.environ.get("MUSIQUE_SEARCH_TIMEOUT", "60")),
    "corpus_version": os.environ.get("MUSIQUE_CORPUS_VERSION", "musique_v1"),
    "return_logprob": True,
    "fewshot": os.environ.get("STATEFLOW_FEWSHOT", "1") == "1",
    "clean_prompt": os.environ.get("MUSIQUE_CLEAN_PROMPT", "1") == "1",
}

SEMAPHORE = asyncio.Semaphore(CONFIG["search_concurrency"])

_FEWSHOT_PREAMBLE = (
    "You solve multi-hop questions by searching step by step.\n"
    "Use the search engine by outputting exactly one executable action at a time. "
    "Do not describe the format, do not mention that you cannot browse, and do not add prose "
    "before action tags. After EVERY <information>, your next action must be exactly one "
    "<evidence_state> JSON action. The JSON is the complete current state, not a delta. "
    "Rules will validate it exactly and will not repair it.\n"
    "The state schema is exactly: "
    '{"evidence":[{"span_id":"doc:sN","text":"verbatim source excerpt","role":"bridge|direct"}],'
    '"open_questions":["one concrete missing fact"],"decision":"search|answer"}.\n'
    "For each evidence item, copy either the whole visible span or one informative contiguous "
    "excerpt exactly as written; never paraphrase or combine text across spans. Keep still-needed earlier evidence. "
    "Select only the smallest useful evidence set, usually one to three items and never more than six. "
    "When source text contains quotation marks or backslashes, JSON-escape them without changing the decoded text. "
    "Use decision search with at least one open question when another retrieval is needed. "
    "Use decision answer only with non-empty evidence and an empty open_questions list. "
    "Inside or outside <evidence_state>, do not add comments, markdown, thinking, or extra keys. "
    "After an accepted search decision, output one <search> action. After an accepted answer "
    "decision, output one <answer> action.\n\n"
    "=== EXAMPLE ===\n"
    "Question: Which river runs by the headquarters city of company X?\n"
    "<think>First I need company X's headquarters city.</think>\n"
    "<search>company X headquarters city</search>\n"
    "<information>\n[dA:s1] Company X is headquartered in Springfield.\n"
    "[dA:s2] It was founded in 1990.\n</information>\n"
    '<evidence_state>{"evidence":[{"span_id":"dA:s1","text":"Company X is headquartered in Springfield.",'
    '"role":"bridge"}],"open_questions":["Which river runs by Springfield?"],'
    '"decision":"search"}</evidence_state>\n'
    "<think>Now I need the river by Springfield.</think>\n"
    "<search>river by Springfield</search>\n"
    "<information>\n[dB:s1] Springfield lies on the Green River.\n</information>\n"
    '<evidence_state>{"evidence":[{"span_id":"dA:s1","text":"Company X is headquartered in Springfield.",'
    '"role":"bridge"},{"span_id":"dB:s1","text":"Springfield lies on the Green River.",'
    '"role":"direct"}],"open_questions":[],"decision":"answer"}</evidence_state>\n'
    "<answer>Green River</answer>\n"
    "=== END EXAMPLE ===\n\n"
    "Now solve the real question the same way.\n"
)

_CHAT_PROMPT_RE = re.compile(
    r"(?P<prefix>.*?<\|im_start\|>user\n)(?P<content>.*?)(?P<suffix><\|im_end\|>\n<\|im_start\|>assistant\n?)\s*$",
    re.DOTALL,
)
_CHAT_CONTROL_RE = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>(?:assistant|user|system)?")
_THINK_PREFIX_RE = re.compile(r"\s*(?:<think>.*?</think>\s*)?", re.DOTALL)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_STATE_RE = re.compile(r"<evidence_state>.*?</evidence_state>", re.DOTALL)


def _build_prompt(prompt_text: str, sample: Sample) -> tuple[str, str]:
    """Mirror the clean baseline prompt while adding only the state protocol."""
    raw_prompt = str(prompt_text or "")
    meta = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    question = extract_question_text(meta.get("question_text") or meta.get("question") or raw_prompt)
    if not question:
        question = extract_question_text(raw_prompt)
    if not CONFIG["clean_prompt"]:
        return raw_prompt, question

    content = f"{_FEWSHOT_PREAMBLE if CONFIG['fewshot'] else ''}Question: {question}\n"
    match = _CHAT_PROMPT_RE.match(raw_prompt)
    if match:
        return f"{match.group('prefix')}{content}{match.group('suffix')}", question
    return content, question


def _build_replay_prompt(prompt_text: str, sample: Sample, state: DirectEvidenceState) -> tuple[str, str]:
    """Start replay from a certified prefix without asking rules to create a new state."""
    _, question = _build_prompt(prompt_text, sample)
    body = (
        f"{_FEWSHOT_PREAMBLE if CONFIG['fewshot'] else ''}Question: {question}\n"
        f"<certified_evidence_state>{state.to_json()}</certified_evidence_state>\n"
        "This state was accepted in an earlier prefix. Either output one <search> action for a missing "
        "fact, or output one <answer> action if it is already answer-ready. Do not output a new "
        "<evidence_state> until after receiving new <information>.\n"
    )
    raw_prompt = str(prompt_text or "")
    match = _CHAT_PROMPT_RE.match(raw_prompt)
    if match:
        return f"{match.group('prefix')}{body}{match.group('suffix')}", question
    return body, question


def _strip_chat_control(text: str) -> str:
    return _CHAT_CONTROL_RE.sub("", text or "").strip()


def _parse_action(text: str) -> tuple[str | None, str]:
    """Parse exactly one action; evidence_state intentionally permits no think prefix."""
    cleaned = _strip_chat_control(text)
    state_match = _STATE_RE.fullmatch(cleaned)
    if state_match:
        return "evidence_state", cleaned

    # Qwen3.5's assistant template can supply the opening thinking marker, so
    # the decoded continuation begins with a lone closing marker. This is a
    # template boundary, not free-form prose. Evidence-state actions remain
    # strict because this tolerance is applied only to search/answer parsing.
    action_text = re.sub(r"^\s*</think>\s*", "", cleaned, count=1)
    prefix = _THINK_PREFIX_RE.match(action_text)
    action_text = action_text[prefix.end() :] if prefix else action_text
    match = _SEARCH_RE.fullmatch(action_text)
    if match and match.group(1).strip():
        return "search", match.group(1).strip()
    match = _ANSWER_RE.fullmatch(action_text)
    if match and match.group(1).strip():
        return "answer", match.group(1).strip()
    return None, ""


def _ensure_spans(docs: list) -> list:
    for i, doc in enumerate(docs or []):
        data = doc.get("document", doc) if isinstance(doc, dict) else None
        if not isinstance(data, dict) or data.get("spans"):
            continue
        text = data.get("contents") or data.get("text") or data.get("title") or ""
        doc_id = str(data.get("doc_id") or data.get("id") or f"d{i}")
        if text:
            data["spans"] = make_spans(doc_id, text)
    return docs


async def _retrieve(query: str) -> list:
    payload = {"queries": [query], "topk": CONFIG["topk"], "return_scores": False}
    timeout = aiohttp.ClientTimeout(total=CONFIG["search_timeout"])
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(CONFIG["search_url"], json=payload, timeout=timeout) as response:
                response.raise_for_status()
                result = await response.json()
    except Exception as exc:  # noqa: BLE001
        print(f"[generate_stateflow_direct] search error: {exc}", flush=True)
        return []
    return _ensure_spans(result.get("result", [[]])[0])


def _visible_spans(docs: list) -> dict[str, dict[str, Any]]:
    visible = {}
    for doc in docs or []:
        data = doc.get("document", doc) if isinstance(doc, dict) else {}
        file_name = data.get("file_name") or data.get("title") or ""
        doc_id = data.get("doc_id") or data.get("id") or ""
        for span in data.get("spans", []) or []:
            span_id = span.get("span_id")
            if span_id:
                visible[span_id] = {
                    "text": str(span.get("text") or ""),
                    "source": {
                        "document_id": doc_id,
                        "file_name": file_name,
                        "sentence_ids": [span_id],
                        "corpus_version": CONFIG["corpus_version"],
                    },
                }
    return visible


def _render_information(visible: dict[str, dict[str, Any]]) -> str:
    lines = [f"[{span_id}] {item['text']}" for span_id, item in visible.items()]
    body = "\n".join(lines) if lines else "(no results)"
    return f"\n\n<information>\n{body}\n</information>\n\n"


def _state_to_pool_state(
    direct: DirectEvidenceState,
    *,
    qid: str,
    turn_index: int,
    available: dict[str, dict[str, Any]],
    search_rounds_left: int,
    trajectory_id: str,
) -> EvidenceState:
    """Losslessly adapt accepted model text to the existing replay pool schema."""
    capsules = []
    for index, item in enumerate(direct.evidence, 1):
        source = dict((available.get(item.span_id) or {}).get("source") or {})
        capsules.append(
            EvidenceCapsule(
                evidence_id=f"e{index}",
                span_ids=[item.span_id],
                text=item.text,
                source=source,
                supports=[],
                role=item.role,
                provenance_valid=True,
            )
        )
    unresolved = [
        {"constraint_id": f"c{index}", "text": question, "depends_on": []}
        for index, question in enumerate(direct.open_questions, 1)
    ]
    return EvidenceState(
        question_id=qid,
        turn_index=turn_index,
        E=capsules,
        U=unresolved,
        H=[],
        B={"search_rounds_left": max(0, search_rounds_left)},
        evidence_chain_signature="::".join(item.span_id for item in direct.evidence),
        corpus_index_version=CONFIG["corpus_version"],
        trajectory_id=trajectory_id,
    )


def _prior_spans(state: DirectEvidenceState, pool_state: EvidenceState | None) -> dict[str, dict[str, Any]]:
    sources = {}
    source_by_id = {}
    if pool_state is not None:
        for capsule in pool_state.E:
            if len(capsule.span_ids) == 1:
                source_by_id[capsule.span_ids[0]] = dict(capsule.source or {})
    for item in state.evidence:
        sources[item.span_id] = {"text": item.text, "source": source_by_id.get(item.span_id, {})}
    return sources


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    assert not args.partial_rollout, "Partial rollout not supported here."
    generator_state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    meta = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    is_replay = bool(meta.get("is_replay") and meta.get("anchor_state"))
    direct_state = state_from_anchor(meta.get("anchor_state")) if is_replay else DirectEvidenceState()
    pool_state = EvidenceState.from_dict(meta["anchor_state"]) if is_replay else None
    if is_replay:
        prompt_text, question = _build_replay_prompt(sample.prompt, sample, direct_state)
        phase = "decision"
    else:
        prompt_text, question = _build_prompt(sample.prompt, sample)
        phase = "search"
    sample.prompt = prompt_text
    prompt_token_ids = generator_state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    qid = extract_question_id(meta, sample.label if isinstance(sample.label, dict) else None)
    trajectory_id = str(meta.get("trajectory_id") or "")
    initial_pool_state = pool_state or init_state(
        qid,
        search_rounds=CONFIG["max_turns"],
        corpus_index_version=CONFIG["corpus_version"],
        trajectory_id=trajectory_id,
    )
    turn_states = [initial_pool_state.to_dict()]
    accepted_state_count = 0
    search_rounds = 0
    current_visible: dict[str, dict[str, Any]] = {}
    available = _prior_spans(direct_state, pool_state)
    trace_turns: list[dict[str, Any]] = []
    protocol_valid = True
    failure_code = ""
    failure_message = ""
    done = False
    output = None

    response = ""
    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] | None = [] if CONFIG["return_logprob"] else None

    params = dict(sampling_params)
    stop = list(params.get("stop", []) or [])
    for tag in ["</search>", "</evidence_state>", "</answer>", "<|im_end|>", "<|endoftext|>"]:
        if tag not in stop:
            stop.append(tag)
    params["stop"] = stop
    # SGLang trims matched stop strings by default. Keep the model-emitted
    # closing action tag so parser, reward text, tokens, and logprobs align.
    params["no_stop_trim"] = True

    def append_model(text: str, output_meta: dict):
        nonlocal response
        if CONFIG["return_logprob"]:
            values = output_meta.get("output_token_logprobs")
            if values is None:
                raise RuntimeError("output_token_logprobs missing; set return_logprob=True")
            ids = [item[1] for item in values]
            log_probs = [item[0] for item in values]
        else:
            ids = generator_state.tokenizer(text, add_special_tokens=False)["input_ids"]
            log_probs = []
        response += text
        response_token_ids.extend(ids)
        loss_mask.extend([1] * len(ids))
        if rollout_log_probs is not None:
            rollout_log_probs.extend(log_probs)

    def append_environment(text: str):
        nonlocal response
        ids = generator_state.tokenizer(text, add_special_tokens=False)["input_ids"]
        response += text
        response_token_ids.extend(ids)
        loss_mask.extend([0] * len(ids))
        if rollout_log_probs is not None:
            rollout_log_probs.extend([0.0] * len(ids))

    def fail(code: str, message: str, raw: str):
        nonlocal protocol_valid, failure_code, failure_message
        protocol_valid = False
        failure_code = code
        failure_message = message
        trace_turns.append({"type": "gate_failure", "code": code, "message": message, "raw": raw[:1200]})

    for _step in range(CONFIG["max_steps"]):
        payload = {"text": prompt_text + response, "sampling_params": params}
        if CONFIG["return_logprob"]:
            payload["return_logprob"] = True
        output = await post(url, payload)
        finish_reason = output["meta_info"]["finish_reason"]["type"]
        if finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            return sample
        current = output["text"]
        append_model(current, output["meta_info"])
        if finish_reason == "length":
            fail("GENERATION_LENGTH", "generation stopped before a complete action", current)
            break

        action, content = _parse_action(current)
        if phase in {"search", "decision"} and action == "search":
            if search_rounds >= CONFIG["max_turns"]:
                fail("SEARCH_BUDGET_EXHAUSTED", "search action exceeds the configured budget", current)
                break
            search_rounds += 1
            async with SEMAPHORE:
                docs = await _retrieve(content)
            current_visible = _visible_spans(docs)
            available.update(current_visible)
            append_environment(_render_information(current_visible))
            trace_turns.append(
                {
                    "type": "search",
                    "query": content,
                    "retrieved": [
                        {
                            "span_id": span_id,
                            "text": value["text"],
                            "source": (value.get("source") or {}).get("file_name", ""),
                        }
                        for span_id, value in list(current_visible.items())[:24]
                    ],
                }
            )
            phase = "evidence"
            continue

        if phase == "evidence" and action == "evidence_state":
            validation = parse_and_validate_state(
                content,
                available,
                require_current_span_ids=set(current_visible),
            )
            if not validation.ok or validation.state is None:
                fail(validation.code, validation.message, current)
                break
            direct_state = validation.state
            accepted_state_count += 1
            pool_state = _state_to_pool_state(
                direct_state,
                qid=qid,
                turn_index=(pool_state.turn_index if pool_state else 0) + 1,
                available=available,
                search_rounds_left=CONFIG["max_turns"] - search_rounds,
                trajectory_id=trajectory_id,
            )
            turn_states.append(pool_state.to_dict())
            trace_turns.append(
                {
                    "type": "evidence_state",
                    "state": direct_state.to_dict(),
                    "gate": 1,
                    "new_span_ids": sorted(set(current_visible) & {item.span_id for item in direct_state.evidence}),
                }
            )
            # The accepted state is a replacement, not an append-only bag.
            # Unselected spans are no longer valid provenance in later turns.
            available = _prior_spans(direct_state, pool_state)
            if direct_state.decision == "search":
                append_environment("\nEvidence state accepted. Output exactly one SEARCH action next.\n")
                phase = "search"
            else:
                append_environment("\nEvidence state accepted. Output exactly one ANSWER action next.\n")
                phase = "answer"
            current_visible = {}
            continue

        if phase in {"decision", "answer"} and action == "answer":
            trace_turns.append({"type": "answer", "text": content})
            done = True
            break

        expected = {
            "search": "search",
            "decision": "search or answer",
            "evidence": "evidence_state",
            "answer": "answer",
        }[phase]
        fail("UNEXPECTED_ACTION", f"expected {expected}, received {action or 'invalid output'}", current)
        break

    if protocol_valid and not done:
        fail("INCOMPLETE_TRAJECTORY", "rollout ended without a final answer", "")

    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text
    if rollout_log_probs is not None:
        sample.rollout_log_probs = rollout_log_probs or None
    if not isinstance(getattr(sample, "metadata", None), dict):
        sample.metadata = {}
    sample.metadata.update(
        {
            "_is_eval": bool(evaluation),
            "turn_states": turn_states,
            "num_search_rounds": search_rounds,
            "protocol_valid": protocol_valid,
            "gate_failure_code": failure_code,
            "gate_failure_message": failure_message,
            "accepted_state_count": accepted_state_count,
            "final_direct_state": direct_state.to_dict(),
            "trace": {
                "qid": qid,
                "question_text": question,
                "is_replay": is_replay,
                "turns": trace_turns,
            },
        }
    )
    # A Gate rejection is a completed negative sample, not an infrastructure
    # failure. Keeping it completed lets GRPO learn from its explicit zero.
    sample.status = Sample.Status.COMPLETED
    return sample


def _reward_trace(sample: Sample) -> list[dict]:
    meta = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    turns = list((meta.get("trace") or {}).get("turns") or [])
    if meta.get("is_replay") and not any(turn.get("type") == "search" for turn in turns):
        # The certified anchor represents the omitted search prefix. This is
        # used only by the shared sequence Gate, never as evidence supervision.
        return [{"type": "search", "certified_replay_prefix": True}] + turns
    return turns


async def reward_func(args, sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("sample must be a Sample instance")
    meta = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    targets = _extract_targets(sample)
    evaluation = bool(kwargs.get("evaluation") or meta.get("_is_eval"))
    if not meta.get("protocol_valid", False):
        result = compute_eval_em_score("", targets) if evaluation else compute_score(
            solution_str="",
            targets=targets,
            label=getattr(sample, "label", None),
            trace_turns=[],
            include_evidence=False,
            require_state_update=False,
        )
        result["protocol_gate"] = 0.0
        result["gate_failure_code"] = meta.get("gate_failure_code", "UNKNOWN")
    elif evaluation:
        result = compute_eval_em_score(baseline_reward_view(sample.response or ""), targets)
        result["protocol_gate"] = 1.0
    else:
        result = compute_score(
            solution_str=baseline_reward_view(sample.response or ""),
            targets=targets,
            label=getattr(sample, "label", None),
            trace_turns=_reward_trace(sample),
            include_evidence=False,
            require_state_update=False,
        )
        result["protocol_gate"] = 1.0
    meta["reward_detail"] = result
    _write_trace(sample, result, targets)
    return result["score"]


def _write_trace(sample: Sample, result: dict, targets: list[str]):
    trace_dir = os.environ.get("STATEFLOW_TRACE_DIR")
    if not trace_dir:
        return
    try:
        os.makedirs(trace_dir, exist_ok=True)
        meta = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
        trace = dict(meta.get("trace") or {})
        trace.update(
            {
                "targets": targets,
                "em": result.get("em", 0),
                "f1": result.get("f1", 0.0),
                "reward": result.get("score", 0.0),
                "outcome": result.get("outcome", 0.0),
                "gate": result.get("gate", 0.0),
                "protocol_gate": result.get("protocol_gate", 0.0),
                "gate_failure_code": meta.get("gate_failure_code", ""),
                "efficiency": result.get("efficiency", 0.0),
                "format_soft": result.get("format_soft", 0.0),
                "accepted_state_count": meta.get("accepted_state_count", 0),
                "status": getattr(getattr(sample, "status", None), "name", ""),
                "final_answer": extract_solution(baseline_reward_view(sample.response or "")),
            }
        )
        line = json.dumps(trace, ensure_ascii=False) + "\n"
        with open(os.path.join(trace_dir, "traces.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(line)
        if result.get("em"):
            with open(os.path.join(trace_dir, "correct_traces.jsonl"), "a", encoding="utf-8") as handle:
                handle.write(line)
    except Exception as exc:  # noqa: BLE001
        print(f"[generate_stateflow_direct] write trace error: {exc}", flush=True)
