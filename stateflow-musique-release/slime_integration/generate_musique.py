"""slime custom generate + reward for MuSiQue agentic RAG (step-2 minimal closed loop).

基于 slime examples/search-r1/generate_with_search.py 改写：
- 多轮 <think>/<search>/<information>/<answer> rollout，逻辑与 search-r1 一致（token 级、
  return_logprob、检索 token loss_mask=0）。
- 检索后端指向我们的检索服务（search-r1 兼容的 /retrieve 契约），URL 由环境变量
  MUSIQUE_SEARCH_URL 覆盖，默认 http://127.0.0.1:8000/retrieve。
- reward 复用 slime_integration.musique_reward（EM/F1 + 格式分），读 sample.label 的
  ground_truth.target（与 musique_to_slime.py 产出的 reward_model 结构兼容）。

通过 slime 启动参数接入：
  --custom-generate-function-path slime_integration.generate_musique.generate
  --custom-rm-path                slime_integration.generate_musique.reward_func
需保证 PYTHONPATH 含 rag-agent 项目根（使 slime_integration 可导入）。
"""
import asyncio
import os
import re

import aiohttp

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from slime_integration.musique_reward import baseline_reward_view, compute_eval_em_score, compute_score, _extract_targets
from slime_integration.prompt_utils import extract_question_id, extract_question_text

MUSIQUE_CONFIGS = {
    "max_turns": int(os.environ.get("MUSIQUE_MAX_TURNS", "3")),
    "topk": int(os.environ.get("MUSIQUE_TOPK", "3")),
    "search_concurrency": int(os.environ.get("MUSIQUE_SEARCH_CONCURRENCY", "64")),
    "search_url": os.environ.get("MUSIQUE_SEARCH_URL", "http://127.0.0.1:8000/retrieve"),
    "return_logprob": True,
    "format_score": float(os.environ.get("MUSIQUE_FORMAT_SCORE", "0.2")),
    "search_timeout": int(os.environ.get("MUSIQUE_SEARCH_TIMEOUT", "60")),
    "fewshot": os.environ.get("MUSIQUE_FEWSHOT", "1") == "1",
    "clean_prompt": os.environ.get("MUSIQUE_CLEAN_PROMPT", "1") == "1",
}

SEMAPHORE = asyncio.Semaphore(MUSIQUE_CONFIGS["search_concurrency"])


_FEWSHOT_PREAMBLE = (
    "You solve multi-hop questions by searching step by step.\n"
    "Use the search engine by outputting exactly one executable action at a time. "
    "Do not describe the format, do not mention that you cannot browse, and do not add prose "
    "before action tags. After you receive <information>, think briefly and either search again "
    "or provide the final answer.\n\n"
    "=== EXAMPLE ===\n"
    "Question: Which river runs by the headquarters city of company X?\n"
    "<think>First I need company X's headquarters city.</think>\n"
    "<search>company X headquarters city</search>\n"
    "<information>\n"
    "Doc 1(Title: Company X) Company X is headquartered in Springfield.\n"
    "</information>\n"
    "<think>Now I need the river by Springfield.</think>\n"
    "<search>river by Springfield</search>\n"
    "<information>\n"
    "Doc 1(Title: Springfield) Springfield lies on the Green River.\n"
    "</information>\n"
    "<think>The evidence identifies the river.</think>\n"
    "<answer>Green River</answer>\n"
    "=== END EXAMPLE ===\n\n"
    "Now solve the real question the same way.\n"
)


_CHAT_PROMPT_RE = re.compile(
    r"(?P<prefix>.*?<\|im_start\|>user\n)(?P<content>.*?)(?P<suffix><\|im_end\|>\n<\|im_start\|>assistant\n?)\s*$",
    re.DOTALL,
)
_CHAT_CONTROL_RE = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>(?:assistant|user|system)?")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_DANGLING_THINK_RE = re.compile(r"<think>.*$", re.DOTALL)
_ACTION_RE = re.compile(r"(?:^|\n)\s*<(search|answer)>(.*?)</\1>", re.DOTALL)


def _build_prompt(prompt_text: str, sample: Sample) -> tuple[str, str]:
    """Build a clean Search-R1-style baseline prompt without touching parquet files."""
    raw_prompt = str(prompt_text or "")
    meta = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    question_text = extract_question_text(meta.get("question_text") or meta.get("question") or raw_prompt)
    if not question_text:
        question_text = extract_question_text(raw_prompt)

    if not MUSIQUE_CONFIGS["clean_prompt"]:
        return raw_prompt, question_text

    content = f"{_FEWSHOT_PREAMBLE if MUSIQUE_CONFIGS['fewshot'] else ''}Question: {question_text}\n"
    match = _CHAT_PROMPT_RE.match(raw_prompt)
    if match:
        return f"{match.group('prefix')}{content}{match.group('suffix')}", question_text
    return content, question_text


def _clean_action_text(text: str) -> str:
    text = _CHAT_CONTROL_RE.sub("", text or "")
    text = _THINK_BLOCK_RE.sub("\n", text)
    text = _DANGLING_THINK_RE.sub("", text)
    return text.strip()


async def _local_search(query: str) -> list:
    """调用 search-r1 兼容的 /retrieve，返回 [{'document': {'contents': ...}}, ...]。"""
    payload = {"queries": [query], "topk": MUSIQUE_CONFIGS["topk"], "return_scores": False}
    timeout = aiohttp.ClientTimeout(total=MUSIQUE_CONFIGS["search_timeout"])
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(MUSIQUE_CONFIGS["search_url"], json=payload, timeout=timeout) as resp:
                resp.raise_for_status()
                result = await resp.json()
    except Exception as e:  # noqa: BLE001
        print(f"[generate_musique] search error at {MUSIQUE_CONFIGS['search_url']}: {e}", flush=True)
        return []
    hits = result.get("result", [[]])[0]
    contexts = []
    for item in hits:
        # 服务可能直接返回 doc（含 contents），也可能返回 {'document': doc}
        doc = item.get("document", item) if isinstance(item, dict) else {}
        contents = doc.get("contents", "")
        if contents:
            contexts.append({"document": {"contents": contents}})
    return contexts


def _passages2string(retrieval_result) -> str:
    out = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        out += f"Doc {idx+1}(Title: {title}) {text}\n"
    return out


async def _search(query: str) -> str:
    async with SEMAPHORE:
        result = await _local_search(query)
    return _passages2string(result)


def _postprocess_predictions(prediction: str):
    cleaned = _clean_action_text(prediction)
    for match in _ACTION_RE.finditer(cleaned):
        action = match.group(1)
        content = match.group(2).strip()
        # Ignore meta-format fragments such as "between <answer> and </answer>".
        if action == "answer" and content.lower() in {"and", "...", "answer"}:
            continue
        if action == "search" and content.lower() in {"query", "...", "search query"}:
            continue
        if content:
            return action, content
    return None, ""


async def _execute_predictions(prediction: str):
    action, content = _postprocess_predictions(prediction)
    if action == "search":
        results = await _search(content)
        return (
            f"\n\n<information>{results.strip()}</information>\n\n",
            False,
            {"type": "search", "query": content, "information": results[:2000]},
        )
    elif action == "answer":
        return "", True, {"type": "answer", "text": content}
    else:
        return (
            "\nMy previous action is invalid. If I want to search, I should put the query "
            "between <search> and </search>. If I want to give the final answer, I should put "
            "the answer between <answer> and </answer>. Let me try again.\n",
            False,
            None,
        )


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    assert not args.partial_rollout, "Partial rollout not supported here."
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_text, question_text = _build_prompt(sample.prompt, sample)
    sample.prompt = prompt_text
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_mask = []
    rollout_log_probs = [] if MUSIQUE_CONFIGS["return_logprob"] else None

    sampling_params = dict(sampling_params)
    stop = list(sampling_params.get("stop", []) or [])
    for tag in ["</search>", "</answer>", "<|im_end|>", "<|endoftext|>"]:
        if tag not in stop:
            stop.append(tag)
    sampling_params["stop"] = stop

    output = None
    trace_turns = []
    for _turn in range(MUSIQUE_CONFIGS["max_turns"]):
        payload = {"text": prompt_text + response, "sampling_params": sampling_params}
        if MUSIQUE_CONFIGS["return_logprob"]:
            payload["return_logprob"] = True

        output = await post(url, payload)

        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]
        if MUSIQUE_CONFIGS["return_logprob"]:
            if "output_token_logprobs" not in output["meta_info"]:
                raise RuntimeError("output_token_logprobs missing; set return_logprob=True in payload.")
            cur_response_token_ids = [it[1] for it in output["meta_info"]["output_token_logprobs"]]
            cur_response_log_probs = [it[0] for it in output["meta_info"]["output_token_logprobs"]]
        else:
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)
        if MUSIQUE_CONFIGS["return_logprob"]:
            rollout_log_probs += cur_response_log_probs

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        next_obs, done, event = await _execute_predictions(cur_response)
        if event:
            trace_turns.append(event)
        if done:
            break
        assert next_obs != "", "observation should not be empty"
        obs_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_ids
        loss_mask += [0] * len(obs_ids)
        if MUSIQUE_CONFIGS["return_logprob"]:
            rollout_log_probs += [0.0] * len(obs_ids)
            assert len(response_token_ids) == len(rollout_log_probs)

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text
    if not isinstance(getattr(sample, "metadata", None), dict):
        sample.metadata = {}
    sample.metadata["_is_eval"] = bool(evaluation)
    sample.metadata["trace"] = {
        "qid": extract_question_id(sample.metadata, getattr(sample, "label", None)),
        "question_text": question_text,
        "is_replay": False,
        "turns": trace_turns,
    }
    if MUSIQUE_CONFIGS["return_logprob"]:
        sample.rollout_log_probs = rollout_log_probs if rollout_log_probs else None

    reason = output["meta_info"]["finish_reason"]["type"] if output else "stop"
    if reason == "length":
        sample.status = Sample.Status.TRUNCATED
    elif reason == "abort":
        sample.status = Sample.Status.ABORTED
    else:
        sample.status = Sample.Status.COMPLETED
    return sample


async def reward_func(args, sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("sample must be a Sample instance")
    targets = _extract_targets(sample)
    trace_turns = None
    if isinstance(getattr(sample, "metadata", None), dict):
        trace_turns = (sample.metadata.get("trace") or {}).get("turns")
    if kwargs.get("evaluation") or (
        isinstance(getattr(sample, "metadata", None), dict) and sample.metadata.get("_is_eval")
    ):
        result = compute_eval_em_score(baseline_reward_view(sample.response or ""), targets)
        if isinstance(getattr(sample, "metadata", None), dict):
            sample.metadata["reward_detail"] = result
        return result["score"]
    # 仅解析 response（不拼 prompt——避免 prompt 格式说明/示例污染奖励）
    result = compute_score(
        solution_str=baseline_reward_view(sample.response or ""),
        targets=targets,
        label=getattr(sample, "label", None),
        trace_turns=trace_turns,
        include_evidence=False,
        require_state_update=False,
    )
    # slime 用 --reward-key 取标量；这里直接返回标量 score，附带指标存 metadata
    if isinstance(getattr(sample, "metadata", None), dict):
        sample.metadata["reward_detail"] = result
    return result["score"]
