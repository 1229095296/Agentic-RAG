"""把 MuSiQue.json 转换为 slime 训练/评测所需的数据格式（jsonl）。

slime 数据字段约定（见 slime/utils/data.py 与 examples/search-r1）：
- `--input-key`  -> Sample.prompt   （本脚本用 "prompt"，已内嵌 search-r1 风格的指令模板）
- `--label-key`  -> Sample.label    （本脚本用 "reward_model"，结构对齐 search-r1 的 reward_func）

每条输出：
{
  "prompt": "<search-r1 指令模板 + Question: ...>",
  "reward_model": {
      "ground_truth": {"target": [reference_answer, *answer_aliases]},  # reward 用
      "evidence_file_names": ["FC Barcelona 1", ...],                    # 检索指标用（gold 证据文件名）
      "hop": 2,                                                          # 课程/分组统计用
      "qid": "2hop__13548_13529"
  },
  "source": "musique"
}

用法（在 rag-agent 项目根目录下）：
  python -m slime_integration.musique_to_slime --input MuSiQue.json --output musique_slime.jsonl
可选：
  --limit N        只转换前 N 条（用于快速冒烟）
  --hop 2          只保留指定 hop（如只取 2-hop 子集）

注意：按项目规范，MuSiQue 1000 题为冻结测试集，调参/训练应另用开发集。
本脚本只做格式转换，不自动划分 train/dev；训练数据来源需另行决定。
"""
import argparse
import json
import os
import re
import sys

# search-r1 官方使用的指令模板（保证 slime 侧 generate_with_search 的 <think>/<search>/<answer> 解析可直接工作）
PROMPT_TEMPLATE = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack some "
    "knowledge, you can call a search engine by <search> query </search> and it will return "
    "the top searched results between <information> and </information>. You can search as many "
    "times as you want. If you find no further external knowledge needed, you can directly "
    "provide the answer inside <answer> and </answer>, without detailed illustrations. For "
    "example, <answer> Beijing </answer>. Question: {question}\n"
)

_HOP_RE = re.compile(r"^(\d+)hop", re.IGNORECASE)


def parse_hop(qid: str) -> int | None:
    """从 MuSiQue id（如 '2hop__13548_13529' / '3hop1__...' / '4hop3__...'）解析跳数。"""
    if not qid:
        return None
    m = _HOP_RE.match(str(qid))
    return int(m.group(1)) if m else None


def build_targets(record: dict) -> list:
    """构造答案匹配目标：reference_answer + answer_aliases（去重、保序、去空）。"""
    targets = []
    ref = record.get("reference_answer")
    if ref:
        targets.append(ref)
    for alias in record.get("answer_aliases", []) or []:
        if alias and alias not in targets:
            targets.append(alias)
    return targets


def build_evidence_file_names(record: dict) -> list:
    """收集 gold 证据文件名（去重、保序），用于后续 Recall/证据召回指标。"""
    names = []
    for ev in record.get("evidence", []) or []:
        fn = ev.get("file_name")
        if fn and fn not in names:
            names.append(fn)
    return names


def convert(input_path: str, output_path: str, limit: int | None = None, hop: int | None = None):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    written = 0
    hop_dist: dict = {}
    skipped_no_answer = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for record in data:
            qid = record.get("id", "")
            rec_hop = parse_hop(qid)

            if hop is not None and rec_hop != hop:
                continue

            question = record.get("query", "")
            targets = build_targets(record)
            if not question or not targets:
                skipped_no_answer += 1
                continue

            sample = {
                "prompt": PROMPT_TEMPLATE.format(question=question),
                "reward_model": {
                    "ground_truth": {"target": targets},
                    "evidence_file_names": build_evidence_file_names(record),
                    "hop": rec_hop,
                    "qid": qid,
                },
                "source": "musique",
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
            hop_dist[rec_hop] = hop_dist.get(rec_hop, 0) + 1

            if limit is not None and written >= limit:
                break

    print(f"[musique_to_slime] wrote {written} samples -> {output_path}", flush=True)
    print(f"[musique_to_slime] hop distribution: "
          f"{dict(sorted(hop_dist.items(), key=lambda x: (x[0] is None, x[0])))}", flush=True)
    if skipped_no_answer:
        print(f"[musique_to_slime] skipped {skipped_no_answer} records (missing question/answer)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="MuSiQue.json -> slime jsonl 转换器")
    parser.add_argument("--input", type=str, default="MuSiQue.json")
    parser.add_argument("--output", type=str, default="musique_slime.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="只转换前 N 条（冒烟用）")
    parser.add_argument("--hop", type=int, default=None, help="只保留指定 hop（如 2 表示只取 2-hop）")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[musique_to_slime] input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    convert(args.input, args.output, limit=args.limit, hop=args.hop)


if __name__ == "__main__":
    main()
