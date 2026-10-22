"""确定性句子切分工具（S3.0）。

用于把检索文档切成稳定、可重复的句子级 span——这是 StateFlow 抽取式证据状态的地基：
同一篇文档、同一句话必须永远得到同一个句子序号，才能让"模型选 span_id、规则回填原文"稳定可回放。

纯正则实现，无外部依赖，保证确定性（同输入永远同输出）。对 V1（HotpotQA/2Wiki/MuSiQue 维基文本）够用；
后续若需要更精细的切分，可替换本函数，但必须保持确定性。
"""
import re

# 在句末标点(.!?)之后、遇到空白 + (大写字母/引号/左括号/数字) 处切分。
# 使用 lookbehind/lookahead 保证不吞字符。
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(\[])')


def split_sentences(text: str) -> list[str]:
    """把一段文本确定性地切成句子列表（去空、strip）。"""
    if not text:
        return []
    # 统一换行为空格，避免换行影响边界判定；折叠多空白
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = _SENT_BOUNDARY.split(normalized)
    return [p.strip() for p in parts if p.strip()]


def make_spans(doc_id: str, text: str) -> list[dict]:
    """把文档文本切成带稳定 span_id 的句子列表。

    span_id 形如 '{doc_id}:s{N}'（N 从 1 开始，doc 内递增）。
    返回: [{"span_id": "d17:s1", "text": "..."}, ...]
    """
    spans = []
    for i, sent in enumerate(split_sentences(text), start=1):
        spans.append({"span_id": f"{doc_id}:s{i}", "text": sent})
    return spans
