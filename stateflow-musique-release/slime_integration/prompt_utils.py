"""Small prompt/metadata helpers shared by StateFlow rollout components."""
from __future__ import annotations

import re
from typing import Any


_QUESTION_PREFIX_RE = re.compile(r"Question\s*:\s*", re.IGNORECASE)
_CHAT_BOUNDARY_RE = re.compile(
    r"(?:<\|im_end\|>|<\|im_start\|>assistant|<\|endoftext\|>)",
    re.IGNORECASE,
)


def extract_question_text(value: Any) -> str:
    """Return the raw user question, not the full SearchR1/chat prompt."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""

    matches = list(_QUESTION_PREFIX_RE.finditer(text))
    if matches:
        # Use the last Question: because few-shot preambles contain examples.
        candidate = text[matches[-1].end():]
        candidate = _CHAT_BOUNDARY_RE.split(candidate, maxsplit=1)[0]
        candidate = candidate.strip()
        if candidate:
            return candidate
    return text


def extract_question_id(metadata: dict | None = None, label: dict | None = None, default: str = "q") -> str:
    """Best-effort qid extraction across MuSiQue conversion variants."""
    metadata = metadata if isinstance(metadata, dict) else {}
    label = label if isinstance(label, dict) else {}
    for key in ("qid", "question_id", "id"):
        value = metadata.get(key)
        if value:
            return str(value)
    for key in ("qid", "question_id", "id"):
        value = label.get(key)
        if value:
            return str(value)
    return default
