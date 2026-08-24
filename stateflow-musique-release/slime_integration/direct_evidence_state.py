"""Strict model-authored evidence state protocol for StateFlow V2.

The model writes the complete evidence state. Rules validate it but never
repair, summarize, or backfill model output.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


MAX_EVIDENCE_ITEMS = 6
MAX_OPEN_QUESTIONS = 4
MAX_STATE_CHARS = 5000
MIN_EVIDENCE_EXCERPT_CHARS = 12
MIN_EVIDENCE_EXCERPT_WORDS = 3
VALID_ROLES = {"bridge", "direct"}
VALID_DECISIONS = {"search", "answer"}

_STATE_RE = re.compile(r"<evidence_state>(.*?)</evidence_state>", re.DOTALL)


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_verbatim_excerpt(item_text: str, source_text: str) -> tuple[bool, bool]:
    """Return whether text is grounded and whether a matching excerpt is too short."""
    item = _norm_text(item_text)
    source = _norm_text(source_text)
    if not item or not source:
        return False, False
    if item == source:
        return True, False
    if item not in source:
        return False, False

    word_count = len(re.findall(r"\w+", item, flags=re.UNICODE))
    too_short = len(item) < MIN_EVIDENCE_EXCERPT_CHARS or word_count < MIN_EVIDENCE_EXCERPT_WORDS
    return not too_short, too_short


@dataclass(frozen=True)
class EvidenceItem:
    span_id: str
    text: str
    role: str

    def to_dict(self) -> dict[str, str]:
        return {"span_id": self.span_id, "text": self.text, "role": self.role}


@dataclass
class DirectEvidenceState:
    evidence: list[EvidenceItem] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    decision: str = "search"

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": [item.to_dict() for item in self.evidence],
            "open_questions": list(self.open_questions),
            "decision": self.decision,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class StateValidation:
    ok: bool
    code: str = ""
    message: str = ""
    state: DirectEvidenceState | None = None


def parse_and_validate_state(
    tagged_text: str,
    available_spans: dict[str, dict[str, Any]],
    *,
    require_current_span_ids: set[str] | None = None,
) -> StateValidation:
    """Validate one exact model-authored ``<evidence_state>`` action.

    ``available_spans`` contains current retrieval spans plus evidence already
    present in the accepted state. Evidence text must be either the complete
    source span or a sufficiently informative contiguous verbatim excerpt after
    whitespace normalization. No field is repaired or dropped.
    """
    text = str(tagged_text or "").strip()
    if len(text) > MAX_STATE_CHARS:
        return StateValidation(False, "STATE_TOO_LONG", "evidence_state exceeds the character limit")

    matches = list(_STATE_RE.finditer(text))
    if len(matches) != 1:
        return StateValidation(False, "INVALID_STATE_TAGS", "expected exactly one evidence_state tag pair")
    if text[: matches[0].start()].strip() or text[matches[0].end() :].strip():
        return StateValidation(False, "STATE_EXTRA_TEXT", "text outside evidence_state is not allowed")

    try:
        payload = json.loads(matches[0].group(1).strip())
    except json.JSONDecodeError as exc:
        return StateValidation(False, "INVALID_STATE_JSON", f"invalid JSON: {exc.msg}")
    if not isinstance(payload, dict):
        return StateValidation(False, "INVALID_STATE_SCHEMA", "state must be a JSON object")

    required_keys = {"evidence", "open_questions", "decision"}
    if set(payload) != required_keys:
        return StateValidation(
            False,
            "INVALID_STATE_SCHEMA",
            f"state keys must be exactly {sorted(required_keys)}",
        )

    raw_evidence = payload["evidence"]
    raw_open = payload["open_questions"]
    decision = payload["decision"]
    if not isinstance(raw_evidence, list) or not isinstance(raw_open, list):
        return StateValidation(False, "INVALID_STATE_SCHEMA", "evidence and open_questions must be lists")
    if not isinstance(decision, str) or decision not in VALID_DECISIONS:
        return StateValidation(False, "INVALID_STATE_SCHEMA", "decision must be search or answer")
    if len(raw_evidence) > MAX_EVIDENCE_ITEMS:
        return StateValidation(False, "TOO_MUCH_EVIDENCE", "too many evidence items")
    if len(raw_open) > MAX_OPEN_QUESTIONS:
        return StateValidation(False, "TOO_MANY_OPEN_QUESTIONS", "too many open questions")

    evidence: list[EvidenceItem] = []
    seen: set[str] = set()
    for raw in raw_evidence:
        if not isinstance(raw, dict) or set(raw) != {"span_id", "text", "role"}:
            return StateValidation(
                False,
                "INVALID_EVIDENCE_SCHEMA",
                "each evidence item must contain exactly span_id, text, and role",
            )
        span_id, item_text, role = raw["span_id"], raw["text"], raw["role"]
        if not all(isinstance(value, str) for value in (span_id, item_text, role)):
            return StateValidation(False, "INVALID_EVIDENCE_SCHEMA", "evidence fields must be strings")
        if not span_id or span_id in seen:
            return StateValidation(False, "INVALID_EVIDENCE_SCHEMA", "span_id must be non-empty and unique")
        if role not in VALID_ROLES:
            return StateValidation(False, "INVALID_EVIDENCE_SCHEMA", "role must be bridge or direct")
        source = available_spans.get(span_id)
        if not isinstance(source, dict):
            return StateValidation(False, "INVALID_PROVENANCE", f"span_id is not visible: {span_id}")
        source_text = str(source.get("text") or "")
        grounded, too_short = _is_verbatim_excerpt(item_text, source_text)
        if too_short:
            return StateValidation(
                False,
                "EVIDENCE_EXCERPT_TOO_SHORT",
                f"text excerpt from span {span_id} is too short",
            )
        if not grounded:
            return StateValidation(
                False,
                "EVIDENCE_TEXT_MISMATCH",
                f"text is not a verbatim excerpt from span {span_id}",
            )
        evidence.append(EvidenceItem(span_id=span_id, text=item_text.strip(), role=role))
        seen.add(span_id)

    current_ids = set(require_current_span_ids or ())
    if current_ids and not (seen & current_ids):
        return StateValidation(False, "NO_CURRENT_EVIDENCE", "state must select evidence from the latest retrieval")

    open_questions: list[str] = []
    for item in raw_open:
        if not isinstance(item, str) or not _norm_text(item) or len(item) > 200:
            return StateValidation(False, "INVALID_OPEN_QUESTION", "open questions must be non-empty strings <= 200 chars")
        open_questions.append(_norm_text(item))

    if decision == "search" and not open_questions:
        return StateValidation(False, "INCONSISTENT_DECISION", "search requires at least one open question")
    if decision == "answer" and open_questions:
        return StateValidation(False, "INCONSISTENT_DECISION", "answer requires no open questions")
    if decision == "answer" and not evidence:
        return StateValidation(False, "INCONSISTENT_DECISION", "answer requires evidence")

    return StateValidation(
        True,
        state=DirectEvidenceState(evidence=evidence, open_questions=open_questions, decision=decision),
    )


def state_from_anchor(anchor_state: dict[str, Any] | None) -> DirectEvidenceState:
    """Convert a certified V1 pool state into the V2 model-visible state."""
    evidence = []
    for cap in (anchor_state or {}).get("E", []) or []:
        if not isinstance(cap, dict):
            continue
        span_ids = [sid for sid in (cap.get("span_ids") or []) if isinstance(sid, str)]
        if len(span_ids) != 1 or not cap.get("text"):
            continue
        role = cap.get("role") if cap.get("role") in VALID_ROLES else "direct"
        evidence.append(EvidenceItem(span_ids[0], str(cap["text"]), role))
    open_questions = []
    for unresolved in (anchor_state or {}).get("U", []) or []:
        text = unresolved.get("text") if isinstance(unresolved, dict) else None
        if text:
            open_questions.append(_norm_text(text)[:200])
    return DirectEvidenceState(
        evidence=evidence[:MAX_EVIDENCE_ITEMS],
        open_questions=open_questions[:MAX_OPEN_QUESTIONS],
        decision="search" if open_questions else "answer",
    )
