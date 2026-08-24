import json
import unittest

from slime_integration.direct_evidence_state import parse_and_validate_state, state_from_anchor


VISIBLE = {
    "d1:s1": {"text": "Company X is headquartered in Springfield.", "source": {"file_name": "Company X"}},
    "d2:s1": {"text": "Springfield lies on the Green River.", "source": {"file_name": "Springfield"}},
}


def tagged(payload):
    return f"<evidence_state>{json.dumps(payload)}</evidence_state>"


class DirectEvidenceStateTest(unittest.TestCase):
    def test_valid_search_state(self):
        result = parse_and_validate_state(
            tagged(
                {
                    "evidence": [
                        {
                            "span_id": "d1:s1",
                            "text": "Company X is headquartered in Springfield.",
                            "role": "bridge",
                        }
                    ],
                    "open_questions": ["Which river runs by Springfield?"],
                    "decision": "search",
                }
            ),
            VISIBLE,
            require_current_span_ids={"d1:s1"},
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.state.decision, "search")

    def test_valid_answer_state_keeps_prior_and_current(self):
        result = parse_and_validate_state(
            tagged(
                {
                    "evidence": [
                        {"span_id": "d1:s1", "text": VISIBLE["d1:s1"]["text"], "role": "bridge"},
                        {"span_id": "d2:s1", "text": VISIBLE["d2:s1"]["text"], "role": "direct"},
                    ],
                    "open_questions": [],
                    "decision": "answer",
                }
            ),
            VISIBLE,
            require_current_span_ids={"d2:s1"},
        )
        self.assertTrue(result.ok)
        self.assertEqual(len(result.state.evidence), 2)

    def test_accepts_informative_verbatim_excerpt(self):
        payload = {
            "evidence": [
                {"span_id": "d1:s1", "text": "headquartered in Springfield", "role": "bridge"}
            ],
            "open_questions": ["Which river runs by Springfield?"],
            "decision": "search",
        }
        result = parse_and_validate_state(tagged(payload), VISIBLE)
        self.assertTrue(result.ok)
        self.assertEqual(result.state.evidence[0].text, "headquartered in Springfield")

    def test_rejects_malformed_json(self):
        result = parse_and_validate_state(
            '<evidence_state>{"evidence": [}</evidence_state>', VISIBLE
        )
        self.assertEqual(result.code, "INVALID_STATE_JSON")

    def test_rejects_extra_text_and_thinking(self):
        payload = {"evidence": [], "open_questions": ["x"], "decision": "search"}
        result = parse_and_validate_state(f"<think>x</think>{tagged(payload)}", VISIBLE)
        self.assertEqual(result.code, "STATE_EXTRA_TEXT")

    def test_rejects_extra_key(self):
        payload = {"evidence": [], "open_questions": ["x"], "decision": "search", "summary": "x"}
        result = parse_and_validate_state(tagged(payload), VISIBLE)
        self.assertEqual(result.code, "INVALID_STATE_SCHEMA")

    def test_rejects_hallucinated_span(self):
        payload = {
            "evidence": [{"span_id": "missing:s1", "text": "Invented.", "role": "direct"}],
            "open_questions": [],
            "decision": "answer",
        }
        result = parse_and_validate_state(tagged(payload), VISIBLE)
        self.assertEqual(result.code, "INVALID_PROVENANCE")

    def test_rejects_rewritten_evidence(self):
        payload = {
            "evidence": [{"span_id": "d1:s1", "text": "Springfield is the HQ.", "role": "bridge"}],
            "open_questions": ["Which river?"],
            "decision": "search",
        }
        result = parse_and_validate_state(tagged(payload), VISIBLE)
        self.assertEqual(result.code, "EVIDENCE_TEXT_MISMATCH")

    def test_rejects_uninformative_short_excerpt(self):
        payload = {
            "evidence": [{"span_id": "d1:s1", "text": "Springfield", "role": "bridge"}],
            "open_questions": ["Which river?"],
            "decision": "search",
        }
        result = parse_and_validate_state(tagged(payload), VISIBLE)
        self.assertEqual(result.code, "EVIDENCE_EXCERPT_TOO_SHORT")

    def test_rejects_text_copied_from_a_different_span(self):
        payload = {
            "evidence": [
                {"span_id": "d1:s1", "text": "lies on the Green River", "role": "direct"}
            ],
            "open_questions": [],
            "decision": "answer",
        }
        result = parse_and_validate_state(tagged(payload), VISIBLE)
        self.assertEqual(result.code, "EVIDENCE_TEXT_MISMATCH")

    def test_rejects_state_without_latest_retrieval(self):
        payload = {
            "evidence": [{"span_id": "d1:s1", "text": VISIBLE["d1:s1"]["text"], "role": "bridge"}],
            "open_questions": ["Which river?"],
            "decision": "search",
        }
        result = parse_and_validate_state(tagged(payload), VISIBLE, require_current_span_ids={"d2:s1"})
        self.assertEqual(result.code, "NO_CURRENT_EVIDENCE")

    def test_rejects_inconsistent_decisions(self):
        search = parse_and_validate_state(
            tagged({"evidence": [], "open_questions": [], "decision": "search"}), VISIBLE
        )
        answer = parse_and_validate_state(
            tagged({"evidence": [], "open_questions": ["x"], "decision": "answer"}), VISIBLE
        )
        self.assertEqual(search.code, "INCONSISTENT_DECISION")
        self.assertEqual(answer.code, "INCONSISTENT_DECISION")

    def test_anchor_conversion_is_extractively_lossless(self):
        anchor = {
            "E": [
                {
                    "span_ids": ["d1:s1"],
                    "text": VISIBLE["d1:s1"]["text"],
                    "role": "bridge",
                }
            ],
            "U": [{"text": "Which river?"}],
        }
        state = state_from_anchor(anchor)
        self.assertEqual(state.evidence[0].text, VISIBLE["d1:s1"]["text"])
        self.assertEqual(state.open_questions, ["Which river?"])
        self.assertEqual(state.decision, "search")


if __name__ == "__main__":
    unittest.main()
