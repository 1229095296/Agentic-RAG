import asyncio
import unittest

from slime.utils.types import Sample

from slime_integration.generate_stateflow_direct import _parse_action, reward_func
from slime_integration.musique_reward import baseline_reward_view, compute_score


class DirectStateFlowGeneratorTest(unittest.TestCase):
    def _sample(self, answer="gold", protocol_valid=True):
        response = (
            "<search>q</search>"
            "<information>[d:s1] supporting text</information>"
            '<evidence_state>{"evidence":[{"span_id":"d:s1","text":"supporting text",'
            '"role":"direct"}],"open_questions":[],"decision":"answer"}</evidence_state>'
            f"<answer>{answer}</answer>"
        )
        sample = Sample(
            response=response,
            label={"ground_truth": {"target": ["gold"]}, "hop": 1},
        )
        sample.metadata = {
            "protocol_valid": protocol_valid,
            "gate_failure_code": "INVALID_STATE_JSON" if not protocol_valid else "",
            "trace": {
                "turns": [
                    {"type": "search", "query": "q"},
                    {"type": "evidence_state", "gate": 1},
                    {"type": "answer", "text": answer},
                ]
            },
        }
        return sample

    def test_qwen_template_closing_think_is_tolerated_for_search(self):
        self.assertEqual(_parse_action("</think>\n<search>q</search>"), ("search", "q"))

    def test_thinking_is_not_tolerated_before_evidence_state(self):
        action, _ = _parse_action(
            '<think>x</think><evidence_state>{"evidence":[],"open_questions":["x"],'
            '"decision":"search"}</evidence_state>'
        )
        self.assertIsNone(action)

    def test_protocol_failure_forces_zero_even_with_correct_answer(self):
        score = asyncio.run(reward_func(None, self._sample(protocol_valid=False)))
        self.assertEqual(score, 0.0)

    def test_valid_correct_trajectory_gets_shared_maximum(self):
        score = asyncio.run(reward_func(None, self._sample()))
        self.assertEqual(score, 1.0)

    def test_valid_wrong_trajectory_matches_baseline_reward(self):
        sample = self._sample(answer="wrong")
        actual = asyncio.run(reward_func(None, sample))
        expected = compute_score(
            baseline_reward_view(sample.response),
            ["gold"],
            label=sample.label,
            trace_turns=sample.metadata["trace"]["turns"],
            include_evidence=False,
            require_state_update=False,
        )["score"]
        self.assertEqual(actual, expected)
        self.assertEqual(sample.metadata["reward_detail"]["state"], 0.0)
        self.assertEqual(sample.metadata["reward_detail"]["path"], 0.0)
        self.assertEqual(sample.metadata["reward_detail"]["retrieval"], 0.0)

    def test_eval_is_binary_em(self):
        sample = self._sample()
        sample.metadata["_is_eval"] = True
        score = asyncio.run(reward_func(None, sample, evaluation=True))
        self.assertEqual(score, 1.0)
        self.assertEqual(sample.metadata["reward_detail"]["eval_0_1"], 1)

    def test_certified_answer_ready_replay_uses_shared_reward(self):
        sample = Sample(
            response="<answer>gold</answer>",
            label={"ground_truth": {"target": ["gold"]}, "hop": 1},
        )
        sample.metadata = {
            "is_replay": True,
            "protocol_valid": True,
            "trace": {"turns": [{"type": "answer", "text": "gold"}]},
        }
        score = asyncio.run(reward_func(None, sample))
        self.assertEqual(sample.metadata["reward_detail"]["gate"], 1.0)
        self.assertAlmostEqual(score, 0.905)


if __name__ == "__main__":
    unittest.main()
