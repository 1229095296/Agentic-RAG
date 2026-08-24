import random

from slime_integration.state_pool import StatePool


def _state(turn, signature, unresolved=True):
    return {
        "turn_index": turn,
        "E": [{"span_ids": [f"d:s{turn}"], "text": "evidence"}],
        "U": [{"text": "missing fact"}] if unresolved else [],
        "evidence_chain_signature": signature,
        "corpus_index_version": "test-v1",
    }


def test_pool_deduplicates_and_round_trips(tmp_path):
    pool = StatePool(str(tmp_path))
    added = pool.add_states(
        question_id="q1",
        states=[_state(1, "a"), _state(1, "a"), _state(2, "b", unresolved=False)],
        trajectory_id="traj",
        verifier_result=1.0,
        corpus_index_version="test-v1",
        index_version_current="test-v1",
        label={"ground_truth": {"target": ["answer"]}},
        question_text="question",
    )
    assert added == 2
    assert pool.num_states() == 2

    loaded = StatePool(str(tmp_path))
    loaded.load()
    assert loaded.num_states() == 2
    anchor = loaded.sample_anchor(random.Random(0))
    assert anchor["turn_index"] == 1


def test_adaptive_anchor_moves_earlier_after_high_success(tmp_path):
    pool = StatePool(str(tmp_path), adaptive_k=True, min_obs_for_move=2)
    pool.add_states(
        question_id="q1",
        states=[_state(1, "a"), _state(2, "b"), _state(3, "c")],
        trajectory_id="traj",
        verifier_result=1.0,
        corpus_index_version="test-v1",
        index_version_current="test-v1",
    )
    first = pool.sample_anchor(random.Random(0))
    assert first["turn_index"] == 2
    pool.update_suffix_success("q1", 2, 1.0)
    pool.update_suffix_success("q1", 2, 1.0)
    harder = pool.sample_anchor(random.Random(0))
    assert harder["turn_index"] == 1
