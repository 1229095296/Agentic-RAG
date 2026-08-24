# StateFlow-GRPO for MuSiQue

This repository contains the core implementation of StateFlow-GRPO used for
multi-hop retrieval training on MuSiQue. It is intentionally limited to the
algorithm, the Search-R1-style baseline needed for comparison, data conversion,
and unit tests. Cluster launch scripts, checkpoints, logs, indexes, credentials,
and machine-specific paths are not included.

## Method

StateFlow adds a persistent, model-authored evidence state to an agentic RAG
trajectory. After every retrieval, the policy must emit exactly one structured
`<evidence_state>` containing extractive evidence spans, unresolved questions,
and a search-or-answer decision. A deterministic Gate validates schema,
provenance, extractiveness, current-turn evidence use, and decision consistency.
Invalid states terminate the trajectory with zero reward; accepted states are
preserved at turn boundaries.

Successful full trajectories contribute their intermediate states to an
append-only state pool. Later GRPO groups can start from a certified intermediate
state instead of always starting from the original question. Full-start and
replay-start samples remain separate GRPO groups, while an adaptive curriculum
uses suffix success rates to move replay anchors earlier or later.

See [docs/ALGORITHM.md](docs/ALGORITHM.md) for the complete execution flow.

## Layout

```text
slime_integration/
  generate_stateflow_direct.py  # StateFlow rollout, Gate, and reward entrypoint
  direct_evidence_state.py      # model-authored state schema and validation
  evidence_state.py             # replay/pool state representation
  stateflow_data_source.py      # full-start/replay-start GRPO sampling
  state_pool.py                 # persistent state pool and adaptive anchors
  harvest_states.py             # harvest successful intermediate states
  musique_reward.py             # shared answer/format/efficiency reward utilities
  generate_musique.py           # Search-R1-style GRPO baseline
  musique_to_slime.py           # MuSiQue-to-Slime data conversion
configs/stateflow.env.example    # portable runtime configuration
tests/                           # logic and reward regression tests
```

## Requirements

- Python 3.10+
- `aiohttp`
- `pytest` for tests
- A compatible Slime installation for rollout/training
- An HTTP retriever implementing `POST /retrieve`

The retriever request is:

```json
{"queries": ["query"], "topk": 3, "return_scores": false}
```

Each returned document should contain `spans`, where every span has a stable
`span_id` and source text. If spans are absent, the rollout performs a
deterministic sentence split as a fallback.

## Prepare MuSiQue

Download MuSiQue separately and convert it without committing the dataset:

```bash
python -m slime_integration.musique_to_slime \
  --input /path/to/MuSiQue.json \
  --output data/musique_train.jsonl
```

Use `--limit N` for a smoke dataset or `--hop 2` to select one hop count.

## Configure

```bash
cp configs/stateflow.env.example .env
set -a
source .env
set +a
```

The Slime hooks are:

```text
--custom-generate-function-path slime_integration.generate_stateflow_direct.generate
--custom-rm-path slime_integration.generate_stateflow_direct.reward_func
--data-source-path slime_integration.stateflow_data_source.StateFlowDataSource
--rollout-all-samples-process-path slime_integration.harvest_states.process
```

For the baseline, use:

```text
--custom-generate-function-path slime_integration.generate_musique.generate
--custom-rm-path slime_integration.musique_reward.reward_func
```

All model, Slime, Megatron, dataset, output, and cluster settings should be
provided by the caller's training launcher. This repository does not assume a
specific filesystem or scheduler.

## Tests

Pure state and reward tests do not require a running retriever or model server:

```bash
python -m pytest -q
```

## Reproducibility notes

- Evaluation returns binary exact match (`0/1`), not shaped training reward.
- Replay begins at step 50 by default and is disabled again after step 500.
- Traces and state-pool records may contain dataset text and must remain outside
  version control.
- No API credentials are required by the core implementation. Configure any
  external retriever credentials only in your runtime environment.
