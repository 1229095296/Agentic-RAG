# Algorithm

## Full-start rollout

1. The data source samples a MuSiQue question and creates one GRPO group with
   `n_samples_per_prompt` independent trajectories.
2. The policy emits one `<search>` action. The environment calls `/retrieve`,
   assigns stable sentence-level span identifiers, and injects the resulting
   `<information>` block with loss mask zero.
3. The policy emits the complete current `<evidence_state>`. It contains an
   extractive evidence list, concrete unresolved questions, and a decision.
4. The deterministic Gate checks that the output contains only the state action,
   follows the exact schema, cites visible spans, copies informative contiguous
   excerpts, uses evidence from the latest retrieval, and makes a decision
   consistent with unresolved questions. An accepted state replaces the prior
   model-visible state; the policy may retain useful earlier evidence or discard
   evidence it no longer needs.
5. A failed Gate ends the trajectory as a completed negative sample with reward
   zero. An accepted search state is recorded and followed by another search. An
   accepted answer-ready state permits one final `<answer>` action.
6. Training reward shares answer quality, executable action format, and search
   efficiency with the baseline. The protocol Gate is the rule-verifiable process
   constraint. Evaluation bypasses shaped terms and reports binary exact match.

## Harvest and replay

After a rollout batch, `harvest_states.process` considers only successful,
full-start trajectories above the configured reward threshold. It validates
provenance and leakage, then stores accepted turn-boundary states in append-only
JSONL files keyed by question identifier. Replay continuations are never
harvested back into the main pool.

`StateFlowDataSource` samples full-start and replay-start groups independently,
so GRPO normalization never mixes trajectories with different starting
difficulty. Replay prompts expose only the question and certified prefix state;
labels and future answers remain outside the prompt.

The default curriculum is:

- before step 50: full trajectories only, used to populate the pool;
- steps 50-499: replay is introduced and gradually reduced;
- step 500 onward: full trajectories only for full-task hardening.

For each replay anchor, the state pool maintains an exponential moving average
of suffix success. High success moves the anchor earlier, making the continuation
harder; low success moves it later, making the continuation easier.

## Baseline control

`generate_musique.py` uses the same retrieval endpoint, action boundaries,
answer matcher, format reward, efficiency term, sampling interface, and binary
evaluation. It omits evidence-state actions, the protocol Gate, state harvesting,
and replay. This keeps non-evidence behavior aligned while isolating StateFlow's
algorithmic contribution.
