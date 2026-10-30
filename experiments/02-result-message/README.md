# Experiment 02: result block in its own chat message (prompted agent)

**Question.** core-v1 pastes the `<result>` block inline, inside the model's open assistant
message. Does putting it in its own user message (close the assistant turn after `</search>`,
user message with the results, new assistant turn; Qwen's chat template) make the untrained
agent's format more valid, and can it replace the one-line reminder?

**Setup.** Prompted Qwen2.5-1.5B-Instruct, no LoRA, greedy, BM25 env, core-v1 prompt unchanged
(its few-shot example still shows results inline). First 200 train questions (where the prompt
was tuned). Option `SearchEnv(separate_results=True, reminder=False)`, CLI
`--separate_results --no_reminder`; both default off, so core-v1 is unchanged.
`checks/env_mask.py` scripts a policy through all four layouts: mask is 1 only on generated tokens,
and the separate layout equals the chat template rendering of the same messages.

```
python -m checks.env_mask
sbatch -o $SCRATCH/grpop/logs/%x-%j.out eval_prompted.sh   # or run its lines on an interactive GPU
```

## Results (200 train questions)

| layout | reminder | format valid | done | max_searches | malformed no_action | recall@9 | all_gold@9 | searches | eps with a repeated query | reward |
|---|---|---|---|---|---|---|---|---|---|---|
| inline (core-v1) | yes | 0.935 | 125 | 62 | 13 | 0.588 | 0.300 | 1.99 | 72 | 0.675 |
| inline | no | 0.460 | 86 | 6 | 108 | 0.570 | 0.280 | 1.24 | 0 | 0.600 |
| separate message | yes | **1.000** | 124 | 76 | 0 | **0.613** | **0.350** | 2.38 | 131 | **0.694** |
| separate message | no | 0.860 | 138 | 34 | 28 | 0.583 | 0.290 | 1.55 | 18 | 0.677 |

All 500 train questions, same ordering of settings: format 0.948 / 0.452 / 0.998 / 0.830,
recall@9 0.588 / 0.581 / 0.634 / 0.596, searches 1.96 / 1.24 / 2.37 / 1.49.

**Separate messages help format validity, most of all without the reminder** (0.46 -> 0.86):
in its own assistant turn the model mostly starts with an action instead of answering. The
reminder still matters (0.86 -> 1.00) and the two combine to 1.00 with zero malformed turns.

**But the gain is partly hollow.** With separate messages the model drops the "think one sentence"
part and emits bare actions, and with the reminder it repeats a query verbatim in 131/200 episodes
(inline: 72): e.g. `<search>2017 Afghanistan Olympics</search>` twice, then `<done/>`. Every
episode searches at least twice, yet unique passages per episode fall (4.18 vs 4.46 inline).
recall@9 rises only 0.025 (0.588 -> 0.613; all_gold 0.30 -> 0.35), far from GRPO run2 (0.915 on eval).
The remaining failure without the reminder is the same as inline: the model answers the question
after the first result.

Inline + reminder reproduces promptdev_v2 (0.94 format, 0.588 recall; one episode differs,
vLLM batching non-determinism) and inline without reminder reproduces promptdev_v1 (0.46).
Outputs: `$GRPOP_ROOT/outputs/resmsg_*` (200 q) and `resmsg500_*` (all 500).

## Results (200 eval questions)

| layout | reminder | format valid | done | max_searches | malformed no_action | recall@9 | all_gold@9 | searches | eps with a repeated query | reward |
|---|---|---|---|---|---|---|---|---|---|---|
| inline (core-v1) | yes | 0.945 | 118 | 71 | 11 | 0.623 | 0.335 | 2.09 | 78 | 0.707 |
| inline | no | 0.445 | 81 | 8 | 111 | 0.635 | 0.345 | 1.31 | 0 | 0.659 |
| separate message | yes | **1.000** | 123 | 77 | 0 | **0.675** | **0.420** | 2.37 | 123 | **0.757** |
| separate message | no | 0.800 | 133 | 27 | 40 | 0.635 | 0.350 | 1.49 | 15 | 0.721 |

Paired per question, 10k-sample bootstrap, separate minus inline:

| | reminder | Δ recall@9 [95% CI] | Δ all_gold@9 [95% CI] | questions better / worse (recall) |
|---|---|---|---|---|
| eval | yes | **+0.053 [+0.020, +0.085]** | **+0.085 [+0.025, +0.145]** | 34 / 13 |
| eval | no | +0.000 [-0.028, +0.028] | +0.005 [-0.040, +0.050] | 14 / 14 |
| train | yes | +0.025 [-0.005, +0.055] | +0.050 [-0.005, +0.105] | 24 / 14 |
| train | no | +0.013 [-0.015, +0.043] | +0.010 [-0.045, +0.065] | 20 / 15 |

**On eval, separate + reminder is a real, small search gain**: recall@9 0.623 -> 0.675 and
all_gold 0.335 -> 0.420, with the CI excluding zero, even though it gets fewer unique passages
(4.13 vs 4.48) and still repeats queries. Without the reminder it improves format only, not recall.
The inline + reminder eval row reproduces the core prompted baseline (0.620 / 0.945 in RESULTS.md).
Still far below single-shot BM25 (0.825) and GRPO run2 (0.915).

## Limitations

One greedy run per setting, 200 questions per split.
The few-shot example still shows the inline format, so the separate layout is slightly off-prompt;
a matched few-shot example is untested. User role only (Qwen's `tool` role not tried). Not used in
any training run.
