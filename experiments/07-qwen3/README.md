# Experiment 07: the run2 recipe on Qwen3-1.7B, does it teach search or only the harness?

**Status: run 2026-11-06.** Day-0 job 27581724 (es_debug, 1 GPU); training jobs 27582368 (seed 0,
1:31 wall) and 27582369 (seed 1, 1:13 wall), es_normal, 4 GPUs each; evals 27583980 / 27584676. The
rules below were fixed before any job was submitted and were not changed once numbers were in.

**Verdict: unstable at this scale (the seeds split), decided by Rule 2.** Both seeds pass Rule 1: step
150 beats the format-fixed prompted baseline (0.678 recall@9) by +0.203 [+0.160, +0.245] (seed 0) and
+0.217 [+0.175, +0.260] (seed 1), with all_gold@9 up by +0.35 / +0.37, so H2 (the gain is at most a
format effect) is false on both seeds. Rule 2 splits: in Exp 06's harness T seed 1 keeps the gain
(−0.037 [−0.070, −0.005], as run2 did) and seed 0 loses about half of it (−0.083 [−0.125, −0.040], CI
entirely below −0.03). The seed gap in core is 0.015, so no third seed was run.

**Question.** Qwen3 models carry a lot more RL post-training than Qwen2.5. The claim to test, raised
by a reader of the repo: on such a model, GRPO with our recipe mostly teaches the harness (emit the
tags, keep the turn shape), not search. Run2's gain on Qwen2.5-1.5B is 0.620 → 0.915 recall@9; Exp 02
bounds the part of that which format alone can explain at 0.055 (prompted model in the Exp 02 layout:
format validity 1.000, recall@9 0.675). Does the same recipe, unchanged, produce a gain of the same
kind on Qwen3-1.7B?

**Hypothesis H2 (the claim).** On Qwen3-1.7B, trained minus format-fixed prompted is at most a format
effect: a paired Δ recall@9 under +0.03, or with a CI containing zero.

**Falsification rules, fixed before running.**

- **Format-fixed baseline** = prompted Qwen3-1.7B evaluated in core **and** in the Exp 02 layout
  (`--separate_results`, reminder on); the higher recall@9 of the two is the baseline. This is the
  model's best shot at the format without training, as 0.675 is for Qwen2.5.
- **Rule 1, gain beyond format.** Trained Qwen3 (step 150, core harness) minus the baseline: paired
  Δ recall@9 over the 200 eval questions, 10k bootstrap, 95% CI. Passes if Δ > +0.03 and the CI
  excludes zero, **and** the query behaviour moved (searches per episode, all_gold@9, recall@3 per
  query), so that the gain is not a tag story.
- **Rule 2, transfer.** The trained adapter evaluated in Exp 06's harness T passes Exp 06's rule: paired
  drop vs its own core eval has a CI containing a value ≥ −0.03, and recall@9 in T stays above the
  format-fixed baseline. Format validity under 0.90 in T is reported as "could not act".
- **Verdict.** The recipe teaches search on Qwen3 only if **every seed** passes both rules. H2 holds
  if every seed fails at least one. A split across seeds is reported as "unstable at this scale", which
  is also an answer, not averaged away.
- **Seeds.** Two training seeds (`--seed 0`, `--seed 1`). If |recall@9(seed 0) − recall@9(seed 1)| at
  step 150 in core is **greater than 0.03 (absolute)**, run seed 2 and report all three as a range. The
  adapter used everywhere is **step 150**, never the best checkpoint by eval recall (that would be
  selection on the test set). Note: `--seed` fixes the prompt order and the trainer's RNG; vLLM
  sampling during training is unseeded (`seed=None` in `grpo.py`), so two runs with the same seed would
  not be identical either.
- **Side prediction, reported either way.** If the prompted Qwen3 baseline is already ≥ 0.85, the
  size of run2's gain on Qwen2.5 came largely from Qwen2.5's weak start.

## Design

**Model.** `Qwen/Qwen3-1.7B` (the post-trained chat model; closest in size to Qwen2.5-1.5B, same
151,936-token vocabulary, so the logits memory cost is the same). Fallback if it does not fit the
trainer GPU with the chunked fp32 log-probs: `Qwen/Qwen3-0.6B`, recorded as a hardware limitation.

**Thinking mode: off.** Build prompts with `apply_chat_template(..., enable_thinking=False)` so the
turn-token budget (256) goes to the action, not to reasoning. This is the one place the harness differs
from Qwen2.5's and it has to, Qwen2.5 has no thinking mode. Log how many turns are cut by the token
cap before an action (`query_too_long`/`no_action` with 256 generated tokens) and how many turns
contain `<think>` anyway.

**Recipe: run2, unchanged.** `--lr 5e-5 --w_step 0`, LoRA r=16, G=8, 16 prompts per step, 150 steps,
500 train questions (`train.jsonl`), core harness (inline results, reminder on), core reward. Nothing
is re-tuned for Qwen3; see Limitations.

**Day-0 checks, before the real runs (es_debug, 1 GPU, capped at 1 h).**
1. `serve.sh 8000 Qwen/Qwen3-1.7B` with the sm_75 flags answers one request with LoRA enabled.
2. One trainer step on one GPU with the real batch (G=8, 16 prompts): fits in 10.5 GiB, log-probs in
   fp32 have no NaN/inf (Qwen3 was trained in bf16, we run fp16 weights).
3. Step-0 variance check as in Exp 04 (`experiments/04-summary-judge/base_groups.py` pattern): share of
   train groups with nonzero reward variance under the core reward. Below 5% means GRPO has no signal
   on this model; stop and report that instead of training.
4. Prompted eval in core (run 1 below) comes out of the same job.

**Runs.**

| # | what | harness | notes |
|---|---|---|---|
| 1 | prompted Qwen3-1.7B, eval | core | raw start |
| 2 | prompted Qwen3-1.7B, eval | Exp 02 layout | format-fixed baseline candidate |
| 3 | GRPO seed 0, `q3_run2_s0` | core | full node, es_normal, ~2 h |
| 4 | GRPO seed 1, `q3_run2_s1` | core | parallel with 3 if nodes are free |
| 5, 6 | step0150 of each seed, eval | core | Rule 1 |
| 7, 8 | step0150 of each seed, eval | Exp 06 harness T | Rule 2 |
| 9+ | seed 2, if the seed rule fires | | repeats 3, 5, 7 |

Out names `exp07_<prompted|s0|s1|s2>_<core|sep|T>`. Training greedy evals and curves as in Exp 01.

**Metrics.** recall@9, all_gold@9, nDCG@9, recall@3 per query, searches per episode, format validity,
ends done / max / malformed, truncated turns, `<think>` turns, tokens generated; training curves
(train recall@9, searches, format, zero-variance share) for each seed. Paired bootstraps from
`experiments/04-summary-judge/analyze.py`.

## Implementation

- `nanosearch/grpo.py` and `nanosearch/agent.py`: `--model` (default `Qwen/Qwen2.5-1.5B-Instruct`,
  recorded in the run config and eval summary); `--no_thinking` or auto when the tokenizer's chat
  template accepts `enable_thinking`. `SearchEnv.prompt_ids` passes `enable_thinking=False` when
  asked. Defaults byte-identical to core-v1 for Qwen2.5 (assert in the job, as Exp 05/06 do).
- `experiments/07-qwen3/run.sh`: `experiments/01-hotpotqa-grpo/run.sh` with the model passed to
  `serve.sh` and to the trainer. `evals.sh` on the Exp 05/06 pattern, one vLLM server per model.
- `experiments/07-qwen3/analyze.py` → `out/{eval.json, curves.png, examples.txt}`: the table, the
  two rules per seed, the seed-gap check, three verbatim episodes per seed (success, miss, one in T).

## Run

```
sbatch -o $SCRATCH/grpop/logs/%x-%j.out experiments/07-qwen3/day0.sh            # checks 1-4
sbatch -o $SCRATCH/grpop/logs/%x-%j.out experiments/07-qwen3/run.sh q3_run2_s0 --lr 5e-5 --w_step 0 --seed 0
sbatch -o $SCRATCH/grpop/logs/%x-%j.out experiments/07-qwen3/run.sh q3_run2_s1 --lr 5e-5 --w_step 0 --seed 1
sbatch -o $SCRATCH/grpop/logs/%x-%j.out experiments/07-qwen3/evals.sh
PYTHONPATH=. python experiments/07-qwen3/analyze.py
```

## Results

All evals: 200 eval questions, greedy, core-v1 corpus and BM25; paired Δ = 10k bootstrap, 95 % CI.
Raw numbers in `out/eval.json`, curves in `out/curves.png`, episodes in `out/examples.txt`.

### Day-0 checks (job 27581724, 13 min)

1. vLLM 0.19 serves `Qwen/Qwen3-1.7B` with the sm_75 flags (up in ~2.5 min, KV cache 4.83 GiB) and
   loads a LoRA at runtime; the step-1 adapter answered 16 eval questions through the agent.
2. One trainer step on 32 kept episodes (1,060 model tokens, max sequence 1,310): peak 3.85 GiB (run2
   on Qwen2.5: 3.69), every fp32 log-prob finite (min −7.45, mean −0.14), parameters finite after the
   step. No fallback to 0.6B needed.
3. Step-0 variance: 500 train questions × G=8 at temperature 1, core reward with w_step 0: **32 % of
   groups have nonzero reward variance** (Qwen2.5, Exp 04: 97 %). Above the 5 % stop rule, so the runs
   went ahead, but two thirds of every batch is filtered from the start. Sampled recall@9 0.647
   against 0.678 greedy: Qwen3's temperature-1 samples are close to its greedy output.
4. Prompt checks: Qwen2.5 prompts byte-identical to core-v1 (no template kwarg); the Qwen3 generation
   prompt ends in `<think>\n\n</think>\n\n`; `<think>` is not a special token, so `think_turns`
   would see it. It never appeared: 0 `<think>` turns and 0 truncated turns in every eval and every
   training step of both seeds.

### Eval table

| system | recall@9 | all_gold@9 | nDCG@9 | recall@3 (1st query) | searches | format | tokens / ep | ends |
|---|---|---|---|---|---|---|---|---|
| prompted Qwen3, core (**baseline**) | 0.678 | 0.450 | 0.652 | 0.568 | 2.00 | 1.000 | 31 | done 199 |
| prompted Qwen3, Exp 02 layout | 0.658 | 0.420 | 0.641 | 0.568 | 1.51 | 1.000 | 25 | done 199 |
| seed 0 step 150, core | 0.880 | 0.795 | 0.751 | 0.517 | 3.00 | 1.000 | 58 | max 200 |
| seed 0 step 150, harness T | 0.798 | 0.675 | 0.687 | 0.512 | 2.97 | 1.000 | 103 | max 194, done 6 |
| seed 1 step 150, core | 0.895 | 0.820 | 0.760 | 0.522 | 3.00 | 1.000 | 41 | max 200 |
| seed 1 step 150, harness T | 0.858 | 0.755 | 0.738 | 0.520 | 2.99 | 0.995 | 48 | max 198, done 1, malformed 1 |
| Qwen2.5 prompted, core (Exp 01) | 0.620 | 0.330 | 0.624 | 0.525 | 2.08 | 0.945 | 42 | done 118, max 71, malformed 11 |
| Qwen2.5 run2 step 150, core (Exp 01) | 0.915 | 0.850 | 0.759 | 0.490 | 2.97 | 1.000 | 40 | max 194, done 6 |

The format-fixed baseline is the core layout (the Exp 02 layout is 0.020 [−0.040, 0.000] lower, with
fewer searches). Side prediction: the baseline is 0.678, not ≥ 0.85, so it does not fire; prompted
Qwen3 starts where Qwen2.5's *format-fixed* baseline is (0.675, Exp 02) and already has format
validity 1.000 untrained (Qwen2.5: 0.945).

### Rule 1, gain beyond format (step 150 in core minus the baseline)

| seed | Δ recall@9 | Δ all_gold@9 | Δ recall@3 (1st query) | Δ searches | pass |
|---|---|---|---|---|---|
| 0 | +0.203 [+0.160, +0.245] | +0.345 [+0.270, +0.420] | −0.050 [−0.090, −0.013] | +0.995 | yes |
| 1 | +0.217 [+0.175, +0.260] | +0.370 [+0.295, +0.445] | −0.045 [−0.083, −0.010] | +0.995 | yes |

"Query behaviour moved" was read as: the paired CI of all_gold@9 or of recall@3 excludes zero (it
does for both, on both seeds). Both seeds pass. The gain is 0.20–0.22 against run2's 0.295 on Qwen2.5,
from a higher start to a lower end (0.880 / 0.895 vs 0.915).

### Rule 2, transfer (step 150 in harness T minus its own core eval)

| seed | Δ recall@9 | Δ all_gold@9 | Δ nDCG@9 | T recall@9 > baseline | format in T | pass |
|---|---|---|---|---|---|---|
| 0 | −0.083 [−0.125, −0.040] | −0.120 [−0.185, −0.055] | −0.064 [−0.098, −0.032] | 0.798 > 0.678 | 1.000 | **no** (CI below −0.03) |
| 1 | −0.037 [−0.070, −0.005] | −0.065 [−0.120, −0.010] | −0.022 [−0.048, +0.002] | 0.858 > 0.678 | 0.995 | yes |

Neither seed "could not act" (format ≥ 0.995 in T) and neither wrote an old token in T (0 of ~1,200
turns), so the renamed tags are not what seed 0 loses on. For reference, run2 (Qwen2.5) lost −0.035
[−0.062, −0.010] in the same harness (Exp 06).

**Seed rule.** |0.880 − 0.895| = 0.015 (paired −0.015 [−0.043, +0.013]), under 0.03: no seed 2.

### Training (curves in `out/curves.png`, run2 in grey)

| run | train recall@9 steps 0–9 → 125–149 | searches 0–9 → 125–149 | groups filtered (mean) | steps without update | wall |
|---|---|---|---|---|---|
| run2, Qwen2.5 | 0.65 → 0.905 | 2.59 → 2.99 | 0.67 (0.18 at the start, 0.80 at the end) | 2 | 1:40 |
| seed 0, Qwen3 | 0.77 → 0.89 | 2.62 → 3.00 | 0.79 (0.68 → 0.84) | 4 | 1:31 |
| seed 1, Qwen3 | 0.71 → 0.89 | 2.00 → 3.00 | 0.83 (0.70 → 0.85) | 4 | 1:13 |

Format validity stays ≥ 0.99 on both seeds for the whole run (run2 climbed from 0.89). Seed 0 reached
3 searches per episode by step ~20, seed 1 only by step ~40 and dipped to 2.4 around step 90 before
recovering. Qwen3 trains on a fifth of each batch for the whole run; Qwen2.5 trained on most of it
early on. Both end at the same train recall as run2 within noise.

### What the trained Qwen3 does

- **Never stops.** Prompted Qwen3 ends 199/200 episodes with `<done/>` after 2 searches; both trained
  seeds end 200/200 at the search cap in core (run2: 194/200). With w_step 0 the third search is free.
- **Splits the question.** all_gold@9 rises by 0.35, but recall@3 of the *first* query falls by 0.05 on
  both seeds: the trained model opens with one entity and leaves the rest to later queries (prompted:
  `<search>Maria Golovin English language opera composer</search>` twice, then done, recall 0.5;
  trained seed 1: `Italian-American composer and librettist`, `Maria Golovin English language opera`,
  `Maria Golovin composer`, recall 1.0). Repeated identical queries drop from 102/200 episodes
  (prompted) to 3/200 (seed 0) and 17/200 (seed 1).
- **The seeds differ in style, and the style decides Rule 2.** Seed 1 writes the bare action in 600/600
  core turns (median 13 tokens). Seed 0 opens a third of its core turns with a "First I need …"
  sentence (58 tokens per episode). Harness T's prompt asks for one or two sentences per turn: seed 1
  ignores it (bare in 538/599 turns, 48 tokens per episode), seed 0 follows it in 598/600 turns, 103
  tokens per episode, and its sentences state guesses that then steer the queries, e.g. *"The airplane
  is the B-17 Flying Fortress. Now I need the name of the 514th Flight Test Squadron."* Seed 0 also
  repeats a query in 44/200 T episodes (3/200 in core; seed 1: 31 vs 17). Two seeds are one
  observation, not a mechanism; the pattern is written down as a hypothesis for a follow-up.

Three verbatim episodes per seed (trained hit where prompted missed, trained miss, hit in core but
miss in T) are in `out/examples.txt`; seed 0 has 76 / 41 / 36 of each, seed 1 80 / 36 / 23.

### Reading

Pre-registered reading: the recipe teaches search on Qwen3 only if every seed passes both rules, H2
holds if every seed fails one, and a split is "unstable at this scale". The split is not between
"learned search" and "learned the harness": both seeds clear Rule 1 by 0.17 or more at the low end of
the CI, which no format story explains (Exp 02 bounds format at 0.055 on Qwen2.5, and Qwen3's format is
1.000 before training). What is unstable is how much of the gain survives a new harness, and that
tracks a stylistic difference between the seeds that the recipe does not control. Compared with run2,
Qwen3 starts 0.06 higher, gains 0.08 less, ends 0.02–0.035 lower, and offers GRPO a third of the
groups with any signal.

## Limitations (to keep whatever the result)

- Two or three seeds give a range, not a statistic; no CI is computed over seeds. Run2 itself is one
  seed, so a Qwen3-vs-Qwen2.5 difference cannot be claimed beyond what one run2 supports.
- One model size (1.7B), one recipe, one harness family, in-distribution eval only, greedy decoding.
- The recipe's hyperparameters (lr, w_step, r, G, steps) were chosen on Qwen2.5 and not re-tuned, which
  favours Qwen2.5. A fair model comparison would tune both; this experiment asks a narrower question.
- Thinking mode off is a harness choice Qwen2.5 never faced. A Qwen3 run with thinking on would be a
  different experiment (and a different token budget).
- "Format-fixed" means the best of two layouts, not perfect format: if the prompted Qwen3 is below 0.90
  format validity in both, the baseline is weaker than Qwen2.5's and the Rule 1 Δ is harder to read.

## Next, if this is worth more time

A second run2 seed on Qwen2.5 (the asymmetry above); Qwen3-0.6B vs 1.7B (the size ablation Exp 01
already lists); the recipe with thinking on and a larger turn budget; MuSiQue out-of-distribution eval
of the trained Qwen3.
