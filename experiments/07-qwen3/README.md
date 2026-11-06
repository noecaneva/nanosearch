# Experiment 07: the run2 recipe on Qwen3-1.7B, does it teach search or only the harness?

**Status: planned 2026-11-06, not run.** Runs after Exp 06 (it reuses Exp 06's harness T). The rules
below are fixed before any job is submitted and are not changed once numbers are in.

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

Not yet run. Fill in: the day-0 numbers (fit, NaN check, zero-variance share), the eval table, Rule 1
and Rule 2 per seed with CIs, the seed gap and whether seed 2 ran, training curves, the side
prediction. One-sentence verdict at the top of this file, naming the rule it was decided by.

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
