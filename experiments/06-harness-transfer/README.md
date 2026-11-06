# Experiment 06: did run2 learn to search, or learn the harness?

**Status: planned 2026-11-06, not run.** Eval only, no training. The hypothesis and the rules that
decide it are written here before any job is submitted; the Results section is filled in afterwards
and the rules are not changed once the numbers are in.

**Question.** Run2 (Exp 01, `run2_wstep0/adapters/step0150`) reaches recall@9 0.915 against 0.620 for
the prompted model and 0.853 for dense single-shot. All of that was measured in the harness it was
trained in: the core-v1 system prompt, the `<search>`/`<done/>` tokens, the inline result layout. A
reader of the repo asked whether the gain is learned search or learned harness, i.e. the model
getting good at the format it was trained in. Exp 02/03 hint at an answer (the layout matters for
the prompted model, +0.053, and not after training, run4 vs run2 within noise) but every adapter
there was evaluated in the harness it was trained in, so they do not settle it.

**Hypothesis H1.** Run2 learned search. Prediction: in a harness it has never seen, recall@9 stays
above dense single-shot (0.853) and the paired drop against its own in-harness score is at most 0.03.

**Falsification rules, fixed before running.**

- H1 is **falsified** if, in any tested harness, the paired Δ recall@9 (new harness minus core, 200
  eval questions, 10k-sample bootstrap, 95% CI) has its CI entirely below −0.03, or recall@9 falls
  under 0.853.
- H1 **survives** if, in every tested harness, the CI contains a value ≥ −0.03 and recall@9 ≥ 0.853.
  A CI that contains zero is not evidence *for* H1 on its own; the margin is what carries the claim.
- **Confounded, reported separately:** format validity under 0.90 in a harness. Then the model
  could not act and the harness says nothing about search. Report it as "could not act", not as
  a search failure.
- **Trajectory (Part B):** the reading "what it learns transfers at every stage" is rejected if the
  gap (core minus harness T recall@9) at step 150 exceeds the gap at step 50 by more than 0.03. A gap
  that is large at step 10 and closes by step 150 is read as "format first, search later".

## Design

**Models.** `run2_wstep0` adapters `step0010`, `step0050`, `step0150`, from
`$GRPOP_ROOT/outputs/runs/run2_wstep0/adapters/` (the trainer saves every step, see
`sync_adapter` in `nanosearch/grpo.py`; check the three directories exist before anything else).
Plus the untrained prompted model as the control.

**Harnesses.** A nested ladder; each rung keeps the previous rung's changes.

- **core** = core-v1, unchanged: `SYSTEM_PROMPT`, `<search>…</search>` / `<done/>`, inline results,
  reminder on. The default of every code path must stay exactly this (asserted in the job, as Exp 05
  did for `system_note`).
- **W (reworded):** a paraphrased system prompt with the same tokens, same rules and a different
  worked example. Text below, to be used verbatim.
- **T (tokens + layout):** the W prompt with the action tokens renamed to `<query>…</query>` /
  `<stop/>`, the reminder renamed accordingly, and `--separate_results` (Exp 02 layout). The
  `<result>` block tag stays as it is, so the stop strings change only for the two action tokens.
  In T, the **old** tokens are not actions: a turn ending in `<search>…</search>` parses as
  `malformed/no_action`. That is the point of the test, the model must adopt the new tokens. Count
  these turns separately (`old_token_turns`), they measure how sticky the trained format is.

**W system prompt (verbatim):**

```
You search Wikipedia for a reader who has a question that needs facts from more than one passage.
Your task is only to collect those passages, not to state the answer. The search engine is BM25,
so it matches keywords.

In each turn write one or two sentences about what you still need, then finish the turn with a
single action:
<search>keywords</search>   shows you the 3 best-matching passages
<done/>                     ends the search, once every passage the reader needs is in hand

Do not state the answer to the question. Each turn has to finish with <search>...</search> or <done/>.
At most 3 searches are allowed. Good keywords name the exact thing you still need, based on what
the earlier results told you. Never send the same query twice.

Example:
Question: How many inhabitants does the town have in which the composer of the opera Nabucco was born?
The opera is Nabucco, so first I need its composer. <search>Nabucco opera composer</search>
<result>
[1] Nabucco | Nabucco is an opera in four acts by Giuseppe Verdi, first performed in Milan in 1842...
[2] Giuseppe Verdi | Giuseppe Verdi was an Italian composer, born in Le Roncole, a village in the municipality of Busseto...
[3] Va, pensiero | Va, pensiero is a chorus from the opera Nabucco...
</result>
Verdi was born in Le Roncole, which belongs to Busseto. Now I need the population of Busseto. <search>Busseto population</search>
<result>
[1] Busseto | Busseto is a comune in the province of Parma, Emilia-Romagna, with about 7,000 inhabitants...
[2] Province of Parma | The Province of Parma is a province in Emilia-Romagna...
[3] Le Roncole | Le Roncole is a village in the comune of Busseto, known as Verdi's birthplace...
</result>
I have the composer's birthplace and the town's population. <done/>
```

**T system prompt:** the W text with every `<search>` → `<query>`, `</search>` → `</query>`,
`<done/>` → `<stop/>`. **T reminder:** `Do not answer the question. End this turn with
<query>...</query> or <stop/>.` Used with `--separate_results`.

**Evals, all greedy, 200 eval questions, BM25, one vLLM server.** Eight runs:

| # | adapter | harness | purpose |
|---|---|---|---|
| 1 | step0150 | W | Part A, mild change |
| 2 | step0150 | T | Part A, strong change |
| 3 | step0150 | core | same-job rerun, noise floor for greedy decoding (existing: `agent_grpo_run2`) |
| 4 | step0010 | core | Part B |
| 5 | step0010 | T | Part B |
| 6 | step0050 | core | Part B |
| 7 | step0050 | T | Part B |
| 8 | prompted | T | control: how far the untrained model moves under T (core exists: `agent_prompted`) |

Out names `exp06_<adapter>_<harness>`. Load and unload adapters one at a time (`serve.sh` runs
with `--max-loras 2`).

**Metrics.** recall@9, all_gold@9, nDCG@9, recall@3 of each query, searches per episode, format
validity, ends done / max / malformed, `old_token_turns` (T only), tokens generated. Paired
bootstrap per question against the same adapter in core (reuse `bootstrap` from
`experiments/04-summary-judge/analyze.py`, as Exp 05 does). For the trajectory, the gap core − T per
checkpoint with its CI.

## Implementation

Eval-only code change, `grpo.py` untouched. Keep it small:

- `nanosearch/env.py`: a `Harness` record (system prompt, search open/close tokens, done token,
  reminder) with `CORE` holding today's module constants. `SearchEnv(..., harness=CORE)`;
  `parse_turn(text, harness=CORE)`; `STOP` built from the harness. Module-level `SYSTEM_PROMPT`,
  `STOP`, `REMINDER`, `ACTION_RE` keep their current values so `grpo.py` and the Exp 04/05 analysis
  scripts import unchanged.
- `nanosearch/agent.py`: `--harness core|W|T` (default `core`), recorded in the eval summary next to
  `system_note`; the stop strings passed to vLLM come from the harness. `--harness T` implies
  `--separate_results`.
- `experiments/06-harness-transfer/evals.sh` (es0, `es_debug`, 1 GPU, ~45 min, modelled on Exp 05)
  with a prompt check before the evals: default env equals core-v1 exactly; harness W contains
  `<search>` and no `<query>`; harness T contains `<query>` and `<stop/>` and no `<search>`; in T,
  `parse_turn("... <search>x</search>")` is `malformed/no_action`.
- `experiments/06-harness-transfer/analyze.py` → `out/{eval.json, examples.txt}`: the table above,
  the paired Δs, the trajectory gaps, and three verbatim T episodes of step0150 (one success, one
  miss, one with an old-token turn if any).

## Run

```
sbatch -o $SCRATCH/grpop/logs/%x-%j.out experiments/06-harness-transfer/evals.sh
PYTHONPATH=. python experiments/06-harness-transfer/analyze.py      # es0 node -> out/
```

## Results

Not yet run. Fill in: the eight-row table, Part A Δs with CIs against the rules above, the Part B
trajectory, the control's core → T move, three verbatim T episodes. State the verdict on H1 in one
sentence at the top of this file, with the rule it was decided by.

## Limitations (to keep whatever the result)

- One seed, one model (Qwen2.5-1.5B), 200 in-distribution eval questions, greedy decoding.
- Two harnesses plus the training one. W changes the words, T changes words, tokens and layout;
  the ladder has no rung for "tokens only", so a drop in T is not attributed to one of its parts.
- A harness the model has never seen is still a harness built by the same author in the same
  style; a harness from someone else would be a stronger test.
- The format-vs-search question for Qwen2.5 itself (how much of 0.620 → 0.915 is format) is not
  answered here; that control (a few-shot format-fixed prompted baseline) belongs to Exp 07.

## Next, if this is worth more time

Three training seeds of run2 and the full checkpoint trajectory (steps 0, 10, 25, 50, 100, 150); a
"tokens only" rung; a harness written by someone else; MuSiQue as an out-of-distribution eval,
which tests transfer across data rather than across harness.
