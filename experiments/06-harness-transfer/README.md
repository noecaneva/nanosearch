# Experiment 06: did run2 learn to search, or learn the harness?

**Status: run 2026-11-06 (job 27579564, es_debug, 1 GPU, 11 min).** Eval only, no training. The
hypothesis and the rules that decide it were written before the job was submitted; the Results
section was filled in afterwards and the rules were not changed.

**Verdict: H1 survives, narrowly, by the survival rule.** In both unseen harnesses recall@9 stays
above dense single-shot (W 0.900, T 0.880 vs 0.853) and the paired Δ CI contains a value ≥ −0.03
(W −0.015 [−0.037, +0.010], T −0.035 [−0.062, −0.010]). The T drop is real (its CI excludes zero)
and its point estimate sits just past the margin, so the claim that survives is "mostly search, with
a harness tax of about 0.035 to 0.05 recall@9 that is present at every checkpoint".

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

Job 27579564 (2026-11-06, es0/es_debug, 1 GPU, 10:53 wall; prompt check PASS). Numbers from
`out/eval.json` (`analyze.py`, srun on es0). 200 eval questions, greedy, BM25. Core rows for step
150 and the prompted model are the existing evals (`agent_grpo_run2`, `agent_prompted`).

| adapter | harness | recall@9 | all_gold@9 | nDCG@9 | searches | format | ends done / max / malf. | tokens | old-token turns |
|---|---|---|---|---|---|---|---|---|---|
| step0150 | core | 0.915 | 0.850 | 0.759 | 2.97 | 1.000 | 6 / 194 / 0 | 40.2 | – |
| step0150 | core rerun | 0.917 | 0.855 | 0.760 | 2.97 | 1.000 | 7 / 193 / 0 | 40.1 | – |
| step0150 | W | 0.900 | 0.815 | 0.748 | 2.87 | 1.000 | 26 / 174 / 0 | 40.4 | 0 |
| step0150 | T | 0.880 | 0.790 | 0.743 | 2.87 | 0.985 | 21 / 176 / 3 | 48.1 | 0 |
| step0050 | core | 0.873 | 0.775 | 0.730 | 3.00 | 1.000 | 0 / 200 / 0 | 40.8 | – |
| step0050 | T | 0.823 | 0.705 | 0.721 | 2.90 | 0.995 | 16 / 183 / 1 | 45.5 | 0 |
| step0010 | core | 0.777 | 0.595 | 0.709 | 3.00 | 1.000 | 0 / 200 / 0 | 42.5 | – |
| step0010 | T | 0.728 | 0.550 | 0.679 | 2.98 | 0.995 | 2 / 197 / 1 | 49.3 | 0 |
| prompted | core | 0.620 | 0.330 | 0.624 | 2.08 | 0.945 | 118 / 71 / 11 | 42.0 | – |
| prompted | T | 0.565 | 0.360 | 0.540 | 2.32 | 0.995 | 71 / 128 / 1 | 59.3 | 0 |

**Noise floor.** Same-job greedy rerun of step0150 in core vs the stored eval: 197/200 episodes
identical, recall@9 Δ +0.003 [0.000, +0.007]. Every Δ below is far outside this.

**Part A (step 150), paired Δ harness − core, 10k bootstrap, 95% CI.**

| harness | recall@9 | all_gold@9 | searches | format | rule |
|---|---|---|---|---|---|
| W | −0.015 [−0.037, +0.010] | −0.035 [−0.080, +0.010] | −0.10 [−0.15, −0.06] | 0 | survives (0.900 ≥ 0.853, CI reaches −0.03) |
| T | −0.035 [−0.062, −0.010] | −0.060 [−0.105, −0.015] | −0.11 [−0.16, −0.06] | −0.015 [−0.035, 0] | survives (0.880 ≥ 0.853, CI upper −0.010 ≥ −0.03) |

Neither harness is confounded (format 1.000 and 0.985). The W wording makes the model stop earlier
(26 `<done/>` endings instead of 6; its prompt says "ends the search, once every passage the reader
needs is in hand"), which costs all_gold more than recall. T adds a real recall drop on top.

**Part B, gap core − T in recall@9 per checkpoint.**

| checkpoint | core | T | gap [CI] |
|---|---|---|---|
| step0010 | 0.777 | 0.728 | +0.050 [+0.013, +0.090] |
| step0050 | 0.873 | 0.823 | +0.050 [+0.015, +0.085] |
| step0150 | 0.915 | 0.880 | +0.035 [+0.010, +0.062] |

Gap(150) − gap(50) = −0.015 [−0.058, +0.028]: the "transfers at every stage" reading is **not
rejected** (it would need +0.03). The gap is also not large early and closed late, so the "format
first, search later" reading has no support either. What the data show is a roughly constant
harness tax of 0.035 to 0.05 recall@9 while the in-harness score rises from 0.777 to 0.915: nearly
all of the improvement from training transfers into T.

**Control.** The untrained prompted model moves under T by −0.055 [−0.105, −0.003] recall@9, with
format 0.945 → 0.995 (the separate layout, as in Exp 02), searches +0.24 and +17 generated tokens
per episode. The trained adapter's drop under T (−0.035) is smaller than the control's, so T is
not an easier harness that the adapter fails to exploit; it is a harder one for both.

**Old tokens.** `old_token_turns` = 0 in all 800 T episodes (about 2,300 turns): no adapter ever
wrote `<search>` or `<done/>` in harness T. The trained format is not sticky at the token level;
the model reads the tokens off the prompt. The 3 malformed step0150 episodes in T are
answer-instead-of-act failures (after the second result the model writes e.g. `[1] 1919`), the
same failure the prompted model shows in core.

**Three verbatim T episodes of step0150** (`out/examples.txt`; 158 clean successes, 17 misses that
core got, 0 old-token episodes, so the third slot is empty):

- Success. *Which Italian-American composer and librettist wrote the English language opera, Maria
  Golovin?* `<query>Maria Golovin opera</query>` → `<query>Gian Carlo Menotti Italian-American
  composer</query>` → `<query>Librettist Gian Carlo Menotti English language opera</query>`,
  recall@9 1.0. Core: the same first query, then `Gian Carlo Menotti composer and librettist`,
  `Italian-American composer and librettist`, recall@9 1.0.
- Miss in T, hit in core. *Near which town did the family of author of the "Adventures of
  Huckleberry Finn" own several tracts of land?* T: `Author of "Adventures of Huckleberry Finn"` →
  `Owner of several tracts of land near Tom Sawyer and Huckleberry Finn` → `Tom Sawyer and
  Huckleberry Finn owned several tracts of land in a town near them`, recall@9 0.0 (never names
  Twain). Core: `Adventures of Huckleberry Finn novel author` → `Family of Mark Twain author owning
  land` → `Monroe County, Missouri`, recall@9 1.0. In T the queries drift towards question-like
  sentences (mean query length 5.6 words vs 5.0 in core) and reuse words from the results instead of
  the entity they imply.
- Old-token turn: none.

**Reading.** Run2 learned search, with a measurable harness component. About 0.035 recall@9 of its
0.915 depends on the exact prompt, tokens and layout it was trained in; the remaining gain over the
prompted model (0.880 vs 0.620 in core, 0.565 in T) transfers to a harness it never saw, and does so
from step 10 onwards. The one-seed, same-author caveats below stand.

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
