# Experiment 01: GRPO on HotpotQA multi-hop

**Question.** Does GRPO teach Qwen2.5-1.5B-Instruct (LoRA r=16) to search a BM25 engine better
than the same model prompted, and better than single-shot retrieval at the same 9-passage budget?

**Answer, at this scale.** Yes for recall, once the step penalty is removed from the training
reward. run2 reaches recall@9 0.915 on 200 in-distribution eval questions, against 0.853 for dense
and 0.825 for BM25 single-shot and 0.620 for the prompted model. It does this by always using all 3
searches, so under the core reward (which charges 0.05 per search) it scores 0.967, slightly below
the single-shot baselines (0.975-1.003). With the step penalty in training (run1) the policy
collapses to one search and stays at the prompted level. One seed per run, 1.5B model, 30k-passage
corpus, fp16 + LoRA on 2080 Ti cards. See Limitations.

**Setup.** core-v1, 500 train questions. Union action space (`<search>` / `<done/>`, final set =
union of results, at most 3 searches x 3 passages). G = 8, 16 prompts per step, 150 steps, beta = 0,
clip 0.2, lr 5e-5 (the default in `grpo.py` is 2e-5; both runs passed `--lr 5e-5`, between the two
overfit learning rates), 2 minibatch optimizer steps per batch, token-level loss normalization,
zero-variance groups dropped, sampling at temperature 1. LoRA r=16, alpha 32, all 7 projections.
fp16 base, fp32 LoRA weights, fp32 log-probs. One es0 node: 3 vLLM servers + 1 trainer GPU, fresh
LoRA adapter loaded into vLLM every step. Details in `nanosearch/grpo.py`.

Training reward: `recall@9 + 0.2 * format_valid - w_step * n_searches`. `--w_step` sets the step
penalty for training only (default 0.05, the core value); eval always uses the core reward with 0.05.
run1 trains with 0.05, run2 with 0.

**Run.**

```
sbatch --gres=gpu:2 -c 4 -o $SCRATCH/grpop/logs/%x-%j.out run.sh overfit16 --overfit 16 --steps 20
sbatch -o $SCRATCH/grpop/logs/%x-%j.out run.sh run1 --lr 5e-5
sbatch -o $SCRATCH/grpop/logs/%x-%j.out run.sh run2_wstep0 --lr 5e-5 --w_step 0
sbatch -o $SCRATCH/grpop/logs/%x-%j.out eval.sh $SCRATCH/grpop/outputs/runs/run1/adapters/step0150 agent_grpo_run1
sbatch -o $SCRATCH/grpop/logs/%x-%j.out eval.sh $SCRATCH/grpop/outputs/runs/run2_wstep0/adapters/step0150 agent_grpo_run2
```

Slurm jobs: run1 27123091 (1:18:36 wall), run2 27130797 (1:40:38), evals 27130794 and 27164089.
The lr and w_step values above are the ones recorded in each run's `config.json`.

**Checks before the real run.** The masked token count equals vLLM's generated lengths (asserted
on every generation call); advantages sum to zero per group (asserted); 16 prompts overfit
(reward 0.64 -> 1.00 at lr 1e-4, 0.60 -> 0.90 at lr 2e-5).

## Results

200 eval questions (HotpotQA validation, all level=hard, 160 bridge / 40 comparison), greedy
decoding, step-150 adapters. Rows as in [RESULTS.md](../../RESULTS.md).

| System | recall@9 | all_gold@9 | nDCG@9 | bridge r@9 | comparison r@9 | searches | format valid | reward |
|---|---|---|---|---|---|---|---|---|
| BM25, question as query | 0.825 | 0.665 | 0.751 | 0.794 | 0.950 | 1 | – | 0.975 |
| Dense (bge-small), question as query | 0.853 | 0.740 | 0.807 | 0.816 | 1.000 | 1 | – | 1.003 |
| Oracle (gold passages) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | – | – | 1.150 |
| Prompted agent, Qwen2.5-1.5B, BM25 env | 0.620 | 0.330 | 0.624 | 0.588 | 0.750 | 2.08 | 0.945 | 0.705 |
| run1: GRPO, w_step 0.05, step 150 | 0.640 | 0.395 | 0.662 | 0.603 | 0.788 | 1.00 | 1.000 | 0.790 |
| run2: GRPO, w_step 0 (training only), step 150 | **0.915** | **0.850** | 0.759 | 0.900 | 0.975 | 2.97 | 1.000 | 0.967 |

Cost columns for the agents, from each `eval.json`. Latency is per episode with 64 episodes in
flight on one 2080 Ti (one vLLM server), so it measures throughput under load, not single-request
latency. BM25 and dense single-shot take 0.67 ms and 13.8 ms (p50) per query.

| System | recall@3 | distinct passages | tokens generated / episode | latency p50 / p95 | eval wall (200 q) |
|---|---|---|---|---|---|
| Prompted agent | 0.525 | 4.48 | 42.0 | 6.5 s / 15.3 s | 27 s |
| run1 | 0.640 | 3.00 | 37.3 | 7.3 s / 9.7 s | 25 s |
| run2 | 0.490 | 7.07 | 40.2 | 13.9 s / 18.8 s | 47 s |

Paired per question (10k-sample bootstrap, 95% CI, computed from the `eval_episodes.jsonl` files):

| comparison | Δ recall@9 | Δ all_gold@9 | questions better / worse (recall) |
|---|---|---|---|
| run2 - dense | +0.062 [+0.020, +0.105] | +0.110 [+0.035, +0.185] | 44 / 21 |
| run2 - BM25 | +0.090 [+0.052, +0.130] | +0.185 [+0.115, +0.255] | 51 / 16 |
| run2 - prompted | +0.295 [+0.253, +0.338] | +0.520 [+0.445, +0.590] | |
| run1 - prompted | +0.020 [-0.022, +0.062] | +0.065 [-0.005, +0.140] | |

### run1: collapse to one search

Training curve (`outputs/runs/run1/metrics.jsonl`, 128 episodes per step at temperature 1):

- Searches per episode 1.95 at step 0, first at or below 1.05 at step 18, and between 0.95 and
  1.04 at every step from 50 on. Query length 4.7 words at step 0, mean 9.5-11.1 words per 25-step block
  after step 10.
- Train recall@9 0.48 at step 0, mean 0.68 over steps 125-149. It cannot go much higher with one
  search of 3 passages.
- Groups dropped for zero reward variance: mean 0.32 (steps 10-24), 0.52 (25-49), 0.75 (50-74),
  0.81 (75-99), 0.88 (100-124), 0.89 (125-149). 7 steps had every group dropped.
- Step time (t_gen + t_train): 86 s over the first 10 steps, 13.5 s over the last 25, because
  dropped groups are not trained on. That is why the run took 1:18 rather than the planned ~4 h.

On eval, all 200 episodes issue exactly one query (mean 9.7 words, longest 31, one under the
32-word cap) and then stop. recall@9 = recall@3 = 0.640, about what BM25 gets from the raw
question in its top 3 (0.660).

### run2: no step penalty in training

Same settings, `--w_step 0` (`outputs/runs/run2_wstep0/metrics.jsonl`):

- Searches per episode 1.95 at step 0, first above 2.9 at step 8, between 2.87 and 3.00 at every
  step from 50 on. Queries stay short: 4.5 words at step 0, mean 4.7-5.7 per block.
- Train recall@9 0.50 at step 0, mean 0.905 over steps 125-149 (0.938 at step 149). Train
  all_gold@9 0.23 -> 0.83 (mean of the last 25 steps).
- Groups dropped: mean 0.18 (steps 0-9), 0.42 (10-24), 0.65 (25-49), 0.71 (50-74), 0.72 (75-99),
  0.82 (100-124), 0.80 (125-149). 2 steps had every group dropped.
- Step time: 82 s over the first 10 steps (t_gen 15 s, t_train 66 s), 27 s over the last 25
  (t_gen 10 s, t_train 17 s).

On eval, 194 of 200 episodes end at the search cap. The gain over single-shot is mostly getting
both hops: all_gold@9 0.850 vs 0.740 for dense. No single query beats the question (recall@3 0.490
vs 0.660 for BM25 on the question); the gain comes from the union of decomposed sub-queries.
Replaying the logged queries against the core-v1 BM25 index (the union matches the logged
`retrieved` list on 200/200 episodes), the first query finds 196 of the 366 gold passages
retrieved, the second 149, and the third 21. The third search is mostly spent: it adds no new
passage at all in 34 of 194 episodes.

### Three episodes, verbatim

Model turns are copied from `eval_episodes.jsonl`. The titles under each turn are the 3 passages
that query returned, from the BM25 replay above; `*` marks a gold passage. The result blocks and
the reminder line the environment inserts between turns are left out.

**1. run2 success (recall@9 1.0).** BM25, dense, the prompted model and run1 all get 0.5 on this
question.

```
Question: In what year did the playwright who wrote "The City Heiress" die?
Gold: The City Heiress; Aphra Behn

turn 1: <search>The City Heiress play</search>
        -> *The City Heiress, The Heiress (1786 play), The Hypocrites (1923 film)
turn 2: <search>Aphra Behn playwright</search>
        -> *Aphra Behn, *The City Heiress, Ari Behn
turn 3: <search>Aphra Behn died year</search>
        -> *The City Heiress, *Aphra Behn, Comedy of intrigue
end: max_searches, recall@9 1.0, reward 1.05
```

The second query names the playwright, who appears only in the first result block. The turns
carry no reasoning text (see the failure log). The third query adds one passage that is not gold.
On the same question the prompted model searches once and then answers inside a valid stop turn
(`The play "The City Heiress" was written by Aphra Behn, who died in 1689. <done/>`), and run1
searches `The City Heiress playwright died year` once.

**2. run2 miss (recall@9 0.5).**

```
Question: Are both Cypress and Ajuga genera?
Gold: Ajuga; Cypress

turn 1: <search>Cypress genus</search>
        -> Fokienia, Cupressus, Cypress-pine
turn 2: <search>Ajuga genus</search>
        -> *Ajuga, Ajugeae, Tritoniopsis
turn 3: <search>Are Cypress and Ajuga genera?</search>
        -> Cypress-pine, Ajugeae, Cupressus
end: max_searches, recall@9 0.5, reward 0.55
```

"Cypress genus" returns three related pages but not the "Cypress" article, and the third query
falls back to the question itself and returns nothing new. Dense on the raw question gets both
gold passages (recall 1.0); BM25 gets 0.5. The policy does not rephrase after a miss: it keeps its
usual "entity + type word" pattern, and the reward gives it no signal about which hop failed.

**3. run1 search once and stop (recall@9 0.5).**

```
Question: What arcade game is named after the only player in major league history to have a
0.300 batting average for 7 consecutive seasons?
Gold: Frank Thomas (designated hitter); Frank Thomas' Big Hurt

turn 1: <search>arcade game named after only player in major league history with 0.300 batting average for 7 consecutive seasons</search>
        -> *Frank Thomas (designated hitter), List of Major League Baseball batting champions, Ron Herbel
turn 2: Now I have the player's name Frank Thomas and the year 1991. <done/>
end: done, recall@9 0.5, reward 0.65
```

The query is the question with a few words dropped. It finds the first hop, and the model stops
with the second hop one query away. The stop turn mentions a year that is not in the question. run2
on the same question follows its first query with `Arcade game named after Frank Thomas` and gets
both.

## Reward-hacking / failure log

Counts below come from the eval JSONLs (200 episodes each) and from `rollouts.jsonl`, which logs one
group (8 episodes of one question) per training step, so 1,200 episodes per run.

**1. The prompted model answers instead of acting.** After its first result block the base model
writes the answer and ends the turn without an action, which is malformed. Format validity on 200
train questions (`outputs/promptdev_v{0,1,2}/train.json`): v0 (system prompt + one example) 32%,
136 episodes malformed `no_action`; v1 (+ "Never write the answer to the question") 46%, 108;
v2 (+ a one-line reminder appended by the environment after every result block) 94%, 12. v2 is
core-v1 and both GRPO runs train with the reminder. A typical v0 turn after the first search:
`Howard Ashman worked with Alan Menken on the musical "Little Shop of Horrors" released in 1982.`
The answer still leaks into valid turns: see episode 1, where the stop turn contains the answer.
In training a malformed episode keeps its recall and loses only the 0.2 format bonus; in run2's
rollouts an episode that ended in an answer instead of an action (step 0) still scored 1.0.

**2. The prompted model searches for "...".** 98 of the prompted agent's 416 eval queries contain
no letter or digit, almost all `<search>...</search>`, in 49 of 200 episodes. They count as valid
searches and return unrelated passages (on the Cypress question: 1974 Asian Games medal table,
Gamera: Guardian of the Universe, Yarmouth Runic Stone). Our guess is that it copies the "..." that
ends each passage in the few-shot result block. The prompted model also repeats a query in 78 of
200 episodes (29 counting only real queries), and 88 of 200 episodes end with only 3 distinct
passages. Neither run1 nor run2 issues a "..." query on eval; run2's rollouts have them in 6
episodes, the last at step 93.

**3. run1 collapses to one search under w_step 0.05.** Described above. A second search costs
0.05 for certain and pays only if it finds a new gold passage, while one long query already keeps
the format bonus. Our reading is that "search once, stop" becomes a safe local optimum early (step
~18) and the zero-variance filter then removes most of the signal that could move it: by the end
88-89% of groups have identical rewards. The long queries are question rewrites: 159 of 200 eval
queries use only words from the question (run2: 197 of 594), and 44 of 200 have at least 80% of
the question's length with at least 90% of words from it (run2: 4 of 594).

**4. run1's stop turn is a template.** All 200 run1 eval stop turns start with "Now I have", and 92
of them contain "novel's author", taken from the few-shot example's last line ("I have the novel's
author and the university's founding year"), whatever the question. On the Cypress question:
`Now I have the novel's author and the genus name. <done/>` (recall 0.0). Some name facts that are
not in the question, as in episode 3. The text costs nothing and earns nothing, so nothing
corrects it.

**5. run2 always searches 3 times.** With w_step 0 a search is free in training, so using the cap
is the optimum of the training reward, not a hack of it. It is the reason run2 scores below the
single-shot baselines on the core reward, and it means stopping is not learned at all: 194 of 200
eval episodes end at the cap. The third query adds a new gold passage in 21 of 194 episodes.

**6. Reasoning text disappears.** The system prompt asks for one or two sentences of thought before
each action. Every one of run2's 600 eval turns is a bare action tag, and every run1 search turn
is too (the prompted model: 394 of 545 turns). Nothing in the reward pays for the text, and the
policy still uses entities from the results (episode 1), so the thinking was not needed at this
scale.

**7. Rare: query stuffing, tag mimicry, degenerate repetition.** In the rollout samples, one run1
and one run2 episode hit the 32-word query cap (`query_too_long`); one run2 episode wrote its own
`<result>` tag (step 29) and was stopped by the stop string; one run2 turn (step 60) degenerated
into repeating "turn-based 4X video game" until the 256-token cap. None of these occur in the run1
or run2 eval episodes. Queries copied verbatim from the question are rare: 2 of 200 in run1's eval,
none exactly in run2's, though run2's last query sometimes restates the question (episode 2).

## Limitations

- **One run per configuration, one seed** (seed 0), one greedy eval on 200 questions. The run1 vs
  run2 contrast is large, but nothing here measures seed variance.
- **fp16 + LoRA r=16 on RTX 2080 Ti** (Turing, 10.5 GB, no bf16). fp16 base with fp32 LoRA weights
  and fp32 log-probs. No full fine-tuning, and vLLM needs the Triton attention workaround on sm_75.
- **Small, clean corpus**: 30,358 passages built from the questions' own HotpotQA paragraphs plus
  distractor paragraphs from 2,500 other questions. Roughly three orders of magnitude below a
  production index, and every question's gold passages are guaranteed to be in it.
- **Adapter reload instead of weight sync.** The trainer saves the LoRA adapter and vLLM unloads and
  reloads it every step. Correct, but not how a larger setup would sync weights.
- **In-distribution eval only**: train and eval are both HotpotQA distractor (train vs validation
  split). No MuSiQue or other out-of-distribution set.
- **run2 does not win on the core reward.** It scores 0.967 vs 0.975 (BM25) and 1.003 (dense)
  because it pays for 3 searches. The recall gain is real (CI above), but it was bought by
  training without the cost the eval reward charges.
- The environment appends a one-line "act, do not answer" reminder after each result block; without
  it the prompted 1.5B model answers after its first search (format validity 32% vs 94% on 200 train
  questions). Tag protocol, not native tool calling. wandb ran offline; the logged metrics are the
  `metrics.jsonl` files.

## Next steps

- **GPU split: 2 vLLM servers + 2 trainer GPUs (DDP) instead of 3 + 1.** The trainer is the
  bottleneck: in run2, generation took 11-15 s/step and training 20-67 s/step, so the servers idle
  most of the step. vLLM and the trainer do not fit on one 10.5 GB card together, so at least one
  GPU must stay a server. Estimate: ~82 s -> ~57 s/step early, ~31 s -> ~27 s late (the gain
  shrinks as filtered groups leave less to train on). Speed only, results unchanged. Time ~3
  smoke steps per split before switching. Not done.
- **Results inline vs as a separate chat message.** Not tested in this experiment, which uses the
  core-v1 inline layout. Tested later: for the prompted model a separate message helps
  (Exp 02, eval recall@9 +0.053), but after GRPO it makes no difference (Exp 03, run4).
- **Calibrated stopping.** run1 stops too early and run2 never stops. Next: a step penalty that
  ramps up after the policy has learned to search, or a penalty only on searches that add no new
  passage, so the model learns when another search is worth it. Not done.
- **0.5B vs 1.5B.** The first ablation to run: does the 0.5B model learn query decomposition at
  all, and does it collapse the same way under w_step 0.05? Not done.
- **MuSiQue out-of-distribution eval** of the run2 adapter, to see whether the "one sub-query per
  entity" policy transfers beyond HotpotQA. Not done.
