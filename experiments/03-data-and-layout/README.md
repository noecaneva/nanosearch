# Experiment 03: more training questions, and the separate-message layout in training

**Question.** run2 (500 train questions, inline results) reaches eval recall@9 0.915, but by the end
81% of its groups have zero reward variance and are dropped (each question seen ~5 times). Does
training on 3,000 questions help (run3 vs run2)? Does the separate-message result layout of Exp 02,
which helped the prompted agent, also help GRPO (run4 vs run3)?

**Data.** `core-v1/data/train3000.jsonl` = the 500 core-v1 train questions + the 2,500 extra
train-split questions whose paragraphs were already in the corpus as distractors (`corpus.py`,
same seed and permutation). The corpus is unchanged: a rebuild gives byte-identical corpus,
train and eval files (md5s in `core-v1/MD5SUMS`). All 3,000 have both gold passages in the index.
Same level mix as the 500 (~19% easy, 64% medium, 17% hard; 80% bridge). Eval: the same 200
eval questions as everything else.

**Runs.** Identical to run2 except the listed change: 150 steps, 16 prompts x G=8, lr 5e-5,
w_step 0 in training (eval reward keeps the core 0.05), LoRA r=16, seed 0, full es0 node.

| run | train questions | layout | compared with |
|---|---|---|---|
| run2 (Exp 01) | 500 | inline | – |
| run3_x3000 | 3,000 | inline | run2: effect of data |
| run4_x3000_sep | 3,000 | separate message + reminder | run3: effect of layout |

```
C="--lr 5e-5 --w_step 0 --train_file train3000.jsonl"
sbatch -J nanosearch-run3 -o $SCRATCH/grpop/logs/%x-%j.out ../01-hotpotqa-grpo/run.sh run3_x3000 $C
sbatch -J nanosearch-run4 -o $SCRATCH/grpop/logs/%x-%j.out ../01-hotpotqa-grpo/run.sh run4_x3000_sep $C --separate_results
sbatch -o $SCRATCH/grpop/logs/%x-%j.out ../01-hotpotqa-grpo/eval.sh <adapter> <out_name> [--separate_results]
```

**Smoke** (interactive, 1 server + 1 trainer, separate layout, train3000, 3 steps x 16 x 8): asserts
hold, grad scale 65536, 12-25% groups filtered, **trainer peak memory 3.6 GB** of 10.5 GB.

## Results (200 eval questions, greedy, step 150, each adapter in its training layout)

| run | train q | layout | recall@9 | all_gold@9 | nDCG@9 | recall@3 | bridge r@9 | comparison r@9 | searches | format | reward |
|---|---|---|---|---|---|---|---|---|---|---|---|
| run2 (Exp 01) | 500 | inline | **0.915** | **0.850** | 0.759 | 0.490 | 0.900 | 0.975 | 2.97 | 1.000 | 0.967 |
| run3_x3000 | 3,000 | inline | 0.893 | 0.810 | 0.745 | 0.495 | 0.872 | 0.975 | 2.99 | 0.995 | 0.942 |
| run4_x3000_sep | 3,000 | separate | 0.910 | 0.845 | **0.775** | **0.540** | 0.897 | 0.963 | 3.00 | 1.000 | 0.960 |

Paired per question, 10k-sample bootstrap:

| comparison | Δ recall@9 [95% CI] | Δ all_gold@9 [95% CI] | questions better / worse |
|---|---|---|---|
| run3 - run2 (more data) | -0.023 [-0.053, +0.005] | -0.040 [-0.090, +0.010] | 11 / 19 |
| run4 - run3 (layout) | +0.018 [-0.013, +0.048] | +0.035 [-0.020, +0.090] | 19 / 12 |
| run4 - run2 (both) | -0.005 [-0.033, +0.023] | -0.005 [-0.050, +0.040] | 11 / 13 |

**Negative result: neither more data nor the separate layout improves the trained model.** All
three end at recall@9 0.89-0.915 and every CI contains zero. run4 has the best ranking metrics
(recall@3 0.540, nDCG 0.775) and the highest early training reward, but the final recall is the same.
Exp 02's +0.053 for the prompted agent does not carry over: GRPO learns the format on its own within
~10 steps, whatever the layout.

**More data did not stop the zero-variance groups.** Training curves (window means):

| steps | run2 filtered | run3 filtered | run4 filtered | run2 train r@9 | run3 train r@9 | run4 train r@9 |
|---|---|---|---|---|---|---|
| 0-10 | 0.18 | 0.16 | 0.29 | 0.650 | 0.670 | 0.683 |
| 10-30 | 0.45 | 0.39 | 0.49 | 0.796 | 0.801 | 0.824 |
| 30-75 | 0.69 | 0.59 | 0.62 | 0.857 | 0.845 | 0.855 |
| 75-120 | 0.76 | 0.71 | 0.73 | 0.893 | 0.866 | 0.855 |
| 120-150 | 0.79 | 0.81 | 0.73 | 0.901 | 0.897 | 0.889 |

With 3,000 questions each is sampled < 1 time in 150 x 16 prompts, yet ~80% of groups still end
with identical rewards for all 8 rollouts. So run2's filtering was not memorization of 500
questions: the policy becomes near-deterministic (mean -logp 0.1-0.2) and reaches the same recall on
fresh questions (train window 0.89-0.90) as on eval. The plateau looks like a ceiling of this
policy + environment (3 BM25 searches x 3 passages, ~5-7-word queries), not a data limit.

Trainer peak memory stayed at 3.5-3.6 GB of 10.5 GB in both runs (r=16). Wall 1:54 (run3), 1:50
(run4); eval jobs 27184698, 27184699. One seed per run.

## What this suggests next (not run)

Raise the ceiling rather than the data: more passages or searches per episode, keep exploration up
(temperature, larger G, or an entropy/KL term), or harder questions (all level=hard, like eval).
Rank is unlikely to matter here; memory would allow r=64.
