# nanosearch

A testbed for small search agents. The aim is to teach very small open models (0.5–1.5B) to drive
a fixed search engine well: write good follow-up queries from what they have read, and know when
to stop.

The core stays frozen: corpus, environment, eval harness, baselines (BM25, dense, prompted agent,
oracle) and a from-scratch GRPO trainer. Every idea becomes a numbered experiment measured against
the same table, [RESULTS.md](RESULTS.md).

Built with Claude Code. Noè Canevascini designed the experiments and checked the analyses; the
code, runs and write-ups were produced by the agent under that direction.

## Headline result (Experiment 01)

GRPO on Qwen2.5-1.5B-Instruct (LoRA r=16), searching a BM25 engine over 30k HotpotQA passages with
at most 3 searches x 3 passages, 200 in-distribution eval questions:

| System | recall@9 | all_gold@9 | searches | reward (core) |
|---|---|---|---|---|
| BM25, question as query | 0.825 | 0.665 | 1 | 0.975 |
| Dense (bge-small), question as query | 0.853 | 0.740 | 1 | 1.003 |
| Prompted 1.5B agent | 0.620 | 0.330 | 2.08 | 0.705 |
| GRPO run1, step penalty 0.05 in training | 0.640 | 0.395 | 1.00 | 0.790 |
| GRPO run2, no step penalty in training | **0.915** | **0.850** | 2.97 | 0.967 |

- With the core reward's step penalty (0.05 per search), training collapses to one long query and
  stop (run1), no better than the prompted model.
- Without it (run2), the model learns to split the question into one short query per hop and beats
  dense single-shot by +0.062 recall@9 (95% CI +0.020 to +0.105) and +0.110 all_gold@9.
- It always uses all 3 searches, so on the core reward, which charges for them, it scores slightly
  below the single-shot baselines (0.967 vs 0.975-1.003). It has not learned when to stop.

**Limitations, stated up front.** One run per configuration, one seed. 1.5B model, LoRA and fp16 on
RTX 2080 Ti cards. A 30k-passage corpus built from the questions' own paragraphs plus distractors,
about three orders of magnitude smaller and cleaner than production search. Weight sync by
reloading a LoRA adapter into vLLM every step. In-distribution eval only (HotpotQA validation, no
MuSiQue). The first ablation to run next is 0.5B vs 1.5B, to see whether query decomposition needs
the larger model; then calibrated stopping, since run2 stops before the search cap in only 6 of
200 eval episodes.

Full write-up with training curves, three verbatim episodes and the reward-hacking log:
[experiments/01-hotpotqa-grpo/README.md](experiments/01-hotpotqa-grpo/README.md). All numbers:
[RESULTS.md](RESULTS.md).

## Layout

| Path | What |
|---|---|
| `nanosearch/` | The core (`core-v1`): `corpus.py`, `indexes.py`, `baselines.py`, `env.py`, `agent.py`, `grpo.py`, `serve.sh` |
| `experiments/NN-name/` | One directory per experiment: how it was run, what came out |
| `checks/` | One-off hardware, vLLM, token-mask and multi-GPU gradient checks |
| `plan/` | Private planning notes and the work log |

## Experiments

- **01 – GRPO on HotpotQA multi-hop** (`experiments/01-hotpotqa-grpo/`). Tagged `v0.1`.
- **02 – result block in its own chat message** (`experiments/02-result-message/`), prompted agent:
  format validity up, eval recall@9 +0.053.
- **03 – 3,000 training questions and the separate layout in GRPO** (`experiments/03-data-and-layout/`):
  no change from run2 (negative result).
- **04 – a grader that reads only the final message** (`experiments/04-summary-judge/`): with a
  rule-based "summary judge" alone, GRPO games it (judge 0.85, recall 0.645, one search); half
  recall + half judge does not (recall 0.873).
- Engineering: trainer on 2 GPUs (`NTRAIN=2`), checked equal to the 1-GPU update, 14-25% faster per
  step early in training. Details at the end of [RESULTS.md](RESULTS.md).
- Next: 0.5B vs 1.5B; a `<select>` action so the model cannot win by searching everything;
  calibrated stopping; out-of-distribution eval (MuSiQue); distillation from a large model's
  search traces; messier real-world corpora.

## Run

Lawrencium es0 (4x RTX 2080 Ti). Everything heavy lives under `$GRPOP_ROOT` (`source env.sh`).

```
python -m nanosearch.corpus          # core-v1 data
python -m nanosearch.indexes         # BM25 + dense
python -m nanosearch.baselines       # single-shot rows of RESULTS.md
nanosearch/serve.sh & python -m nanosearch.agent   # prompted-agent row (GPU node)
sbatch experiments/01-hotpotqa-grpo/run.sh run2_wstep0 --lr 5e-5 --w_step 0   # GRPO, Exp 01 run2
NTRAIN=2 sbatch experiments/01-hotpotqa-grpo/run.sh <name> ...   # same, 2 servers + 2 trainer GPUs
```
