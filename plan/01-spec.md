# 01. Project Specification

## Task

Multi-hop question answering, scored on retrieval rather than on the answer text. The model
receives a question, issues a search query against an index, reads the returned passages, and
decides whether to search again or stop. Reward is computed on the set of passages it ends up
with.

Retrieval is scored rather than answer text for two reasons. It is what a search product is
actually judged on, so it is the axis worth thinking about. And it gives a reward that is cheap and objective to compute,
with no LLM judge in the loop, which matters when the whole point is to iterate fast.

## Why multi-hop

A single embedding lookup cannot solve a well-formed multi-hop question, because the second
document is only findable once you have read the first. That makes the comparison against
embedding-only retrieval meaningful by construction rather than by luck. It is also the setting
where iterative search has an obvious reason to exist.

## Scale

| Dimension | Value | Why |
|---|---|---|
| Policy model | Qwen2.5-1.5B-Instruct | Small enough to iterate, large enough to emit structured actions |
| Ablation model | Qwen2.5-0.5B-Instruct | Turns "how small is too small" into a result instead of an assumption |
| Corpus | ~30,000 passages | Real search problem, indexes in minutes |
| Training questions | ~3,000 | Enough for a few hundred GRPO steps |
| Held-out questions | 500 in-distribution, 200 out-of-distribution | Generalization check |
| Group size G | 8 rollouts per prompt | Standard for GRPO, cheap at this model size |
| Precision and tuning | fp16, LoRA r=16 | Forced by Turing hardware, see `08-hardware-es0.md` |
| Max searches per episode | 3 | Hard cap, keeps episodes short and rewards bounded |
| Passages per search | 3 | Final set is at most 9 documents |

Substitute the current Qwen small-model generation if something newer is out when you start. The
design does not depend on the specific checkpoint, only on it being instruction-tuned and small.

## Which gap each choice closes

| Skill to build | What exercises it here |
|---|---|
| Train models with GRPO | Implementing GRPO from scratch, `04-grpo.md` |
| Design and iterate RL training environments for retrieval | Building the environment yourself, `03-environment.md` |
| Own the entire training pipeline, data curation to wandb | Corpus construction through to logged runs |
| Familiar with vLLM | vLLM serves every rollout |
| Comfortable with torchrun/accelerate/multi-node | Training launched under torchrun across GPUs |
| Clever about getting the data needed | Corpus built and deduplicated by hand from raw dumps |
| Critically evaluate research | Four-way baseline comparison with honest negative results |
| Must articulate ideas well | The technical report, `07-writeup-outline.md` |

Two gaps this project does not close, and the brief says so rather than pretending otherwise:
vision-first embedding models, and anything at frontier scale.

## Success criteria

The project succeeds if the write-up can answer all four of these honestly, whatever the answers
turn out to be.

1. Does the GRPO-trained agent beat the prompted agent on recall at equal document budget?
2. Does either agent beat single-shot dense retrieval, and at what latency and token cost?
3. At what model size does the approach stop working?
4. What did the model do to cheat the reward, and what fixed it?

A result where the small model fails to learn is a publishable finding for this purpose, provided
the failure is diagnosed rather than reported as a shrug. What is not acceptable is not knowing.

## Non-goals

No answer generation or answer-accuracy scoring. No comparison against frontier APIs unless it is
free to do. No attempt to beat published state of the art, since the model is two orders of
magnitude too small for that to be meaningful.
