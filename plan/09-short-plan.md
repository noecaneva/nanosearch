# 09. The Two-and-a-Half-Day Version

This file overrides the scale in `01-spec.md` and the schedule in `06-schedule-and-risks.md`. The
other files still hold the reasoning, but only the parts listed under "Kept" below are built.

The goal changes from "a finished study" to "a correct, measured, honest artifact". Every cut
below is a named next step, not a gap.

## Scale

| Dimension | Value |
|---|---|
| Policy model | Qwen2.5-1.5B-Instruct, LoRA r=16, fp16 with fp32 log-probs |
| Training questions | 500 from HotpotQA distractor |
| Eval questions | 200 from HotpotQA, in-distribution only |
| Corpus | All paragraphs of those 700 questions, deduplicated, roughly 7,000 passages |
| Group size G | 8 |
| Max searches / passages per search | 3 / 3, unchanged |
| KL | beta = 0, citing the verifiable-reward literature. One less thing to debug |
| GPUs | 3 vLLM servers or 2, 1 trainer, no DDP |

## Kept

- Eval harness with BM25 and dense single-shot baselines at equal budget (recall@9), plus the oracle.
- The prompted, untrained agent as the third baseline.
- GRPO from scratch: group advantages, clipped objective, explicit token mask, zero-variance group
  filtering, fp32 log-probs computed in chunks.
- Two smoke tests: masked token count equals vLLM's generated lengths, and 16 prompts overfit.
- One real run, logged to wandb.
- The tag-mimicry guard (stop on `<result>`), because it is silent and costs one line.
- A README write-up with the results table, three verbatim rollouts and the hacking log.

## Cut

0.5B ablation, all other ablations, MuSiQue, torchrun DDP, KL term, in-process weight sync (replaced
by loading a fresh vLLM LoRA adapter each step), resumable checkpointing (add only if es0 preempts
within the run length), the selection action space, the 5 to 8 page report.

## Schedule

**Day 1, morning.** vLLM serves the 1.5B model with `--dtype float16` on a 2080 Ti and accepts a
LoRA adapter at runtime. Cap this at one hour. If vLLM will not run on sm_75, fall back to batched
HuggingFace `generate`, cut G to 4, and write it down as a hardware limitation. Then corpus, both
indexes, the eval harness, and the BM25, dense and oracle numbers.

*Check:* oracle about 1.0, BM25 recall@9 below about 0.85.

**Day 1, afternoon.** Environment with the tag protocol and malformed-action handling. Prompted
agent evaluated on the 200 questions.

*Check:* format validity above about 60 percent with few-shot examples. If it is lower, simplify the
prompt or drop `<think>`, not the model size.

**Day 2.** GRPO loop, both smoke tests, then overfit 16 prompts. Launch the real run in the evening
and let it go overnight, around 100 to 150 steps.

*Check:* the 16-prompt overfit works. If it does not by the end of day 2, do not launch the real run.
Debug the mask and the adapter reload, then publish the floor version below.

**Day 3, morning.** Evaluate the trained adapter, fill in the hacking log from episode logs, write
the README, push the repo.

## The floor

If day 2 fails, the published artifact is the eval harness, three baselines, a unit-tested GRPO
implementation, and a written diagnosis of why the overfit did not work. Call it "correctness work
done, full run pending". That is still a better artifact than a project that was never started.

## What to say about the cuts

Single run, no seed variance. LoRA and fp16 on consumer Turing cards. A corpus three orders of
magnitude smaller than production. Adapter reload rather than a real weight-sync path. No
out-of-distribution eval. State all of this at the top of the README, then say which ablation you
would run first (0.5B versus 1.5B, then selection versus union) and why.
