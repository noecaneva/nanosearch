# Agentic Retrieval with GRPO

A self-contained project brief. Train a very small open-weight model to search a corpus
iteratively, using GRPO implemented from scratch, and measure whether it beats single-shot
retrieval on multi-hop questions.

**Built for:** practising the whole loop of training a small search agent with RL: environment,
GRPO, vLLM rollouts, multi-GPU training, evaluation and write-up. See `01-spec.md` for the mapping.

**Deliberate scale:** extremely small. A 1.5B model, a 30k-passage corpus, roughly 3,000 training
questions. This is a correctness-and-craft project, not a scale project. Everything here is sized for the
es0 partition, 4x RTX 2080 Ti at 10.5 GiB each. See `08-hardware-es0.md`, which overrides the
precision and fine-tuning choices stated elsewhere.

## Files

| File | What it holds |
|---|---|
| `01-spec.md` | The task, scope, success criteria, and which gap each choice closes |
| `02-data-and-eval.md` | Datasets, corpus construction, indexes, metrics, the four baselines |
| `03-environment.md` | Action space, rollout protocol, episode rules |
| `04-grpo.md` | The math, and the two implementation details that decide whether it works |
| `05-reward.md` | Reward terms, and a register of the ways the model will try to cheat |
| `06-schedule-and-risks.md` | Fourteen days, daily checkpoints, kill criteria, risk register |
| `07-writeup-outline.md` | Skeleton of the technical report, which is half the deliverable |
| `08-hardware-es0.md` | Running this on 4x RTX 2080 Ti: fp16, LoRA, GPU split, day-0 checks |
| `09-short-plan.md` | **Active plan.** The 2.5-day cut; overrides scale and schedule above |
| `requirements.txt` | Dependencies |

## Start here

**Time-boxed version:** follow `09-short-plan.md`. It overrides the scale in `01-spec.md` and the
schedule in `06-schedule-and-risks.md`.

1. Read `01-spec.md` to confirm the project is still what you want, then `08-hardware-es0.md`.
2. Read `06-schedule-and-risks.md` and pick a start date.
3. Day 1 work is in `02-data-and-eval.md`. Build evaluation before anything else.

## The one rule

Evaluation comes first. If the eval harness and both single-shot baselines are not working by the
end of day 2, stop and fix that before touching RL. An unmeasurable experiment wastes the other
twelve days.
