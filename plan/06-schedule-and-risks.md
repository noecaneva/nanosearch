# 06. Fourteen Days, With Kill Criteria

Each day has a deliverable you can point at. Three checkpoints have explicit kill criteria, because
the failure mode of a project like this is spending eleven days on training code and two days
discovering the evaluation was wrong.

## Day 0: setup

Cluster allocation confirmed, environment installed, vLLM serving the base model and answering a
test prompt, wandb project created, empty repo initialized with the license and a README.

## Days 1 to 2: data and evaluation

Corpus sampled, deduplicated and written to JSONL with counts recorded. BM25 and dense indexes
built. Evaluation harness runs end to end. Both single-shot baselines produce a metrics JSON.
Reward function calibrated offline against both baselines and an oracle.

**Checkpoint 1, end of day 2.** BM25 and dense recall@9 are both computed, and the oracle ceiling is
approximately 1.0.

**Kill criteria.** If BM25 recall@9 is above roughly 0.85, the task is too easy: add distractors or
sample harder questions. If the oracle ceiling is below 0.95, gold-paragraph matching is broken.
Fix either before proceeding. Do not proceed with a broken measurement.

## Days 3 to 4: environment and prompted baseline

Environment implemented with the tag protocol, episode limits, and malformed-action handling.
Rollouts generated through vLLM. The prompted, untrained agent runs on the eval set and produces its
metrics JSON, which is the third baseline.

**Checkpoint 2, end of day 4.** Three of the four baselines are complete, and you know the format
validity rate of the untrained 1.5B model.

**Kill criteria.** If format validity for the prompted agent is below roughly 60 percent even with
few-shot examples in the prompt, the model is too small for this action space. Respond by
simplifying the action space further, not by moving to a larger model, since small scale is a
deliberate constraint of this project. Note the decision in the write-up.

## Days 5 to 6: GRPO implementation

Advantage computation, masked loss, KL estimator, weight synchronization into vLLM. The four-part
smoke test from `04-grpo.md` passes. Overfitting 16 prompts is demonstrated.

**Checkpoint 3, end of day 6.** The model can overfit 16 prompts. Reward on those 16 rises and the
masking unit test passes.

**Kill criteria.** If you cannot overfit 16 prompts by end of day 7, stop adding features and debug.
The bug is almost certainly the token mask or the weight synchronization. Nothing downstream can
work until this does.

## Days 7 to 10: training and reward iteration

Real runs on the full training set. Expect most of this window to be reward iteration rather than
clean training curves. Keep every run in wandb with a readable name, and keep a dated log of what
you changed and why. That log is the raw material for the most interesting section of the report.

By end of day 10 you want one run you are willing to call the main result, even if it is a negative
one.

## Days 11 to 12: ablations

In priority order, stopping when time runs out:

1. Model size: 0.5B against 1.5B, same everything else.
2. Selection action space against union, from `05-reward.md`.
3. KL coefficient beta = 0 against beta > 0.
4. Step penalty on against off, to quantify the speed and accuracy tradeoff.

Two done properly beats four done carelessly.

## Days 13 to 14: write-up

Follow `07-writeup-outline.md`. Repo cleaned, README written so a stranger can reproduce the result,
report finished.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| vLLM weight sync eats days | High | High | Budget a full day. Fall back to disk reload, which is slow but correct, and note it as a limitation |
| Token masking bug | High | Fatal if undetected | Unit test on day 5, before any real run |
| Model too small to learn reformulation | Medium | Medium | This is a legitimate result. Checkpoint 2 catches it early, and the 0.5B ablation turns it into a finding |
| Zero-variance groups stall training | High | High | Dynamic filtering plus shaping terms, both specified in `04-grpo.md`. Log the filter rate from step one |
| Reward hacking eats the result | High | Medium | Expected and budgeted. The hacking register in `05-reward.md` is a deliverable, not an obstacle |
| Cluster queue times | Medium | Medium | Develop against a tiny subset locally. Reserve the cluster for real runs |
| Scope creep into vision or larger models | Medium | High | Explicitly out of scope in `01-spec.md`. Write the idea down, do not act on it |

## The rule for when it goes wrong

A negative result, diagnosed, is a good outcome for this project. A project abandoned at day 9
because training would not converge is not. If by day 10 the trained model does not beat the
prompted agent, freeze it, run the ablations against the failure, and write the report about why.
That report is still evidence of everything it needs to be evidence of.
