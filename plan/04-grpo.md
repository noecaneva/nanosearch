# 04. GRPO, Implemented From Scratch

This is the core of the project. Roughly 300 to 400 lines once the environment exists. Two
implementation details below decide whether it learns at all, and both are easy to get wrong in a
way that produces a plausible-looking loss curve and no learning.

## The algorithm

GRPO removes the value network from PPO and replaces the learned baseline with the mean reward of a
group of rollouts sampled from the same prompt. For a retrieval task with a cheap objective reward,
that is a good trade: no critic to train, no critic to debug.

For each prompt q:

1. Sample G rollouts from the current policy, which lives in vLLM. Call it pi_old.
2. Score each rollout with the reward function from `05-reward.md`, giving r_1 through r_G.
3. Compute the group-relative advantage, one scalar per rollout:

   ```
   A_i = (r_i - mean(r)) / (std(r) + eps)
   ```

   with eps around 1e-4. Every token generated in rollout i receives the same advantage A_i.

4. Take an optimizer step on the clipped surrogate objective:

   ```
   rho_{i,t} = pi_theta(o_{i,t} | q, o_{i,<t}) / pi_old(o_{i,t} | q, o_{i,<t})

   J = (1/G) * sum_i (1 / |o_i|) * sum_t min(
           rho_{i,t} * A_i,
           clip(rho_{i,t}, 1 - eps_low, 1 + eps_high) * A_i
       )
       - beta * KL(pi_theta || pi_ref)
   ```

5. Push the updated weights into the vLLM server. Repeat.

Start with eps_low = eps_high = 0.2. Raising eps_high alone, to something like 0.28, is a known
trick for keeping entropy up on reasoning tasks and is worth an ablation if the policy collapses to
a single query template.

## Detail one: mask the environment tokens

The trajectory is interleaved. It looks like this:

```
[prompt] [model: <think>..</think><search>q1</search>] [env: <result>..</result>]
         [model: <think>..</think><search>q2</search>] [env: <result>..</result>]
         [model: <done/>]
```

**Only model-generated tokens get loss.** Environment tokens sit in the context and condition
everything that follows, but they were not produced by the policy, so they carry no gradient and
must not appear in the ratio, in the KL, or in the length normalization |o_i|.

Build an explicit boolean mask alongside the token ids as you assemble the trajectory. Do not try to
recover it afterward by re-tokenizing and string-matching the result blocks, which breaks on
tokenizer boundaries in ways that are maddening to find.

Getting this wrong trains the model to predict retrieved documents. The loss goes down. Nothing
improves. This is the single most common bug in multi-turn RL implementations and it is worth a
unit test: assert that the masked token count equals the sum of the per-turn generated lengths
reported by vLLM.

## Detail two: groups with zero variance

When all G rollouts receive the same reward, the standard deviation is zero, every advantage is
zero, and the group contributes no gradient while costing full compute.

With two gold paragraphs and three searches, recall takes values in {0, 0.5, 1}, so this will happen
constantly. Early in training most groups score 0. Later, many score 1. Both regimes are dead.

Two mitigations, and you want both:

- **Filter degenerate groups.** Drop any group whose reward standard deviation is below a small
  threshold before computing the loss. Oversample prompts per step to keep the effective batch size
  stable. This is what DAPO calls dynamic sampling.
- **Densify the reward.** The shaping terms in `05-reward.md` exist partly for this reason. Format
  validity and step cost are continuous enough to break ties inside a group that is otherwise
  uniform on recall.

Log the fraction of groups filtered every step. If it exceeds roughly 70 percent for a sustained
period, the reward is too coarse for the current policy and you should fix the reward rather than
wait it out.

## Reference policy and KL

Keep a frozen copy of the initial model as pi_ref. Use the k3 estimator, which is unbiased and
non-negative:

```
KL_est = exp(log pi_ref - log pi_theta) - (log pi_ref - log pi_theta) - 1
```

Start with beta around 0.001 to 0.01. A great deal of recent work drops the KL term entirely for
tasks with verifiable rewards and reports better results, so **running with beta = 0 is a planned
ablation, not an accident.** Decide it by measurement.

## Length normalization

The `1 / |o_i|` term above is the original formulation and it introduces a length bias: tokens in
short rollouts get more weight each than tokens in long ones. The alternative is to normalize by
total token count across the group instead of per rollout.

At three-turn episodes with a 256-token cap the effect is small, but note which one you implemented,
because the choice changes what the loss optimises.

## Precision note for es0

On RTX 2080 Ti there is no bf16, so everything runs in fp16 and the ratio computation becomes
numerically delicate. Cast logits to fp32 before `log_softmax`, always. See `08-hardware-es0.md`.

## Weight synchronization

After each optimizer step the vLLM server holds stale weights. Options, in increasing order of
effort:

1. Restart the server each step. Simple, correct, far too slow.
2. Write weights to disk and reload. Still slow, but debuggable, and fine for the first smoke test.
3. Use vLLM's in-process weight update path to push tensors directly. This is what you want, and it
   is where the time will go. With LoRA you are syncing adapter tensors only, roughly 18M
   parameters, which makes this materially cheaper than a full-weight sync.

Budget a full day for this. It is the most likely thing to eat days 7 and 8, and it is also the part
that proves you have actually used vLLM rather than read about it.

## Minimum smoke test before any real run

Run the whole loop on 16 prompts with G = 4 for 10 steps and confirm all of the following:

- The masked token count matches vLLM's reported generation lengths.
- Advantages within a group sum to approximately zero.
- With beta = 0 and a constant reward, the gradient is approximately zero.
- Reward on those 16 prompts goes up. If a model cannot overfit 16 prompts, nothing else matters.

Do not start a real run until all four pass.

## What to log to wandb

Reward mean and standard deviation, the four reward components separately, fraction of groups
filtered, format validity rate, mean searches per episode, mean generated tokens, KL to reference,
policy entropy, gradient norm, and the clipped fraction of the ratio.

Policy entropy and the clipped fraction are the two that tell you the run has quietly died before
the reward curve does.
