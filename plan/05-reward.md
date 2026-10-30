# 05. Reward Design

The reward is where your judgment shows, and where the model will spend its effort trying to cheat
you. Expect to rewrite this file during the project. That is not a failure, it is the work.

## Terms

```
R = w_recall * recall
  + w_format * format_valid
  - w_step   * n_searches
  - w_len    * length_overrun
```

| Term | Range | Starting weight | Purpose |
|---|---|---|---|
| `recall` | {0, 0.5, 1} | 1.0 | Fraction of gold paragraphs in the final set. The actual objective |
| `format_valid` | {0, 1} | 0.2 | Every action parsed cleanly and the episode terminated properly |
| `n_searches` | {0, 1, 2, 3} | 0.05 | Cost of searching. Makes the speed axis a trained behaviour, not an accident |
| `length_overrun` | >= 0 | 0.05 | Generated tokens beyond a soft cap, normalized. Suppresses rambling |

Start the step penalty low. Setting it too high early teaches the model to emit `<done/>`
immediately, which scores a reliable zero on recall and is very hard to climb back out of.

## Why format reward is weighted this high at this model size

At 1.5B, and certainly at 0.5B, a meaningful fraction of early episodes will fail to parse. If the
only reward signal is recall, and recall is zero for every rollout because every rollout is
malformed, there is no gradient anywhere and training never starts. The format term gives the model
something to climb before it can do anything useful.

Once format validity is above roughly 95 percent and stable, consider annealing `w_format` down so
that late training optimizes what you actually care about. Log the rate so you can see when.

## The hacking register

Fill this in as it happens. These are the ones to expect, based on how the reward is shaped.

| Expected hack | Mechanism | Detection | Mitigation |
|---|---|---|---|
| Search-everything | Always use all 3 searches regardless of need, since recall is monotone in documents retrieved | `mean_searches` pinned at 3 while recall is flat | Raise `w_step`, or switch to an explicit selection action, see below |
| Query stuffing | Emit one enormous query containing the whole question plus padding, hoping BM25 matches more | Mean query length climbs steadily | Cap query length in the environment and reject overlong queries as malformed |
| Question echo | Collapse to always issuing the question verbatim as the query, for all three searches | Query diversity within an episode goes to zero | This is a real finding, not just a hack. It means the model has not learned reformulation, which is the whole thesis |
| Format farming | Emit a valid episode that does nothing useful, collecting the format reward | Format rate near 1.0, recall near 0, searches near 0 | Anneal `w_format`, and make the format reward conditional on at least one search |
| Tag mimicry | Model generates its own fake `<result>` block to condition on | Contains `<result>` inside a model-generated span | Stop generation on the literal string `<result>` and treat it as malformed |

The last one is worth guarding against in code from day one, because it is silent: the model
hallucinates documents, conditions on them, and the episode looks structurally fine.

## The structural alternative

The search-everything hack exists because the final set is the union of everything retrieved, so
more searching is weakly better. There is a cleaner design where the model must explicitly commit:

```
<select>1,4,7</select>
```

choosing at most 5 passages from everything it has seen. Recall is then computed on the selection,
not the union, and searching more no longer helps by itself.

This is better science and worse odds for a 1.5B model, which has to track document indices across
turns. **Plan: run the union design first, and hold the selection design as the primary ablation.**
If the union design produces the search-everything behaviour, the selection experiment becomes the
most interesting result in the report rather than a footnote.

## Calibrate before training

Compute the reward for all four baseline systems offline before any RL runs. You need to know what
BM25 scores under this reward function, and what a perfect oracle scores.

If BM25 already scores near the ceiling, the task is too easy and the corpus needs more distractors.
If the oracle ceiling is far below 1.0, there is a bug in gold-paragraph matching. Finding either of
those on day 2 costs an hour. Finding them on day 12 costs the project.
