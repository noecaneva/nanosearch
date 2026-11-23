# Experiment 08: building BM25 one idea at a time

**Status: planned 2026-11-23, not run.** No GPU work, no training. The predictions below are fixed
before the job is submitted and are not edited once numbers are in.

**Question.** Our BM25 baseline (recall@9 0.825 on the 200 eval questions) is one point in a space
of lexical scorers. Which of its ideas earn the recall on this corpus: counting term matches, term
frequency, length normalisation, sublinear or saturating TF, IDF? Building up from plain term
overlap shows what each idea adds. It also shows how much of what the search agent works with comes
from the engine rather than from the queries.

## The ladder

Each rung adds one idea. All rungs share the tokenisation of the core BM25 index (`bm25s.tokenize`,
English stopwords, Snowball stemmer, passages as "title. text"). Only the scoring changes.
score(q, d) = Σ over query tokens t of w(t, d). Repeated query tokens count twice, as in bm25s.

| rung | scorer | w(t, d) | idea added |
|---|---|---|---|
| L0 | term overlap | 1[tf > 0] | a query term is present or not |
| L1 | raw tf | tf | repeating a term counts as more evidence |
| L1s | raw tf, stopwords kept | tf | side branch: how much stopword removal already does IDF's job |
| L2 | tf / doc length | tf / (\|d\|/avgdl) | long documents match by chance |
| L3 | sublinear tf | log(1 + tf) | the tenth mention adds less than the first |
| L4 | idf of matched terms | idf(t) · 1[tf > 0] | rare terms are worth more |
| L5 | tf-idf, cosine | log(1 + tf) · idf, doc vector L2-normalised | classic vector space model |
| L6 | saturating tf × idf | idf · tf / (tf + k1) | TF saturates (BM25 with b = 0) |
| L7 | BM25 | idf · tf' / (tf' + k1), tf' = tf / (1 − b + b·\|d\|/avgdl) | pivoted length normalisation |

idf = log(1 + (N − df + 0.5)/(df + 0.5)), k1 = 1.5, b = 0.75: the bm25s Lucene defaults the core
index was built with.

**The ladder is one path through a grid.** The scorer has three axes: TF transform (binary, raw,
log, sat), IDF (none, lucene) and length norm (b = 0, 0.75, 1, or cosine). The run scores all
4 × 2 × 4 = 32 cells. Length norm is applied by dividing tf by the pivot *before* the TF transform,
so b = 0.75 on `sat` is exactly BM25 and b = 1 on `raw` ranks like tf/|d|. The grid shows interactions
the ladder order would hide, e.g. length norm helping only once IDF is on.

**Ties.** L0 and L4 give integer-like scores, so many documents tie. Ties are broken by corpus
order. Each config reports `boundary_tie_rate`, the share of questions where ranks 9 and 10 tie
(the top-9 set is then partly decided by corpus order), and `median_docs_tied_at_rank9`. A rung with
a high tie rate is read with that caveat.

## Sanity check (must pass before any other number is read)

`ladder.py check` scores the 200 eval questions with the L7 cell and compares against the stored
core-v1 bm25s index: max |score difference| < 1e-3, and every top-9 set identical except where the
scores tie at rank 9. `run` calls it first and exits on failure. L7 must then report recall@9 0.825,
the RESULTS.md BM25 row. If either fails, the bug is in the ladder, not in bm25s: fix it, log it,
rerun.

## Predictions, recorded before running

1. L0 is far below BM25 (recall@9 under 0.70). Most of the signal is in *which* terms match, not
   how many.
2. L4 (IDF alone, no TF) recovers most of the gap: within 0.05 of BM25. HotpotQA questions name
   entities, and entity names are rare terms.
3. L1 (raw tf) is no better than L0, and may be worse, because a passage repeating one common term
   wins. L1s (stopwords kept) is clearly worse than L1.
4. Length normalisation matters little on this corpus. HotpotQA paragraphs are short and similar in
   length, so |L7 − L6| < 0.02.
5. Comparison questions saturate early (both entities are named in the question); bridge questions
   carry the differences between rungs.

These are the predictions of the spec. Each one is reported as held or not, with the numbers.

## Run

```
sbatch -o $SCRATCH/grpop/logs/%x-%j.out experiments/08-lexical-ladder/run.sh
```

One es_debug job, CPU only (the GPU is requested only because es0 jobs are GPU jobs), well under
30 min. It runs the selftest (toy corpus), then for the `eval` and `train` splits: the BM25 check,
all 33 configs (32 grid cells + L1s), and the tables.

- Per config: `$GRPOP_ROOT/outputs/exp08/<config>_<split>.json` (summary in the same format as
  `nanosearch/baselines.py`, plus the config, tie rate and tied-doc count) and `..._episodes.jsonl`.
- Tables: `experiments/08-lexical-ladder/out/tables_<split>.md` (ladder with paired 10k-bootstrap
  Δ recall@9 vs the previous rung and vs BM25, and the grid) and `out/summary_<split>.json`. Commit
  `out/`.
- L1s is a side branch. Its Δ is against L1, and L2 steps from L1.

Code: `ladder.py` (scorers, grid runner, check, selftest; reuses `baselines.evaluate` so the metrics
are the core ones), `analyze.py` (tables). The core (`nanosearch/`) is not touched. The selftest
passed locally on bm25s 0.3.16, as did an end-to-end `run` + `check` on a 400-doc fake core-v1.

## After the run

1. The Results section here: the ladder table on eval, the grid, train as a robustness line, each
   prediction held or not.
2. An Exp 08 section in `RESULTS.md`, a bullet in the top-level README's Experiments list, and a
   dated entry in the handoff LOG (job id, runtime, check output).
3. Optional, only if the eval numbers make it interesting: a k1 ∈ {0.5, 0.9, 1.2, 1.5, 2.0} ×
   b ∈ {0.3, 0.5, 0.75, 0.9} sweep at L7, **on train only**. Tuning on eval would be selection on
   the test set.

## Limitations

200 eval questions, so a paired difference under ~0.03 recall@9 is within noise. One corpus, built
from the questions' own paragraphs plus distractors, which favours lexical matching. One tokeniser.
Stemming and stopwords are held fixed, apart from the single L1s branch. Query-side weighting is
fixed to raw query-term counts, as in bm25s.
