# 02. Data, Index, and Evaluation

Build this first. Days 1 and 2 produce nothing but a corpus, two indexes, an evaluation harness and
two baselines. No RL code exists yet at the end of day 2, and that is correct.

## Datasets

**Training and in-distribution evaluation: HotpotQA (distractor setting).** Every question carries
labelled gold supporting paragraphs, usually two, which is what makes a cheap objective reward
possible. Questions are multi-hop by construction. Available through the HuggingFace `datasets`
hub.

**Out-of-distribution evaluation: MuSiQue.** Harder multi-hop composition, different construction
process. Held out entirely, never trained on. If MuSiQue proves awkward to align to the same corpus
format, 2WikiMultihopQA is an acceptable substitute; the requirement is only that it was built by a
different process than the training set.

Both are research datasets with permissive licensing for this use. Record which revision you pulled
and when, because the write-up needs it.

## Corpus construction

The corpus is the union of gold and distractor paragraphs across a sampled subset of questions, not
all of Wikipedia.

1. Sample 3,000 training questions and 500 in-distribution eval questions from HotpotQA, plus 200
   from MuSiQue.
2. Take every paragraph attached to those questions, gold and distractor alike.
3. Deduplicate on normalized title plus first sentence. Expect roughly 30,000 unique passages, down
   from about 37,000 before deduplication.
4. Store as JSONL with fields: `doc_id`, `title`, `text`, and a boolean noting whether it is gold
   for any sampled question.

Keeping the distractors is the point. They are what make the search problem non-trivial at this
corpus size. Do not build a corpus of gold passages only, which would make every baseline look
excellent and the experiment meaningless.

Write down the dedup rule and the final counts. They belong in the write-up, and a reviewer who
cannot reproduce the corpus cannot check anything else.

## Indexes

**Sparse: BM25.** Use `bm25s`, which is fast and pure Python at this corpus size. At 30,000
passages an index builds in seconds and queries in under a millisecond.

**Dense: a small sentence embedder plus FAISS.** `bge-small-en-v1.5` or `e5-small-v2` through
`sentence-transformers`, with a FAISS flat index. At 30,000 passages a flat index is exact and
instant, so there is no reason to approximate and no ANN recall confound to explain away.

Both indexes serve the same retrieval interface: `search(query, k) -> list[doc_id]`. The
environment and the baselines all call that one function, which is what keeps the comparison fair.

## Metrics

Gold supporting paragraphs per question are usually two, so recall is coarse by nature. Report all
of these, not just the headline.

| Metric | Definition | Why it is here |
|---|---|---|
| `recall@9` | Fraction of gold paragraphs present in the final set | Primary. 9 is the agent's maximum budget, so every system gets 9 |
| `nDCG@9` | Rank-aware quality | Catches systems that find gold documents but bury them |
| `mean_searches` | Average search calls per episode | The cost side of the tradeoff |
| `latency_p50`, `latency_p95` | Wall clock per question | Search agents are judged as much on speed as accuracy |
| `tokens_generated` | Mean generated tokens per episode | The other cost axis |
| `format_valid_rate` | Fraction of episodes with well-formed actions throughout | At 1.5B this is a real number, not a formality |

**Equal budget is non-negotiable.** The agent can retrieve at most 9 documents, so the single-shot
baselines retrieve exactly 9. Comparing an agent with 9 documents against a baseline with 3 would
be a rigged result and any reviewer will spot it immediately.

## The four baselines

| System | Description |
|---|---|
| BM25 single-shot | Query is the raw question. Top 9. The baseline everyone skips and then gets asked about |
| Dense single-shot | Same, through the embedder. This is the "embedding-only" number agentic search is measured against |
| Prompted agent, untrained | The same base model, same environment, few-shot prompted. Isolates what RL adds over prompting |
| GRPO agent | The result |

The third baseline is the one that determines whether the project says anything. Without it, an
improvement over single-shot retrieval proves only that iteration helps, which is already known.
With it, you can state what the training contributed.

## Evaluation harness

One script, one command, one JSON output per system. It must be runnable against any of the four
systems without modification, and it must be deterministic given a seed. Fix the seed, fix the
question order, and fix the decoding temperature for evaluation at 0.

Log every episode in full: the question, each query issued, the documents returned, the final set,
and the computed reward components. Aggregate metrics are for the table; episode logs are what you
actually debug from, and they are where the reward-hacking section of the write-up comes from.
