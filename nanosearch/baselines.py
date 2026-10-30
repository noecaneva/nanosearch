"""Single-shot retrieval evaluation at equal budget (k=9) on the eval questions.

Systems:
  bm25    raw question -> BM25 top 9
  dense   raw question -> bge-small + FAISS flat top 9
  oracle  the question's gold passages (checks the gold mapping and the metric code; ~1.0)

One JSON summary per system plus a JSONL with every episode (question, retrieved ids, gold ids,
per-question metrics), written to $GRPOP_ROOT/outputs/baselines/.

  python -m nanosearch.baselines --systems bm25 dense oracle
"""

import argparse
import json
import math
import os
import time

import numpy as np

from nanosearch import DATA_DIR, OUT_DIR
from nanosearch.indexes import DEFAULT_INDEX_DIR, get_retriever

K = 9


def load_questions(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def recall(retrieved, gold):
    return len(set(retrieved) & set(gold)) / len(gold)


def ndcg(retrieved, gold):
    gold = set(gold)
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(retrieved) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), len(retrieved))))
    return dcg / idcg if idcg else 0.0


class OracleRetriever:
    def __init__(self, questions, index_dir=DEFAULT_INDEX_DIR):
        with open(os.path.join(index_dir, "doc_ids.json")) as f:
            valid = set(json.load(f))
        self.gold = {q["question"]: [d for d in q["gold_doc_ids"] if d in valid]
                     for q in questions}

    def search(self, query, k):
        return self.gold[query][:k]


def evaluate(system, retriever, questions, k=K):
    episodes, latencies = [], []
    for q in questions:
        t0 = time.perf_counter()
        retrieved = retriever.search(q["question"], k)
        latencies.append(time.perf_counter() - t0)
        assert len(retrieved) <= k
        episodes.append({
            "qid": q["qid"], "question": q["question"], "type": q["type"],
            "gold_doc_ids": q["gold_doc_ids"], "retrieved": retrieved,
            "recall@9": recall(retrieved, q["gold_doc_ids"]),
            "ndcg@9": ndcg(retrieved, q["gold_doc_ids"]),
            "all_gold@9": float(set(q["gold_doc_ids"]) <= set(retrieved)),
            "recall@3": recall(retrieved[:3], q["gold_doc_ids"]),
        })

    def agg(eps):
        return {m: round(float(np.mean([e[m] for e in eps])), 4)
                for m in ("recall@9", "ndcg@9", "all_gold@9", "recall@3")}

    summary = {
        "system": system, "k": k, "n_questions": len(questions), **agg(episodes),
        "by_type": {t: {"n": sum(e["type"] == t for e in episodes),
                        **agg([e for e in episodes if e["type"] == t])}
                    for t in sorted({e["type"] for e in episodes})},
        "mean_searches": 1.0, "tokens_generated": 0,
        "latency_ms_p50": round(1000 * float(np.percentile(latencies, 50)), 3),
        "latency_ms_p95": round(1000 * float(np.percentile(latencies, 95)), 3),
    }
    return summary, episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["bm25", "dense", "oracle"])
    ap.add_argument("--split", default="eval")
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", default=os.path.join(OUT_DIR, "baselines"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    questions = load_questions(os.path.join(args.data_dir, f"{args.split}.jsonl"))
    for system in args.systems:
        if system == "oracle":
            retriever = OracleRetriever(questions, args.index_dir)
        elif system == "dense":
            retriever = get_retriever(system, args.index_dir, device=args.device)
        else:
            retriever = get_retriever(system, args.index_dir)
        summary, episodes = evaluate(system, retriever, questions)
        with open(os.path.join(args.out_dir, f"{system}_{args.split}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(args.out_dir, f"{system}_{args.split}_episodes.jsonl"), "w") as f:
            for e in episodes:
                f.write(json.dumps(e) + "\n")
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
