"""Exp 07 day-0 check 3 (+ the batch for check 2): the prompted model sampled like GRPO step 0
(temperature 1, G=8 per question, all 500 train questions, core harness, run2 reward w_step 0).
Reports the share of groups with nonzero reward variance (below 5 % = no GRPO signal, stop), mean
recall@9, format validity, ends, truncated and <think> turns. Writes the first --batch_prompts
groups with ids / mask / advantage to train_batch.jsonl for day0_train_step.py.

  PYTHONPATH=. python experiments/07-qwen3/day0_groups.py --model Qwen/Qwen3-1.7B   (via day0.sh)
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from nanosearch import DATA_DIR, OUT_DIR
from nanosearch.agent import MODEL, VLLMClient, score, turn_counts
from nanosearch.baselines import load_questions
from nanosearch.env import SearchEnv
from nanosearch.grpo import advantages
from nanosearch.indexes import get_retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", nargs="+", default=["http://localhost:8000"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--batch_prompts", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="train questions (default all 500)")
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--out_name", default="exp07_day0_groups")
    args = ap.parse_args()
    out_dir = os.path.join(OUT_DIR, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    from transformers import AutoTokenizer
    env = SearchEnv(AutoTokenizer.from_pretrained(args.model), get_retriever("bm25"))
    client = VLLMClient(args.urls, args.model, temperature=1.0, seed=None)
    questions = load_questions(os.path.join(DATA_DIR, "train.jsonl"))[:args.limit]
    jobs = [q for q in questions for _ in range(args.G)]
    with ThreadPoolExecutor(args.workers) as pool:
        eps = list(pool.map(lambda q: env.run(q["question"], client), jobs))
    scores = [score(q, e, 0) for q, e in zip(jobs, eps)]  # run2 recipe: --w_step 0
    adv, keep = advantages([s["reward"] for s in scores], args.G)

    rows = []
    for q, e, s, a in zip(jobs, eps, scores, adv):
        e["adv"] = float(a)
        rows.append({"qid": q["qid"], "question": q["question"], "gold_doc_ids": q["gold_doc_ids"],
                     **s, "adv": float(a), **turn_counts(e),
                     **{k: e[k] for k in ("retrieved", "queries", "turns", "end", "format_valid",
                                          "n_searches", "n_generated")}})
    n_turns = sum(len(r["turns"]) for r in rows)
    ends = {}
    for r in rows:
        ends[r["end"]] = ends.get(r["end"], 0) + 1
    report = {
        "model": args.model, "n_questions": len(questions), "G": args.G, "n_episodes": len(rows),
        **{m: round(float(np.mean([r[m] for r in rows])), 4)
           for m in ("recall@9", "all_gold@9", "recall@3", "reward", "format_valid", "n_searches",
                     "n_generated")},
        "groups_with_variance": round(float(keep.reshape(-1, args.G)[:, 0].mean()), 4),
        "n_turns": n_turns,
        "truncated_turns": int(sum(r["truncated_turns"] for r in rows)),
        "think_turns": int(sum(r["think_turns"] for r in rows)),
        "ends": dict(sorted(ends.items())),
        "max_episode_tokens": int(max(len(e["ids"]) for e in eps)),
        "mean_model_tokens": round(float(np.mean([sum(e["mask"]) for e in eps])), 1),
    }
    with open(os.path.join(out_dir, "episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_batch = args.batch_prompts * args.G
    with open(os.path.join(out_dir, "train_batch.jsonl"), "w") as f:
        for e, k in zip(eps[:n_batch], keep[:n_batch]):
            if k:
                f.write(json.dumps({"ids": e["ids"], "mask": e["mask"], "adv": e["adv"]}) + "\n")
    report["train_batch_episodes"] = int(keep[:n_batch].sum())
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
