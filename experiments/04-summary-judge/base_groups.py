"""Exp 04, before training: the prompted base model sampled like GRPO step 0 (temperature 1, G=8
per question, all 500 train questions), scored with the summary judge. Reports mean judge_score,
true recall and the share of groups with nonzero reward variance under run A, run B and run2's
reward. Also scores the existing greedy prompted-agent train episodes (promptdev_v2) and run2's
logged rollouts.

  nanosearch/serve.sh 8000 &  python -m experiments.04-summary-judge.base_groups  (via base_groups.sh)
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from nanosearch import DATA_DIR, OUT_DIR
from nanosearch.agent import MODEL, VLLMClient, score
from nanosearch.baselines import load_questions
from nanosearch.env import SearchEnv, judge_score, parse_turn, reward, unsupported_titles
from nanosearch.indexes import get_retriever, load_corpus

REWARDS = {  # w_step 0, as in run2's training
    "run2 (recall)": lambda r, j: r,
    "A (judge)": lambda r, j: j,
    "B (0.5 recall + 0.5 judge)": lambda r, j: 0.5 * (r + j),
}


def judge_rows(eps, docs):
    """eps: dicts with gold_doc_ids, turns ([{text, ...}]), retrieved (may be None)."""
    for e in eps:
        titles = [docs[d]["title"] for d in e["gold_doc_ids"]]
        e["judge"] = judge_score(e["turns"], titles)
        if e.get("retrieved") is not None:
            e["n_named"], e["n_unsupported"] = unsupported_titles(
                e["turns"], titles, [docs[d]["title"] for d in e["retrieved"]])
    return eps


def summary(eps, G=None):
    s = {"n": len(eps), "judge": float(np.mean([e["judge"] for e in eps])),
         "judge>0": float(np.mean([e["judge"] > 0 for e in eps]))}
    if "recall@9" in eps[0]:
        s["recall@9"] = float(np.mean([e["recall@9"] for e in eps]))
    if "n_named" in eps[0]:
        s["unsupported"] = sum(e["n_unsupported"] for e in eps) / max(1, sum(e["n_named"] for e in eps))
        s["titles_named"] = float(np.mean([e["n_named"] for e in eps]))
    if G:
        for name, f in REWARDS.items():
            r = np.array([reward(f(e["recall@9"], e["judge"]), e["format_valid"], e["n_searches"], 0)
                          for e in eps]).reshape(-1, G)
            s[f"groups_with_variance {name}"] = float((r.std(1) > 1e-6).mean())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", nargs="+", default=["http://localhost:8000"])
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--out_name", default="exp04_base_groups")
    args = ap.parse_args()
    out_dir = os.path.join(OUT_DIR, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    docs = {d["doc_id"]: d for d in load_corpus()}
    report = {}

    # existing outputs first (no GPU needed)
    v2 = [json.loads(l) for l in open(os.path.join(OUT_DIR, "promptdev_v2", "train_episodes.jsonl"))]
    report["prompted greedy, 200 train q (promptdev_v2)"] = summary(judge_rows(v2, docs))
    qs = {q["question"]: q for q in load_questions(os.path.join(DATA_DIR, "train.jsonl"))}
    ro = [json.loads(l) for l in open(os.path.join(OUT_DIR, "runs", "run2_wstep0", "rollouts.jsonl"))]
    for e in ro:
        e["gold_doc_ids"] = qs[e["question"]]["gold_doc_ids"]
        e["turns"] = [{"text": t, "action": parse_turn(t)[0]} for t in e["turns"]]
    judge_rows(ro, docs)
    report["run2 rollouts, step 0 (1 group)"] = summary([e for e in ro if e["step"] == 0])
    report["run2 rollouts, steps 125-149"] = summary([e for e in ro if e["step"] >= 125])

    # fresh step-0-like sample of the base model
    from transformers import AutoTokenizer
    env = SearchEnv(AutoTokenizer.from_pretrained(MODEL), get_retriever("bm25"))
    client = VLLMClient(args.urls, MODEL, temperature=1.0, seed=None)
    questions = list(qs.values())
    jobs = [q for q in questions for _ in range(args.G)]
    with ThreadPoolExecutor(args.workers) as pool:
        eps = list(pool.map(lambda q: env.run(q["question"], client), jobs))
    rows = []
    for q, e in zip(jobs, eps):
        rows.append({"qid": q["qid"], "question": q["question"], "gold_doc_ids": q["gold_doc_ids"],
                     **score(q, e, 0), **{k: e[k] for k in ("retrieved", "queries", "turns", "end",
                                                          "format_valid", "n_searches")}})
    judge_rows(rows, docs)
    report[f"base model, temp 1, 500 train q x G={args.G}"] = summary(rows, args.G)
    with open(os.path.join(out_dir, "episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
