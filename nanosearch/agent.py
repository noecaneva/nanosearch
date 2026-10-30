"""Run the search agent through vLLM on a question split and score it like the baselines.

The same code evaluates the prompted (untrained) agent and, with --lora, a trained adapter.
Writes a JSON summary and a JSONL with every episode (full transcript, queries, retrieved ids,
metrics, reward) to $GRPOP_ROOT/outputs/<out_name>/.

  nanosearch/serve.sh 8000 &          # on a GPU node
  python -m nanosearch.agent --out_name agent_prompted
"""

import argparse
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from nanosearch import DATA_DIR, OUT_DIR
from nanosearch.baselines import K, load_questions, ndcg, recall
from nanosearch.env import STOP, W_STEP, SearchEnv, reward
from nanosearch.indexes import get_retriever

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class VLLMClient:
    """generate(prompt_ids, max_tokens) -> token ids, round-robin over one or more servers."""

    def __init__(self, urls, model, temperature=0.0, seed=0):
        self.urls = itertools.cycle(urls)
        self.model, self.temperature, self.seed = model, temperature, seed
        self.session = requests.Session()

    def __call__(self, prompt_ids, max_tokens):
        req = {"model": self.model, "prompt": prompt_ids, "max_tokens": max_tokens,
               "temperature": self.temperature, "stop": STOP,
               "include_stop_str_in_output": True, "return_token_ids": True,
               "skip_special_tokens": False}
        if self.seed is not None:  # None when sampling groups: a fixed seed makes all G identical
            req["seed"] = self.seed
        r = self.session.post(f"{next(self.urls)}/v1/completions", json=req, timeout=600)
        r.raise_for_status()
        out = r.json()
        ids = out["choices"][0]["token_ids"]
        # The env's loss mask is built from these ids: they must be exactly what vLLM generated.
        assert len(ids) == out["usage"]["completion_tokens"], (len(ids), out["usage"])
        return ids


def score(q, ep, w_step=W_STEP):
    gold = q["gold_doc_ids"]
    r = recall(ep["retrieved"], gold)
    return {"recall@9": r, "ndcg@9": ndcg(ep["retrieved"], gold),
            "all_gold@9": float(set(gold) <= set(ep["retrieved"])),
            "recall@3": recall(ep["retrieved"][:3], gold),
            "reward": reward(r, ep["format_valid"], ep["n_searches"], w_step)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", nargs="+", default=["http://localhost:8000"])
    ap.add_argument("--lora", default=None, help="name of a LoRA adapter loaded in vLLM")
    ap.add_argument("--lora_path", default=None, help="load this adapter dir into vLLM first")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out_name", default="agent_prompted")
    ap.add_argument("--separate_results", action="store_true",
                    help="result block in its own user message (default: inline, core-v1)")
    ap.add_argument("--no_reminder", action="store_true", help="drop the reminder after results")
    ap.add_argument("--system_note", default="",
                    help="sentence appended to the system prompt (default none: core-v1)")
    args = ap.parse_args()
    out_dir = os.path.join(OUT_DIR, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    if args.lora_path:
        args.lora = os.path.basename(os.path.normpath(args.lora_path)) + "-eval"
        for url in args.urls:
            requests.post(f"{url}/v1/load_lora_adapter", json={
                "lora_name": args.lora, "lora_path": args.lora_path}).raise_for_status()
    from transformers import AutoTokenizer
    env = SearchEnv(AutoTokenizer.from_pretrained(MODEL), get_retriever("bm25"),
                    separate_results=args.separate_results, reminder=not args.no_reminder,
                    system_note=args.system_note)
    generate = VLLMClient(args.urls, args.lora or MODEL)
    questions = load_questions(os.path.join(DATA_DIR, f"{args.split}.jsonl"))[:args.limit]

    def play(q):
        t0 = time.perf_counter()
        ep = env.run(q["question"], generate)
        ep["latency"] = time.perf_counter() - t0
        return ep

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        eps = list(pool.map(play, questions))
    wall = time.time() - t0

    rows = []
    for q, ep in zip(questions, eps):
        assert len(ep["retrieved"]) <= K
        rows.append({"qid": q["qid"], "question": q["question"], "type": q["type"],
                     "gold_doc_ids": q["gold_doc_ids"], **score(q, ep),
                     **{k: ep[k] for k in ("retrieved", "queries", "turns", "end",
                                           "format_valid", "n_searches", "n_generated",
                                           "latency")}})

    def agg(rs):
        return {m: round(float(np.mean([r[m] for r in rs])), 4)
                for m in ("recall@9", "ndcg@9", "all_gold@9", "recall@3", "reward",
                          "format_valid", "n_searches", "n_generated")}

    ends = {}
    for r in rows:
        ends[r["end"]] = ends.get(r["end"], 0) + 1
    summary = {
        "system": args.out_name, "model": MODEL, "lora": args.lora_path or args.lora, "k": K,
        "separate_results": args.separate_results, "reminder": not args.no_reminder,
        "system_note": args.system_note,
        "n_questions": len(rows), **agg(rows),
        "by_type": {t: {"n": sum(r["type"] == t for r in rows),
                        **agg([r for r in rows if r["type"] == t])}
                    for t in sorted({r["type"] for r in rows})},
        "ends": dict(sorted(ends.items())),
        "latency_s_p50": round(float(np.percentile([r["latency"] for r in rows], 50)), 3),
        "latency_s_p95": round(float(np.percentile([r["latency"] for r in rows], 95)), 3),
        "wall_s": round(wall, 1),
    }
    with open(os.path.join(out_dir, f"{args.split}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, f"{args.split}_episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
