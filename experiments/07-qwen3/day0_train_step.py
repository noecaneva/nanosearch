"""Exp 07 day-0 check 2: one real trainer step on one GPU with the batch from day0_groups.py.
Reports peak memory (must fit 10.5 GiB), whether every fp32 log-prob and every gradient is finite
(Qwen3 was trained in bf16, we run fp16 weights), the train_on stats, and saves the updated LoRA
adapter for check 1 (vLLM serves it).

  PYTHONPATH=. python experiments/07-qwen3/day0_train_step.py --model Qwen/Qwen3-1.7B   (via day0.sh)
"""

import argparse
import json
import os
import time

import torch

from nanosearch import OUT_DIR
from nanosearch.grpo import MODEL, build_policy, token_logprobs, train_on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--minibatches", type=int, default=2)
    ap.add_argument("--batch", default=os.path.join(OUT_DIR, "exp07_day0_groups", "train_batch.jsonl"))
    ap.add_argument("--out_name", default="exp07_day0_step")
    args = ap.parse_args()
    out_dir = os.path.join(OUT_DIR, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    eps = [json.loads(l) for l in open(args.batch)]
    t0 = time.time()
    model, params, opt = build_policy(args.lr, args.model)
    t_load = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        lps = [token_logprobs(model, e["ids"], e["mask"]) for e in eps]
    cat = torch.cat(lps)
    report = {"model": args.model, "n_episodes": len(eps),
              "n_model_tokens": int(sum(sum(e["mask"]) for e in eps)),
              "max_seq_len": max(len(e["ids"]) for e in eps), "t_load_s": round(t_load, 1),
              "logp_finite": bool(torch.isfinite(cat).all()),
              "logp_min": float(cat.min()), "logp_mean": float(cat.mean()),
              "peak_mem_gb_logp": torch.cuda.max_memory_allocated() / 2**30}
    t1 = time.time()
    scaler = torch.amp.GradScaler("cuda")
    stats = train_on(model, params, opt, scaler, eps, args.minibatches)
    report.update({"t_train_s": round(time.time() - t1, 1), **stats,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                   "params_finite": all(bool(torch.isfinite(p).all()) for p in params),
                   "lora_update_norm": float(torch.sqrt(sum((p.float() ** 2).sum() for p in params)))})
    report["fits"] = report["peak_mem_gb"] < 10.5
    report["pass"] = report["fits"] and report["logp_finite"] and report["params_finite"] \
        and all(map(lambda v: v == v, stats.values()))
    model.save_pretrained(os.path.join(out_dir, "adapter_step1"))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
