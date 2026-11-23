"""Exp 08 tables from $GRPOP_ROOT/outputs/exp08: the ladder with paired steps, and the full grid.

  python experiments/08-lexical-ladder/analyze.py --split eval   # -> experiments/08-lexical-ladder/out/
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from ladder import IDF, LADDER, NORM, TF, cfg_name  # noqa: E402
from nanosearch import OUT_DIR  # noqa: E402

METRICS = ("recall@9", "all_gold@9", "ndcg@9", "recall@3")


def bootstrap(a, b, n=10_000, seed=0):
    """Paired mean difference a - b over questions with a 95% CI (same as Exp 04's helper)."""
    d = np.asarray(a) - np.asarray(b)
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return round(float(d.mean()), 3), round(float(lo), 3), round(float(hi), 3)


def load(out_dir, name, split):
    with open(os.path.join(out_dir, f"{name}_{split}.json")) as f:
        s = json.load(f)
    with open(os.path.join(out_dir, f"{name}_{split}_episodes.jsonl")) as f:
        eps = sorted((json.loads(x) for x in f), key=lambda e: e["qid"])
    return s, eps


def col(eps, m):
    return [e[m] for e in eps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval")
    ap.add_argument("--out_dir", default=os.path.join(OUT_DIR, "exp08"))
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)

    bm25_s, bm25_e = load(a.out_dir, cfg_name("sat", "lucene", "b=0.75"), a.split)
    lines = [f"## Ladder ({a.split}, n={bm25_s['n_questions']})", "",
             "| rung | scorer | recall@9 | all_gold@9 | nDCG@9 | bridge r@9 | comparison r@9 "
             "| tie@9 | Δ recall@9 vs previous rung [95% CI] | Δ vs BM25 [95% CI] |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    out = {"ladder": [], "grid": []}
    prev = None
    for rung, label, cfg in LADDER:
        s, e = load(a.out_dir, cfg_name(**cfg), a.split)
        # L1s is a side branch: compared with L1, and L2 then steps from L1, not from L1s.
        step = bootstrap(col(e, "recall@9"), col(prev, "recall@9")) if prev else None
        vs = bootstrap(col(e, "recall@9"), col(bm25_e, "recall@9"))
        bt = s["by_type"]
        lines.append(f"| {rung} | {label} | {s['recall@9']:.3f} | {s['all_gold@9']:.3f} | "
                     f"{s['ndcg@9']:.3f} | {bt.get('bridge', {}).get('recall@9', float('nan')):.3f} | "
                     f"{bt.get('comparison', {}).get('recall@9', float('nan')):.3f} | "
                     f"{s['boundary_tie_rate']:.2f} | "
                     + (f"{step[0]:+.3f} [{step[1]:+.3f}, {step[2]:+.3f}]" if step else "–")
                     + f" | {vs[0]:+.3f} [{vs[1]:+.3f}, {vs[2]:+.3f}] |")
        out["ladder"].append({"rung": rung, "label": label, **cfg, **{m: s[m] for m in METRICS},
                              "step_vs_prev": step, "vs_bm25": vs,
                              "boundary_tie_rate": s["boundary_tie_rate"]})
        if rung != "L1s":
            prev = e

    lines += ["", f"## Grid, recall@9 ({a.split})", "",
              "| tf | idf | " + " | ".join(NORM) + " |", "|---|---|" + "---|" * len(NORM)]
    for t in TF:
        for i in IDF:
            cells = []
            for n in NORM:
                s, _ = load(a.out_dir, cfg_name(t, i, n), a.split)
                cells.append(f"{s['recall@9']:.3f}")
                out["grid"].append({"tf": t, "idf": i, "norm": n,
                                    **{m: s[m] for m in METRICS},
                                    "boundary_tie_rate": s["boundary_tie_rate"]})
            lines.append(f"| {t} | {i} | " + " | ".join(cells) + " |")

    md = "\n".join(lines) + "\n"
    print(md)
    with open(os.path.join(HERE, "out", f"tables_{a.split}.md"), "w") as f:
        f.write(md)
    with open(os.path.join(HERE, "out", f"summary_{a.split}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
