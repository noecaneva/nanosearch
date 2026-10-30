"""Exp 05: per (model, condition) metrics, R - L paired bootstrap per model, the trained models' R - L
against the prompted model's (difference in differences), and verbatim L/R pairs. CPU only.

  PYTHONPATH=. python experiments/05-stated-grader/analyze.py
    -> experiments/05-stated-grader/out/{eval.json, examples.txt}
"""

import importlib.util
import json
import os

import numpy as np

from nanosearch import OUT_DIR
from nanosearch.indexes import load_corpus

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
spec = importlib.util.spec_from_file_location(
    "exp04", os.path.join(HERE, "..", "04-summary-judge", "analyze.py"))
exp04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp04)

EVAL = {("A40", "0"): "exp05_A40_0", ("A40", "L"): "exp05_A40_L", ("A40", "R"): "exp05_A40_R",
        ("A150", "0"): "agent_grpo_run5", ("A150", "L"): "exp05_A150_L", ("A150", "R"): "exp05_A150_R",
        ("prompted", "0"): "agent_prompted", ("prompted", "L"): "exp05_prompted_L",
        ("prompted", "R"): "exp05_prompted_R",
        ("A150", "0 rerun"): "exp05_A150_0rerun", ("prompted", "0 rerun"): "exp05_prompted_0rerun"}
METRICS = ("n_searches", "recall@9", "all_gold@9", "judge", "n_named", "n_unsupported", "n_generated")


def main():
    os.makedirs(OUT, exist_ok=True)
    corpus = load_corpus()
    titles = {d["doc_id"]: d["title"] for d in corpus}
    texts = {d["doc_id"]: d["text"] for d in corpus}
    ev = {}
    for k, v in EVAL.items():
        rows = exp04.add_judge(exp04.read_jsonl(os.path.join(OUT_DIR, v, "eval_episodes.jsonl")),
                               titles, texts)
        ev[k] = sorted(rows, key=lambda r: r["qid"])
    qids = [r["qid"] for r in ev[("A150", "0")]]
    assert all([r["qid"] for r in rows] == qids for rows in ev.values())

    table = {}
    for (m, c), rows in ev.items():
        t = {x: round(float(np.mean([r[x] for r in rows])), 3) for x in METRICS}
        t["unsupported_n"] = int(sum(r["n_unsupported"] for r in rows))
        t["named_n"] = int(sum(r["n_named"] for r in rows))
        for s in ("in_q", "in_text", "neither"):
            t[f"unsupported_{s}"] = int(sum(r[f"n_unsupported_{s}"] for r in rows))
        t["one_search"] = int(sum(r["n_searches"] == 1 for r in rows))
        t["format_valid"] = round(float(np.mean([r["format_valid"] for r in rows])), 3)
        table[f"{m} {c}"] = t

    def col(m, c, x):
        return np.array([r[x] for r in ev[(m, c)]], float)

    paired, did, noise = {}, {}, {}
    for m in ("A40", "A150", "prompted"):
        paired[f"{m} R-L"] = {x: exp04.bootstrap(col(m, "R", x), col(m, "L", x)) for x in METRICS}
        paired[f"{m} L-0"] = {x: exp04.bootstrap(col(m, "L", x), col(m, "0", x)) for x in METRICS}
        paired[f"{m} R-0"] = {x: exp04.bootstrap(col(m, "R", x), col(m, "0", x)) for x in METRICS}
    for m in ("A40", "A150"):
        did[f"{m} (R-L) - prompted (R-L)"] = {
            x: exp04.bootstrap(col(m, "R", x) - col(m, "L", x),
                               col("prompted", "R", x) - col("prompted", "L", x)) for x in METRICS}
    for m in ("A150", "prompted"):
        noise[f"{m} 0rerun-0"] = {x: exp04.bootstrap(col(m, "0 rerun", x), col(m, "0", x))
                                  for x in METRICS}
        noise[f"{m} 0rerun-0"]["episodes_identical"] = int(sum(
            a["turns"] == b["turns"] for a, b in zip(ev[(m, "0 rerun")], ev[(m, "0")])))
    res = {"eval": table, "paired": paired, "diff_in_diff": did, "noise_floor": noise}
    with open(os.path.join(OUT, "eval.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

    # L/R pairs for the trained models: largest drop in unsupported titles first, then largest rise
    with open(os.path.join(OUT, "examples.txt"), "w") as f:
        for m in ("A150", "A40"):
            pairs = list(zip(ev[(m, "L")], ev[(m, "R")]))
            pairs.sort(key=lambda p: (p[1]["n_unsupported"] - p[0]["n_unsupported"],
                                      p[0]["n_searches"] - p[1]["n_searches"], p[0]["qid"]))
            for L, R in pairs[:4] + pairs[-2:]:
                f.write(f"=== {m} | Question: {L['question']}\nGold: {'; '.join(L['gold_titles'])}\n")
                for c, r in (("L", L), ("R", R)):
                    for i, t in enumerate(r["turns"], 1):
                        f.write(f"[{c}] turn {i}: {t['text']}\n")
                    f.write(f"[{c}] retrieved: {'; '.join(titles[d] for d in r['retrieved'])}\n"
                            f"[{c}] searches {r['n_searches']}, recall@9 {r['recall@9']}, judge "
                            f"{r['judge']}, named {r['n_named']}, unsupported {r['n_unsupported']}\n")
                f.write("\n")


if __name__ == "__main__":
    main()
