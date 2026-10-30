"""Analysis 2: are searches used out of habit? Per search position (1st/2nd/3rd):
exact repeat of an earlier query (lowercase + strip), new passages added, new gold passage added.

Per-search hits are not logged, so each stored query is re-run on the env's BM25 retriever (k=3).
Checks: on eval, the union of re-run hits must equal the stored `retrieved`; on run2's rollouts
(w_step 0, so reward = recall + 0.2 * format_valid), the re-run recall must match the reward.
    python analysis/habit.py      -> analysis/out/habit.json
"""

import json
import os

import numpy as np

from analysis.common import OUT, EVAL, eval_episodes, read_jsonl, rollouts
from nanosearch import DATA_DIR
from nanosearch.env import MAX_SEARCHES, PASSAGES_PER_SEARCH, W_FORMAT
from nanosearch.indexes import BM25Retriever


def per_search(queries, gold, retriever):
    hits = retriever.search_batch(queries, PASSAGES_PER_SEARCH) if queries else []
    seen, rows, union = set(), [], []
    for i, (q, h) in enumerate(zip(queries, hits)):
        new = [d for d in h if d not in seen]
        rows.append({"pos": i + 1, "gold_missing": bool(gold - seen),
                     "repeat": q.lower().strip() in {p.lower().strip() for p in queries[:i]},
                     "new": len(new), "new_gold": any(d in gold for d in new)})
        seen.update(h)
        union += [d for d in h if d not in union]
    return rows, union


def table(rows):
    out = {}
    for pos in range(1, MAX_SEARCHES + 1):
        r = [x for x in rows if x["pos"] == pos]
        if r:
            out[pos] = {"n": len(r), "repeat": sum(x["repeat"] for x in r),
                        "mean_new": float(np.mean([x["new"] for x in r])),
                        "new_gold": sum(x["new_gold"] for x in r),
                        "gold_missing": sum(x["gold_missing"] for x in r)}
    return out


def fmt(name, t):
    return [f"| {name} | {p} | {v['n']} | {v['repeat']} ({v['repeat'] / v['n']:.2f}) | "
            f"{v['mean_new']:.2f} | {v['gold_missing']} | {v['new_gold']} ({v['new_gold'] / v['n']:.2f}) |" for p, v in t.items()]


def main():
    os.makedirs(OUT, exist_ok=True)
    retriever = BM25Retriever()
    res, lines = {"eval": {}, "train_run2": {}}, []
    hdr = "| {} | search | n | repeated | mean new passages | gold still missing before | adds new gold |\n|---|---|---|---|---|---|---|"
    lines.append(hdr.format("system (eval)"))
    for system in EVAL:
        eps = eval_episodes(system)
        rows, mismatch = [], 0
        for e in eps:
            r, union = per_search(e["queries"], set(e["gold_doc_ids"]), retriever)
            mismatch += union != e["retrieved"]
            rows += r
        n_rep_eps = sum(any(p.lower().strip() in {x.lower().strip() for x in e["queries"][:i]}
                            for i, p in enumerate(e["queries"])) for e in eps)
        res["eval"][system] = {"table": table(rows), "retrieved_mismatch": mismatch,
                               "episodes_with_repeat": n_rep_eps, "episodes": len(eps)}
        print(f"{system}: re-run union != stored retrieved in {mismatch}/{len(eps)} episodes; "
              f"episodes with a repeated query: {n_rep_eps}/{len(eps)}")
        lines += fmt(system, res["eval"][system]["table"])

    gold = {q["question"]: set(q["gold_doc_ids"]) for q in read_jsonl(os.path.join(DATA_DIR, "train.jsonl"))}
    eps = rollouts("run2")
    lines.append("\n" + hdr.format("run2 rollouts, steps"))
    rows_by_bucket, err = {}, []
    for e in eps:
        g = gold[e["question"]]
        r, union = per_search(e["queries"], g, retriever)
        fv = int(not e["end"].startswith("malformed")) * (len(e["queries"]) > 0)
        err.append(abs(len(set(union) & g) / len(g) + W_FORMAT * fv - e["reward"]))
        b = e["step"] // 25 * 25
        rows_by_bucket.setdefault(b, []).extend(r)
    err = np.array(err)
    print(f"run2 rollouts: |re-run reward - logged reward| > 1e-6 in {(err > 1e-6).sum()}/{len(err)} episodes")
    res["train_run2_reward_mismatch"] = int((err > 1e-6).sum())
    for b, rows in sorted(rows_by_bucket.items()):
        res["train_run2"][f"{b}-{b + 25}"] = table(rows)
        lines += fmt(f"{b}-{b + 25}", table(rows))

    with open(os.path.join(OUT, "habit.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(OUT, "habit_tables.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
