"""Exp 06: eight-row table, paired harness - core deltas per adapter, the core - T trajectory gap per
checkpoint, the noise floor for greedy decoding, the verdict by the pre-registered rules, and three
verbatim T episodes of step 150. CPU only.

  PYTHONPATH=. python experiments/06-harness-transfer/analyze.py
    -> experiments/06-harness-transfer/out/{eval.json, examples.txt}
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

# (adapter, harness) -> out dir. The two core references that already existed are reused.
EVAL = {("step0150", "core"): "agent_grpo_run2", ("step0150", "core rerun"): "exp06_step0150_core",
        ("step0150", "W"): "exp06_step0150_W", ("step0150", "T"): "exp06_step0150_T",
        ("step0010", "core"): "exp06_step0010_core", ("step0010", "T"): "exp06_step0010_T",
        ("step0050", "core"): "exp06_step0050_core", ("step0050", "T"): "exp06_step0050_T",
        ("prompted", "core"): "agent_prompted", ("prompted", "T"): "exp06_prompted_T"}
METRICS = ("recall@9", "all_gold@9", "ndcg@9", "recall@3", "n_searches", "format_valid", "n_generated")
DENSE, MARGIN = 0.853, 0.03  # README rules: dense single-shot recall@9, allowed paired drop


def main():
    os.makedirs(OUT, exist_ok=True)
    titles = {d["doc_id"]: d["title"] for d in load_corpus()}
    ev = {}
    for k, v in EVAL.items():
        rows = exp04.read_jsonl(os.path.join(OUT_DIR, v, "eval_episodes.jsonl"))
        for r in rows:
            if "old_token_turns" not in r:  # older evals (core harness): no renamed tokens
                r["old_token_turns"] = 0
        ev[k] = sorted(rows, key=lambda r: r["qid"])
    qids = [r["qid"] for r in ev[("step0150", "core")]]
    assert all([r["qid"] for r in rows] == qids for rows in ev.values())

    table = {}
    for (m, h), rows in ev.items():
        t = {x: round(float(np.mean([r[x] for r in rows])), 3) for x in METRICS}
        ends = {}
        for r in rows:
            e = r["end"].split(":")[0]
            ends[e] = ends.get(e, 0) + 1
        t["ends"] = dict(sorted(ends.items()))
        t["malformed_reasons"] = dict(sorted(
            (k, sum(r["end"] == k for r in rows)) for k in {r["end"] for r in rows} if ":" in k))
        t["old_token_turns"] = int(sum(r["old_token_turns"] for r in rows))
        t["episodes_with_old_token"] = int(sum(r["old_token_turns"] > 0 for r in rows))
        t["mean_query_words"] = round(float(np.mean(
            [len(q.split()) for r in rows for q in r["queries"]] or [0])), 2)
        table[f"{m} {h}"] = t

    def col(m, h, x):
        return np.array([r[x] for r in ev[(m, h)]], float)

    paired = {}
    for m, h in (("step0150", "W"), ("step0150", "T"), ("step0010", "T"), ("step0050", "T"),
                 ("prompted", "T")):
        paired[f"{m} {h}-core"] = {x: exp04.bootstrap(col(m, h, x), col(m, "core", x)) for x in METRICS}
    noise = {x: exp04.bootstrap(col("step0150", "core rerun", x), col("step0150", "core", x))
             for x in METRICS}
    noise["episodes_identical"] = int(sum(a["turns"] == b["turns"] for a, b in zip(
        ev[("step0150", "core rerun")], ev[("step0150", "core")])))
    # Part B: gap core - T per checkpoint, and the change of the gap between checkpoints
    gap = {m: exp04.bootstrap(col(m, "core", "recall@9"), col(m, "T", "recall@9"))
           for m in ("step0010", "step0050", "step0150")}
    gap["step0150 - step0050"] = exp04.bootstrap(
        col("step0150", "core", "recall@9") - col("step0150", "T", "recall@9"),
        col("step0050", "core", "recall@9") - col("step0050", "T", "recall@9"))

    # Verdict by the README rules (Part A, step 150 in W and T)
    verdict = {}
    for h in ("W", "T"):
        rec = table[f"step0150 {h}"]["recall@9"]
        fmt = table[f"step0150 {h}"]["format_valid"]
        mean, lo, hi = paired[f"step0150 {h}-core"]["recall@9"]
        verdict[h] = {"recall@9": rec, "format_valid": fmt, "delta": [mean, lo, hi],
                      "confounded_format_below_0.90": fmt < 0.90,
                      "falsified": bool(hi < -MARGIN or rec < DENSE),
                      "survives": bool(hi >= -MARGIN and rec >= DENSE)}
    verdict["H1"] = ("falsified" if any(verdict[h]["falsified"] for h in ("W", "T"))
                     else "survives")
    verdict["partB_transfer_at_every_stage_rejected"] = bool(gap["step0150 - step0050"][0] > MARGIN)
    res = {"eval": table, "paired": paired, "noise_floor_step0150_rerun-core": noise,
           "trajectory_gap_core_minus_T_recall@9": gap, "verdict": verdict}
    with open(os.path.join(OUT, "eval.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

    # three verbatim T episodes of step 150: one success, one miss, one with an old-token turn
    rows = ev[("step0150", "T")]
    core = {r["qid"]: r for r in ev[("step0150", "core")]}
    picks = []
    ok = [r for r in rows if r["recall@9"] == 1.0 and r["old_token_turns"] == 0]
    miss = [r for r in rows if r["recall@9"] < 1.0 and core[r["qid"]]["recall@9"] == 1.0]
    old = [r for r in rows if r["old_token_turns"] > 0]
    for name, lst in (("success (recall@9 1.0)", ok), ("miss in T, hit in core", miss),
                      ("old-token turn", old)):
        if lst:
            picks.append((name, sorted(lst, key=lambda r: r["qid"])[0]))
        else:
            picks.append((name, None))
    with open(os.path.join(OUT, "examples.txt"), "w") as f:
        f.write(f"step0150 in harness T: {len(ok)} clean successes, {len(miss)} misses that core got, "
                f"{len(old)} episodes with an old-token turn\n\n")
        for name, r in picks:
            f.write(f"=== {name}\n")
            if r is None:
                f.write("(none)\n\n")
                continue
            gold = [titles[d] for d in r["gold_doc_ids"]]
            f.write(f"Question: {r['question']}\nGold: {'; '.join(gold)}\n")
            for i, t in enumerate(r["turns"], 1):
                f.write(f"[T] turn {i} ({t['action']}): {t['text']}\n")
            f.write(f"[T] retrieved: {'; '.join(titles[d] for d in r['retrieved'])}\n"
                    f"[T] end {r['end']}, searches {r['n_searches']}, recall@9 {r['recall@9']}\n")
            c = core[r["qid"]]
            for i, t in enumerate(c["turns"], 1):
                f.write(f"[core] turn {i} ({t['action']}): {t['text']}\n")
            f.write(f"[core] end {c['end']}, searches {c['n_searches']}, recall@9 {c['recall@9']}\n\n")


if __name__ == "__main__":
    main()
