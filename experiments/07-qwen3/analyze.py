"""Exp 07: the eval table (prompted Qwen3 in core and in the Exp 02 layout, each seed's step-150
adapter in core and in Exp 06's harness T), Rule 1 and Rule 2 per seed with paired bootstrap CIs,
the seed-gap check, the day-0 numbers, training curves of each seed next to run2's, and three
verbatim episodes per seed. CPU only.

  PYTHONPATH=. python experiments/07-qwen3/analyze.py [s0 s1 ...]
    -> experiments/07-qwen3/out/{eval.json, curves.png, examples.txt}
"""

import importlib.util
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nanosearch import OUT_DIR  # noqa: E402
from nanosearch.indexes import load_corpus  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
spec = importlib.util.spec_from_file_location(
    "exp04", os.path.join(HERE, "..", "04-summary-judge", "analyze.py"))
exp04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp04)

METRICS = ("recall@9", "all_gold@9", "ndcg@9", "recall@3", "n_searches", "format_valid", "n_generated")
MARGIN = 0.03  # README: Rule 1 gain, Rule 2 allowed drop, seed gap


def read_eval(name):
    rows = exp04.read_jsonl(os.path.join(OUT_DIR, name, "eval_episodes.jsonl"))
    for r in rows:
        r.setdefault("truncated_turns", 0)
        r.setdefault("think_turns", 0)
    return sorted(rows, key=lambda r: r["qid"])


def table_row(rows):
    t = {x: round(float(np.mean([r[x] for r in rows])), 3) for x in METRICS}
    ends = {}
    for r in rows:
        e = r["end"].split(":")[0]
        ends[e] = ends.get(e, 0) + 1
    t["ends"] = dict(sorted(ends.items()))
    t["malformed_reasons"] = dict(sorted(
        (k, sum(r["end"] == k for r in rows)) for k in {r["end"] for r in rows} if ":" in k))
    t["truncated_turns"] = int(sum(r["truncated_turns"] for r in rows))
    t["think_turns"] = int(sum(r["think_turns"] for r in rows))
    t["n_turns"] = int(sum(len(r["turns"]) for r in rows))
    t["mean_query_words"] = round(float(np.mean(
        [len(q.split()) for r in rows for q in r["queries"]] or [0])), 2)
    return t


def col(rows, x):
    return np.array([r[x] for r in rows], float)


def main():
    seeds = sys.argv[1:] or [s for s in ("s0", "s1", "s2")
                             if os.path.exists(os.path.join(OUT_DIR, f"exp07_{s}_core", "eval_episodes.jsonl"))]
    os.makedirs(OUT, exist_ok=True)
    titles = {d["doc_id"]: d["title"] for d in load_corpus()}
    ev = {"prompted core": read_eval("exp07_prompted_core"),
          "prompted sep": read_eval("exp07_prompted_sep")}
    for s in seeds:
        ev[f"{s} core"] = read_eval(f"exp07_{s}_core")
        ev[f"{s} T"] = read_eval(f"exp07_{s}_T")
    qids = [r["qid"] for r in ev["prompted core"]]
    assert all([r["qid"] for r in rows] == qids for rows in ev.values())
    table = {k: table_row(v) for k, v in ev.items()}
    for k in ("agent_prompted", "agent_grpo_run2"):  # Qwen2.5 references, same questions
        table[f"qwen2.5 {k}"] = table_row(read_eval(k))

    # format-fixed baseline: the better of the two prompted layouts by recall@9
    base_key = max(("prompted core", "prompted sep"), key=lambda k: table[k]["recall@9"])
    base = ev[base_key]
    res = {"seeds": seeds, "baseline": {"layout": base_key, **table[base_key]},
           "side_prediction_prompted_qwen3_ge_0.85": table[base_key]["recall@9"] >= 0.85,
           "prompted sep-core": {x: exp04.bootstrap(col(ev["prompted sep"], x), col(ev["prompted core"], x))
                                 for x in METRICS}}
    rules = {}
    for s in seeds:
        core, T = ev[f"{s} core"], ev[f"{s} T"]
        gain = {x: exp04.bootstrap(col(core, x), col(base, x)) for x in METRICS}
        drop = {x: exp04.bootstrap(col(T, x), col(core, x)) for x in METRICS}
        d = gain["recall@9"]
        # "query behaviour moved": retrieval quality itself changed, not only the tags
        moved = {x: bool(gain[x][1] > 0 or gain[x][2] < 0) for x in ("n_searches", "all_gold@9", "recall@3")}
        r1 = bool(d[0] > MARGIN and d[1] > 0 and (moved["all_gold@9"] or moved["recall@3"]))
        t_rec, t_fmt = table[f"{s} T"]["recall@9"], table[f"{s} T"]["format_valid"]
        r2 = bool(drop["recall@9"][2] >= -MARGIN and t_rec > table[base_key]["recall@9"])
        rules[s] = {"rule1_gain_over_baseline": gain, "rule1_moved": moved, "rule1_pass": r1,
                    "rule2_T_minus_core": drop, "rule2_T_recall@9": t_rec,
                    "rule2_T_format_valid": t_fmt, "rule2_could_not_act": t_fmt < 0.90,
                    "rule2_pass": r2, "passes_both": r1 and r2}
    res["rules"] = rules
    if len(seeds) >= 2:
        recs = {s: table[f"{s} core"]["recall@9"] for s in seeds}
        res["seed_gap"] = {"recall@9_step150_core": recs,
                           "max_abs_gap": round(max(recs.values()) - min(recs.values()), 3),
                           "third_seed_required": bool(max(recs.values()) - min(recs.values()) > MARGIN)}
        res["seed_pairs"] = {f"{a}-{b}": exp04.bootstrap(col(ev[f"{a} core"], "recall@9"),
                                                       col(ev[f"{b} core"], "recall@9"))
                             for i, a in enumerate(seeds) for b in seeds[i + 1:]}
        allpass = all(rules[s]["passes_both"] for s in seeds)
        allfail = all(not rules[s]["passes_both"] for s in seeds)
        res["verdict"] = ("recipe teaches search on Qwen3 (every seed passes both rules)" if allpass
                          else "H2 holds (every seed fails at least one rule)" if allfail
                          else "unstable at this scale (seeds split)")
    for name in ("exp07_day0_groups", "exp07_day0_step", "exp07_day0_lora"):
        p = os.path.join(OUT_DIR, name, "summary.json" if "day0_" in name and "lora" not in name else "eval.json")
        if os.path.exists(p):
            res[name] = json.load(open(p))
    res["eval"] = table
    with open(os.path.join(OUT, "eval.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in ("eval",)}, indent=2))
    print(json.dumps(table, indent=2))

    # training curves, each seed next to run2 (Qwen2.5)
    runs = {"run2 (Qwen2.5)": "run2_wstep0", **{f"q3_run2_{s}": f"q3_run2_{s}" for s in seeds}}
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True)
    keys = (("recall@9", "train recall@9"), ("n_searches", "searches / episode"),
            ("format_valid", "format validity"), ("frac_groups_filtered", "zero-variance groups"),
            ("truncated_turns", "truncated turns / episode"), ("think_turns", "<think> turns / episode"))
    for name, d in runs.items():
        p = os.path.join(OUT_DIR, "runs", d, "metrics.jsonl")
        if not os.path.exists(p):
            continue
        m = exp04.read_jsonl(p)
        for ax, (k, lab) in zip(axs.flat, keys):
            if k in m[0]:
                ax.plot([r["step"] for r in m], exp04.smooth([r[k] for r in m]),
                        label=name, color="grey" if "run2 (" in name else None, lw=1.2)
            ax.set_title(lab, fontsize=10)
    for ax in axs[1]:
        ax.set_xlabel("step")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Exp 07: run2 recipe on Qwen3-1.7B (10-step running mean)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "curves.png"), dpi=130)

    with open(os.path.join(OUT, "examples.txt"), "w") as f:
        def dump(tag, r):
            for i, t in enumerate(r["turns"], 1):
                f.write(f"[{tag}] turn {i} ({t['action']}, {t.get('n_tokens', '?')} tok): {t['text']}\n")
            f.write(f"[{tag}] retrieved: {'; '.join(titles[d] for d in r['retrieved'])}\n"
                    f"[{tag}] end {r['end']}, searches {r['n_searches']}, recall@9 {r['recall@9']}\n")

        f.write(f"prompted Qwen3 in {base_key} (baseline), first 2 episodes\n\n")
        for r in base[:2]:
            f.write(f"Question: {r['question']}\nGold: {'; '.join(titles[d] for d in r['gold_doc_ids'])}\n")
            dump("prompted", r)
            f.write("\n")
        for s in seeds:
            core, T = ev[f"{s} core"], {r["qid"]: r for r in ev[f"{s} T"]}
            ok = [r for r in core if r["recall@9"] == 1.0 and base[qids.index(r["qid"])]["recall@9"] < 1.0]
            miss = [r for r in core if r["recall@9"] < 1.0]
            tmiss = [r for r in core if r["recall@9"] == 1.0 and T[r["qid"]]["recall@9"] < 1.0]
            for name, lst in ((f"{s}: trained hit, prompted miss", ok), (f"{s}: trained miss", miss),
                              (f"{s}: hit in core, miss in T (both shown)", tmiss)):
                f.write(f"=== {name} ({len(lst)} such)\n")
                if not lst:
                    f.write("(none)\n\n")
                    continue
                r = sorted(lst, key=lambda r: r["qid"])[0]
                f.write(f"Question: {r['question']}\nGold: {'; '.join(titles[d] for d in r['gold_doc_ids'])}\n")
                dump("core", r)
                if "T" in name:
                    dump("T", T[r["qid"]])
                f.write("\n")


if __name__ == "__main__":
    main()
