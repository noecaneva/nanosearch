"""Exp 04 after training: eval table with the summary judge, paired bootstrap against run2, training
curves, and run A episodes for the write-up. CPU only.

  PYTHONPATH=. python experiments/04-summary-judge/analyze.py
    -> experiments/04-summary-judge/out/{eval.json, curves.png, examples_runA.txt}
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanosearch import OUT_DIR
from nanosearch.env import PASSAGE_WORDS, judge_score, unsupported_titles
from nanosearch.indexes import load_corpus

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
EVAL = {"prompted": "agent_prompted", "run2": "agent_grpo_run2",
        "A (judge)": "agent_grpo_run5", "B (mix)": "agent_grpo_run6"}
RUNS = {"run2": "run2_wstep0", "A (judge)": "run5_judgeA", "B (mix)": "run6_judgeB"}
COLOR = {"recall": "#2a78d6", "judge": "#eb6834", "run2": "#8a8a85"}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def add_judge(rows, titles, texts):
    for r in rows:
        gold = [titles[d] for d in r["gold_doc_ids"]]
        r["judge"] = judge_score(r["turns"], gold)
        r["n_named"], r["n_unsupported"] = unsupported_titles(
            r["turns"], gold, [titles[d] for d in r["retrieved"]])
        r["gold_titles"] = gold
        got = {titles[d].lower() for d in r["retrieved"]}
        text = r["turns"][-1]["text"].lower() if r["turns"] else ""
        # unsupported titles that the question itself spells out (copying, not hallucinating)
        unsup = [t.lower() for t in gold if t.lower() in text and t.lower() not in got]
        shown = " ".join(" ".join(texts[d].split()[:PASSAGE_WORDS]) for d in r["retrieved"]).lower()
        q = r["question"].lower()
        # where an unsupported title comes from: the question, the shown passage text, or neither
        r["n_unsupported_in_q"] = sum(t in q for t in unsup)
        r["n_unsupported_in_text"] = sum(t not in q and t in shown for t in unsup)
        r["n_unsupported_neither"] = sum(t not in q and t not in shown for t in unsup)
    return rows


def bootstrap(a, b, n=10_000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return round(float(d.mean()), 3), round(float(lo), 3), round(float(hi), 3)


def smooth(x, w=10):
    x = np.asarray(x, float)
    return np.array([np.nanmean(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])


def main():
    os.makedirs(OUT, exist_ok=True)
    corpus = load_corpus()
    titles = {d["doc_id"]: d["title"] for d in corpus}
    texts = {d["doc_id"]: d["text"] for d in corpus}
    ev = {k: add_judge(read_jsonl(os.path.join(OUT_DIR, v, "eval_episodes.jsonl")), titles, texts)
          for k, v in EVAL.items()}
    table = {}
    for k, rows in ev.items():
        ends = {}
        for r in rows:
            e = r["end"].split(":")[0]
            ends[e] = ends.get(e, 0) + 1
        table[k] = {m: round(float(np.mean([r[m] for r in rows])), 3)
                    for m in ("recall@9", "all_gold@9", "judge", "n_searches", "format_valid", "n_named")}
        table[k]["unsupported"] = round(sum(r["n_unsupported"] for r in rows)
                                        / max(1, sum(r["n_named"] for r in rows)), 3)
        for m in ("in_q", "in_text", "neither"):
            table[k][f"unsupported_{m}"] = int(sum(r[f"n_unsupported_{m}"] for r in rows))
        table[k]["unsupported_n"] = int(sum(r["n_unsupported"] for r in rows))
        table[k]["judge>recall"] = int(sum(r["judge"] > r["recall@9"] for r in rows))
        table[k]["ends"] = ends
    paired = {f"{k} - run2": {m: bootstrap([r[m] for r in ev[k]], [r[m] for r in ev["run2"]])
                              for m in ("recall@9", "all_gold@9", "judge")}
              for k in ("A (judge)", "B (mix)")}
    paired["A - B"] = {m: bootstrap([r[m] for r in ev["A (judge)"]], [r[m] for r in ev["B (mix)"]])
                       for m in ("recall@9", "all_gold@9", "judge")}
    curves = {k: read_jsonl(os.path.join(OUT_DIR, "runs", v, "metrics.jsonl")) for k, v in RUNS.items()}
    blocks = {}
    for k in ("A (judge)", "B (mix)"):
        ms = curves[k]
        blocks[k] = [{"steps": f"{s}-{s + 24}", **{m: round(float(np.mean([x[m] for x in ms[s:s + 25]])), 3)
                      for m in ("judge", "recall@9", "n_searches", "frac_groups_filtered",
                                "unsupported", "titles_named", "format_valid")}}
                     for s in range(0, len(ms), 25)]
    with open(os.path.join(OUT, "eval.json"), "w") as f:
        json.dump({"eval": table, "paired": paired, "train_blocks": blocks}, f, indent=2)
    print(json.dumps({"eval": table, "paired": paired, "train_blocks": blocks}, indent=2))

    # curves: one panel per run, judge and true recall (10-step mean; raw faint), run2 recall as reference
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, sharey="row")
    for col, k in enumerate(("A (judge)", "B (mix)")):
        ms = curves[k]
        st = [m["step"] for m in ms]
        ax = axs[0, col]
        ax.plot([m["step"] for m in curves["run2"]], smooth([m["recall@9"] for m in curves["run2"]]),
                color=COLOR["run2"], lw=1.5, ls="--", label="run2 recall (reference)")
        for m, c, lab in (("recall@9", COLOR["recall"], "true recall@9"), ("judge", COLOR["judge"], "judge score")):
            y = [x[m] for x in ms]
            ax.plot(st, y, color=c, lw=0.8, alpha=0.25)
            ax.plot(st, smooth(y), color=c, lw=2, label=lab)
        ax.set_title(f"run {k}", loc="left", fontsize=11)
        ax.set_ylim(0, 1.02)
        ax = axs[1, col]
        ax.plot(st, smooth([x["unsupported"] for x in ms]), color=COLOR["judge"], lw=2,
                label="named gold titles not retrieved")
        ax.plot(st, smooth([x["frac_groups_filtered"] for x in ms]), color=COLOR["run2"], lw=2,
                label="zero-variance groups")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("training step")
    axs[0, 0].set_ylabel("mean per step (train)")
    axs[1, 0].set_ylabel("share")
    axs[0, 0].legend(frameon=False, fontsize=9, loc="lower right")
    axs[1, 0].legend(frameon=False, fontsize=9, loc="upper left")
    for ax in axs.flat:
        ax.grid(axis="y", color="#e5e5e2", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Exp 04: summary-only judge vs true recall during training (10-step mean)", x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "curves.png"), dpi=130)

    # run A eval episodes: largest judge - recall gap first, then a few typical ones
    rows = sorted(ev["A (judge)"], key=lambda r: (r["recall@9"] - r["judge"], r["qid"]))
    with open(os.path.join(OUT, "examples_runA.txt"), "w") as f:
        for r in rows[:6] + rows[len(rows) // 2:len(rows) // 2 + 2]:
            f.write(f"Question: {r['question']}\nGold: {'; '.join(r['gold_titles'])}\n")
            for i, t in enumerate(r["turns"], 1):
                f.write(f"turn {i}: {t['text']}\n")
            f.write(f"retrieved: {'; '.join(titles[d] for d in r['retrieved'])}\n"
                    f"end: {r['end']}, recall@9 {r['recall@9']}, judge {r['judge']}, "
                    f"named {r['n_named']}, unsupported {r['n_unsupported']}\n\n")


if __name__ == "__main__":
    main()
