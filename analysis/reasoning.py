"""Analysis 1: does the reasoning text before the action disappear during training?

A turn has reasoning if anything is left after removing its action tag and whitespace.
    python analysis/reasoning.py      -> analysis/out/reasoning.png, reasoning.json
"""

import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.common import OUT, RUNS, EVAL, eval_episodes, metrics, rollouts

TAG_RE = re.compile(r"<search>.*?</search>|<done/>", re.S)
KINDS = {"search": "search", "done": "stop"}  # malformed turns are left out of both


def reasoning(text):
    return re.sub(r"\s+", "", TAG_RE.sub("", text, count=1))


def turn_stats(turns):
    """-> {kind: (n_turns, n_with_reasoning, mean_chars)}"""
    by = defaultdict(list)
    for t in turns:
        if t["action"] in KINDS:
            by[KINDS[t["action"]]].append(len(reasoning(t["text"])))
    return {k: (len(v), int(np.sum(np.array(v) > 0)), float(np.mean(v))) for k, v in by.items()}


def rolling(x, w=10):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - w // 2): i + w // 2]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"train": {}, "eval": {}}
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    colors = {"run1": "C0", "run2": "C1"}
    for run in RUNS:
        eps = rollouts(run)
        steps = sorted({e["step"] for e in eps})
        per = {s: turn_stats([t for e in eps if e["step"] == s for t in e["turns"]]) for s in steps}
        res["train"][run] = {s: per[s] for s in steps}
        for ax, kind, ls in ((axes[0], "search", "-"), (axes[1], "stop", "-")):
            frac = [per[s][kind][1] / per[s][kind][0] if kind in per[s] else np.nan for s in steps]
            ax.scatter(steps, frac, s=6, color=colors[run], alpha=0.2)
            ax.plot(steps, rolling(frac), ls, color=colors[run], label=run)
        m = metrics(run)
        axes[2].plot([r["step"] for r in m], [r["recall@9"] for r in m], color=colors[run],
                     label=f"{run} recall@9")
        axes[2].plot([r["step"] for r in m], [r["n_searches"] / 3 for r in m], ":",
                     color=colors[run], label=f"{run} searches / 3")
    axes[0].set_ylabel("search turns\nwith reasoning")
    axes[1].set_ylabel("stop turns\nwith reasoning")
    axes[2].set_ylabel("all 128 episodes\nper step")
    axes[2].set_xlabel("training step")
    axes[0].set_title("Share of turns with text besides the action tag (logged group, 8 episodes/step; "
                      "line = 10-step rolling mean)", fontsize=9)
    for ax in axes:
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=8, loc="center right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "reasoning.png"), dpi=130)

    for system in EVAL:
        turns = [t for e in eval_episodes(system) for t in e["turns"]]
        res["eval"][system] = turn_stats(turns)
        res["eval"][system]["all"] = (len(turns), sum(len(reasoning(t["text"])) > 0 for t in turns))
    with open(os.path.join(OUT, "reasoning.json"), "w") as f:
        json.dump(res, f, indent=1)

    print("eval: kind n_turns n_with_reasoning share mean_chars")
    for system, st in res["eval"].items():
        for kind in ("search", "stop"):
            if kind in st:
                n, k, c = st[kind]
                print(f"  {system:9s} {kind:6s} {n:4d} {k:4d} {k / n:.3f} {c:6.1f}")
        print(f"  {system:9s} all    {st['all'][0]:4d} {st['all'][1]:4d}  (incl. malformed turns)")

    print("train, 25-step buckets: share with reasoning (n turns), mean chars")
    for run in RUNS:
        for kind in ("search", "stop"):
            row = []
            for b in range(0, 150, 25):
                sts = [res["train"][run][s][kind] for s in range(b, b + 25) if kind in res["train"][run][s]]
                n, k = sum(x[0] for x in sts), sum(x[1] for x in sts)
                c = sum(x[0] * x[2] for x in sts) / n if n else float("nan")
                row.append(f"{k / n:.2f} ({n}) {c:5.1f}" if n else "  -  ")
            print(f"  {run} {kind:6s} " + " | ".join(row))

    # The first search turn is bare even for the prompted model, so split search turns by position.
    def by_pos(eps):
        c = defaultdict(lambda: [0, 0])
        for e in eps:
            for i, t in enumerate([t for t in e["turns"] if t["action"] == "search"], 1):
                c[i][0] += 1
                c[i][1] += len(reasoning(t["text"])) > 0
        return " ".join(f"#{i}: {k}/{n}" for i, (n, k) in sorted(c.items()))

    print("search turns with reasoning, by search position:")
    for system in EVAL:
        print(f"  eval {system:9s} {by_pos(eval_episodes(system))}")
    for run in RUNS:
        eps = rollouts(run)
        for lo, hi in ((0, 10), (10, 40), (40, 150)):
            print(f"  train {run} steps {lo}-{hi}: {by_pos([e for e in eps if lo <= e['step'] < hi])}")

    rng = np.random.default_rng(0)
    for run in RUNS:
        eps = rollouts(run)
        for name, cond in (("early (step < 10)", lambda s: s < 10), ("late (step > 140)", lambda s: s > 140)):
            pool = [(e["step"], t) for e in eps if cond(e["step"]) for t in e["turns"] if t["action"] in KINDS]
            print(f"\n{run} {name}, 3 random turns:")
            for i in rng.choice(len(pool), 3, replace=False):
                s, t = pool[i]
                print(f"  step {s:3d} {t['action']:6s} {t['text']!r}")


if __name__ == "__main__":
    main()
