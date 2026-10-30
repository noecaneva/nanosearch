"""Loaders shared by the two analyses. Rollout turns are logged as plain strings, so they are
re-parsed with the environment's own parse_turn."""

import json
import os

from nanosearch import OUT_DIR
from nanosearch.env import parse_turn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
EVAL = {"prompted": "agent_prompted", "run1": "agent_grpo_run1", "run2": "agent_grpo_run2"}
RUNS = {"run1": "run1", "run2": "run2_wstep0"}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def eval_episodes(system):
    return read_jsonl(os.path.join(OUT_DIR, EVAL[system], "eval_episodes.jsonl"))


def rollouts(run):
    """Episodes with turns as [{text, action, arg}] like the eval files, plus step."""
    eps = read_jsonl(os.path.join(OUT_DIR, "runs", RUNS[run], "rollouts.jsonl"))
    for e in eps:
        e["turns"] = [dict(zip(("action", "arg"), parse_turn(t)), text=t) for t in e["turns"]]
    return eps


def metrics(run):
    return read_jsonl(os.path.join(OUT_DIR, "runs", RUNS[run], "metrics.jsonl"))
