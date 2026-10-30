"""Build the question splits and passage corpus from HotpotQA (distractor setting).

Train questions come from the HotpotQA train split, eval questions from the validation split,
so the two can never overlap. The corpus is every paragraph (gold and distractor) attached to
the sampled questions, deduplicated on normalized title + first sentence.

core-v1 = this script with defaults (seed 0, 500 train, 200 eval, 2,500 extra distractor
questions): 30,358 passages. md5s in $GRPOP_ROOT/core-v1/MD5SUMS.

  python -m nanosearch.corpus

Outputs (in --out_dir):
  corpus.jsonl      doc_id, title, text, is_gold
  train.jsonl       qid, question, answer, type, level, gold_doc_ids, para_doc_ids
  eval.jsonl        same fields
  train3000.jsonl   train + the extra distractor questions as questions (gold already in the
                    corpus, so core-v1 is unchanged); used by --train_file in nanosearch.grpo
  corpus_stats.json counts, dedup rule, dataset revision
"""

import argparse
import json
import os
import random
import re
import unicodedata

from datasets import load_dataset
from huggingface_hub import HfApi

from nanosearch import DATA_DIR

DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def dedup_key(title: str, sentences: list[str]) -> str:
    first = sentences[0] if sentences else ""
    return normalize(title) + "||" + normalize(first)


def sample(split, n, seed, offset=0):
    """Rows idx[offset:offset+n] of one fixed permutation, so different offsets never overlap."""
    idx = list(range(len(split)))
    random.Random(seed).shuffle(idx)
    return [split[i] for i in sorted(idx[offset:offset + n])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=DATA_DIR)
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--n_extra", type=int, default=2500,
                    help="extra train-split questions whose paragraphs are added as distractors "
                         "only; disjoint from the train sample, never used as questions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset(DATASET, CONFIG)
    revision = HfApi().dataset_info(DATASET).sha
    splits = {
        "train": sample(ds["train"], args.n_train, args.seed),
        "eval": sample(ds["validation"], args.n_eval, args.seed + 1),
    }

    key_to_id: dict[str, str] = {}
    docs: list[dict] = []
    n_paragraphs_raw = 0

    def add_paragraph(title, sents):
        key = dedup_key(title, sents)
        if key not in key_to_id:
            key_to_id[key] = f"d{len(docs):05d}"
            docs.append({"doc_id": key_to_id[key], "title": title,
                         "text": "".join(sents).strip(), "is_gold": False})
        return key_to_id[key]

    out_questions = {}
    for name, rows in splits.items():
        out_questions[name] = []
        for ex in rows:
            gold_titles = set(ex["supporting_facts"]["title"])
            para_ids, gold_ids = [], []
            for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
                n_paragraphs_raw += 1
                did = add_paragraph(title, sents)
                para_ids.append(did)
                if title in gold_titles:
                    gold_ids.append(did)
                    docs[int(did[1:])]["is_gold"] = True
            # Every gold title must be present among the question's paragraphs.
            assert len(set(gold_ids)) == len(gold_titles), (ex["id"], gold_titles)
            out_questions[name].append({
                "qid": ex["id"], "question": ex["question"], "answer": ex["answer"],
                "type": ex["type"], "level": ex["level"],
                "gold_doc_ids": sorted(set(gold_ids)), "para_doc_ids": para_ids,
            })

    # Extra distractors come after the question paragraphs, so the doc_ids of the base
    # corpus are identical with or without them.
    # Their gold flags are not set, so corpus.jsonl stays byte-identical to core-v1.
    n_base_passages = len(docs)
    extra_questions = []
    for ex in sample(ds["train"], args.n_extra, args.seed, offset=args.n_train):
        gold_titles = set(ex["supporting_facts"]["title"])
        para_ids, gold_ids = [], []
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            n_paragraphs_raw += 1
            did = add_paragraph(title, sents)
            para_ids.append(did)
            if title in gold_titles:
                gold_ids.append(did)
        if len(set(gold_ids)) == len(gold_titles):  # skip the rare question whose golds dedup
            extra_questions.append({
                "qid": ex["id"], "question": ex["question"], "answer": ex["answer"],
                "type": ex["type"], "level": ex["level"],
                "gold_doc_ids": sorted(set(gold_ids)), "para_doc_ids": para_ids,
            })
    out_questions["train3000"] = out_questions["train"] + extra_questions

    with open(os.path.join(args.out_dir, "corpus.jsonl"), "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    for name, qs in out_questions.items():
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w") as f:
            for q in qs:
                f.write(json.dumps(q) + "\n")

    stats = {
        "dataset": DATASET, "config": CONFIG, "dataset_revision": revision,
        "seed": args.seed, "n_train": len(out_questions["train"]),
        "n_eval": len(out_questions["eval"]),
        "train_source_split": "train", "eval_source_split": "validation",
        "dedup_rule": "NFKC-lowercased, whitespace-collapsed title + first sentence",
        "n_extra_distractor_questions": args.n_extra,
        "n_paragraphs_raw": n_paragraphs_raw, "n_passages": len(docs),
        "n_passages_from_questions": n_base_passages,
        "n_gold_passages": sum(d["is_gold"] for d in docs),
        "gold_per_question": {
            name: sum(len(q["gold_doc_ids"]) for q in qs) / len(qs)
            for name, qs in out_questions.items()},
    }
    with open(os.path.join(args.out_dir, "corpus_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
