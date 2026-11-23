"""Exp 08: lexical scorers from term overlap up to BM25, on the frozen core-v1 corpus.

Every scorer shares one tokenisation (bm25s.tokenize, English stopwords, Snowball stemmer, as in
nanosearch.indexes) and one sparse doc x term count matrix. A scorer is three choices:

  tf    binary  1[tf > 0]
        raw     tf
        log     log(1 + tf)
        sat     tf / (tf + k1)
  idf   none | lucene   log(1 + (N - df + 0.5) / (df + 0.5)), the bm25s default
  norm  b=<x>  tf is divided by (1 - b + b * |d| / avgdl) BEFORE the tf transform
               (b=0: none; b=1: proportional to tf / |d|; sat with b=0.75 is BM25)
        cos    the weighted doc vector is L2-normalised AFTER tf and idf (classic vector space)

score(q, d) = sum over query tokens t (repeats count, as in bm25s) of w(t, d).
Ties are broken by corpus order (lower index first), so every ranking is deterministic.

  python experiments/08-lexical-ladder/ladder.py check            # sat/lucene/b=0.75 == bm25s index
  python experiments/08-lexical-ladder/ladder.py run --split eval # ladder + grid -> outputs/exp08/
  python experiments/08-lexical-ladder/ladder.py selftest         # toy corpus, no core-v1 needed
"""

import argparse
import itertools
import json
import os
import sys
from collections import Counter

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

K1 = 1.5   # bm25s defaults, used to build the core-v1 index
B = 0.75
K = 9

TF = ("binary", "raw", "log", "sat")
IDF = ("none", "lucene")
NORM = ("b=0", "b=0.75", "b=1", "cos")

# The ladder told as one path; every rung is also a grid cell (except 1s, stopwords kept).
LADDER = [
    ("L0", "term overlap",          dict(tf="binary", idf="none",   norm="b=0")),
    ("L1", "raw tf",                dict(tf="raw",    idf="none",   norm="b=0")),
    ("L1s", "raw tf, stopwords kept", dict(tf="raw",  idf="none",   norm="b=0", stopwords=False)),
    ("L2", "tf / doc length",       dict(tf="raw",    idf="none",   norm="b=1")),
    ("L3", "sublinear tf",          dict(tf="log",    idf="none",   norm="b=0")),
    ("L4", "idf of matched terms",  dict(tf="binary", idf="lucene", norm="b=0")),
    ("L5", "tf-idf, cosine",        dict(tf="log",    idf="lucene", norm="cos")),
    ("L6", "saturating tf x idf",   dict(tf="sat",    idf="lucene", norm="b=0")),
    ("L7", "BM25",                  dict(tf="sat",    idf="lucene", norm="b=0.75")),
]


def cfg_name(tf, idf, norm, stopwords=True):
    return f"tf={tf}_idf={idf}_{norm}" + ("" if stopwords else "_keepstop")


def tokenize(texts, stopwords):
    import bm25s
    import Stemmer
    return bm25s.tokenize(texts, stopwords="en" if stopwords else None,
                          stemmer=Stemmer.Stemmer("english"), return_ids=False,
                          show_progress=False)


class Counts:
    """Doc x term count matrix for one tokenisation."""

    def __init__(self, texts, stopwords=True):
        self.stopwords = stopwords
        toks = tokenize(texts, stopwords)
        self.vocab = {}
        rows, cols, vals = [], [], []
        for i, doc in enumerate(toks):
            for t, c in Counter(doc).items():
                rows.append(i)
                cols.append(self.vocab.setdefault(t, len(self.vocab)))
                vals.append(c)
        self.tf = sp.csr_matrix((np.asarray(vals, np.float64), (rows, cols)),
                                shape=(len(toks), len(self.vocab)))
        self.dl = np.asarray([len(d) for d in toks], np.float64)
        self.n_docs = len(toks)
        df = np.bincount(self.tf.indices, minlength=len(self.vocab))
        self.idf = np.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def query_matrix(self, queries):
        rows, cols, vals = [], [], []
        for i, q in enumerate(tokenize(queries, self.stopwords)):
            for t, c in Counter(q).items():
                if t in self.vocab:
                    rows.append(i)
                    cols.append(self.vocab[t])
                    vals.append(c)
        return sp.csr_matrix((np.asarray(vals, np.float64), (rows, cols)),
                             shape=(len(queries), len(self.vocab)))


def weights(c, tf="sat", idf="lucene", norm="b=0.75", k1=K1):
    """Doc x term weight matrix W, so that scores = Q @ W.T."""
    W = c.tf.copy()
    if norm.startswith("b="):
        b = float(norm[2:])
        pivot = 1 - b + b * c.dl / c.dl.mean()
        W = sp.diags(1.0 / pivot) @ W
    W = W.tocsr()
    x = W.data
    if tf == "binary":
        W.data = np.ones_like(x)
    elif tf == "log":
        W.data = np.log1p(x)
    elif tf == "sat":
        W.data = x / (x + k1)
    elif tf != "raw":
        raise ValueError(tf)
    if idf == "lucene":
        W = W @ sp.diags(c.idf)
    elif idf != "none":
        raise ValueError(idf)
    if norm == "cos":
        n = np.sqrt(np.asarray(W.multiply(W).sum(1)).ravel())
        W = sp.diags(1.0 / np.where(n > 0, n, 1.0)) @ W
    elif not norm.startswith("b="):
        raise ValueError(norm)
    return sp.csr_matrix(W)


def topk(scores, k):
    """Rows of top-k doc indices, ties by lower index; plus boundary-tie flags."""
    order = np.argsort(-scores, axis=1, kind="stable")
    top = order[:, :k]
    kth = np.take_along_axis(scores, order[:, k - 1:k + 1], 1)
    # The top-k set is not decided by the scores when rank k and k+1 tie.
    tie = kth[:, 0] == kth[:, 1]
    n_tied = (scores == kth[:, :1]).sum(1)
    return top, tie, n_tied


class LadderRetriever:
    """Same interface as nanosearch.indexes retrievers: search(query, k) -> list[doc_id]."""

    def __init__(self, counts, doc_ids, **cfg):
        self.counts, self.doc_ids = counts, doc_ids
        self.W_T = weights(counts, **cfg).T.tocsr()

    def scores(self, queries):
        return np.asarray((self.counts.query_matrix(queries) @ self.W_T).todense())

    def search_batch(self, queries, k):
        top, _, _ = topk(self.scores(queries), k)
        return [[self.doc_ids[i] for i in row] for row in top]

    def search(self, query, k):
        return self.search_batch([query], k)[0]


# ---------------------------------------------------------------------------------------------

def check_against_bm25s(counts, doc_ids, questions, index_dir, k=K):
    """sat/lucene/b=0.75 must give the stored bm25s index's scores and rankings."""
    import bm25s
    from nanosearch.indexes import BM25Retriever
    ref = BM25Retriever(index_dir)
    qs = [q["question"] for q in questions]
    ours = LadderRetriever(counts, doc_ids, tf="sat", idf="lucene", norm="b=0.75")
    s = ours.scores(qs)   # bm25s's Lucene tf has no (k1 + 1) factor either, so scores match as is
    toks = bm25s.tokenize(qs, stopwords="en", stemmer=ref.stemmer, show_progress=False)
    ref_idx, ref_scores = ref.retriever.retrieve(toks, k=k, show_progress=False, n_threads=1)
    ours_at_ref = np.take_along_axis(s, ref_idx, 1)
    max_abs = float(np.abs(ours_at_ref - ref_scores).max())
    top, _, _ = topk(s, k)
    set_diff = sum(set(a) != set(b) for a, b in zip(top.tolist(), ref_idx.tolist()))
    # A differing set is only acceptable where scores tie at rank k.
    kth = np.sort(s, 1)[:, ::-1][:, k - 1]
    unexplained = sum(set(a) != set(b) and not np.isclose(
        s[i, list(set(b) - set(a))], kth[i], rtol=1e-5).all()
        for i, (a, b) in enumerate(zip(top.tolist(), ref_idx.tolist())))
    out = {"n": len(qs), "max_abs_score_diff": max_abs, "topk_sets_differ": set_diff,
           "differ_not_explained_by_ties": int(unexplained)}
    print(json.dumps(out))
    ok = max_abs < 1e-3 and unexplained == 0
    print("CHECK PASS" if ok else "CHECK FAIL")
    return ok


def run(split, out_dir, data_dir, index_dir):
    from nanosearch.baselines import evaluate, load_questions
    from nanosearch.indexes import load_corpus, passage_text

    os.makedirs(out_dir, exist_ok=True)
    docs = load_corpus(os.path.join(data_dir, "corpus.jsonl"))
    with open(os.path.join(index_dir, "doc_ids.json")) as f:
        doc_ids = json.load(f)
    assert doc_ids == [d["doc_id"] for d in docs], "corpus order differs from the index"
    questions = load_questions(os.path.join(data_dir, f"{split}.jsonl"))
    texts = [passage_text(d) for d in docs]
    counts = {True: Counts(texts, stopwords=True), False: Counts(texts, stopwords=False)}

    if not check_against_bm25s(counts[True], doc_ids, questions, index_dir):
        sys.exit("BM25 reproduction failed; fix before reading any other number")

    cfgs = {cfg_name(**dict(tf=t, idf=i, norm=n)): dict(tf=t, idf=i, norm=n)
            for t, i, n in itertools.product(TF, IDF, NORM)}
    for _, _, cfg in LADDER:
        cfgs.setdefault(cfg_name(**cfg), cfg)

    qs = [q["question"] for q in questions]
    rows = []
    for name, cfg in cfgs.items():
        cfg = dict(cfg)
        stop = cfg.pop("stopwords", True)
        r = LadderRetriever(counts[stop], doc_ids, **cfg)
        summary, episodes = evaluate(name, r, questions)
        _, tie, n_tied = topk(r.scores(qs), K)
        summary.update(cfg, stopwords=stop, boundary_tie_rate=round(float(tie.mean()), 4),
                       median_docs_tied_at_rank9=int(np.median(n_tied)))
        with open(os.path.join(out_dir, f"{name}_{split}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(out_dir, f"{name}_{split}_episodes.jsonl"), "w") as f:
            for e in episodes:
                f.write(json.dumps(e) + "\n")
        rows.append(summary)
        print(json.dumps({k: summary[k] for k in ("system", "recall@9", "all_gold@9", "ndcg@9",
                                                   "boundary_tie_rate")}))
    print(f"wrote {len(rows)} configs to {out_dir}")


def selftest():
    docs = ["Paris is the capital of France. Paris Paris.",
            "The Eiffel Tower is in Paris and is 330 metres tall.",
            "Tower Tower Tower Tower Tower Tower Tower Tower.",
            "Berlin is the capital of Germany.",
            "A very long passage " + "about nothing in particular " * 30 + "Eiffel."]
    queries = ["How tall is the Eiffel Tower?", "capital of France", "tower tower"]
    import bm25s
    c = Counts(docs)
    ids = list(range(len(docs)))
    # 1. BM25 cell equals bm25s scores, repeated query terms included.
    ref = bm25s.BM25(k1=K1, b=B)
    import Stemmer
    ref.index(bm25s.tokenize(docs, stopwords="en", stemmer=Stemmer.Stemmer("english"),
                             show_progress=False), show_progress=False)
    ours = LadderRetriever(c, ids, tf="sat", idf="lucene", norm="b=0.75").scores(queries)
    for i, q in enumerate(tokenize(queries, True)):
        assert np.allclose(ours[i], ref.get_scores(q), atol=1e-5), (q, ours[i], ref.get_scores(q))
    # 2. Binary/no-idf/b=0 counts query tokens present; raw tf prefers the repeated-term doc.
    L0 = LadderRetriever(c, ids, tf="binary", idf="none", norm="b=0").scores(queries)
    L1 = LadderRetriever(c, ids, tf="raw", idf="none", norm="b=0").scores(queries)
    assert L0[0].argmax() == 1 and L1[0].argmax() == 2
    # 3. Ties broken by corpus order, tie flag set.
    top, tie, _ = topk(np.array([[1.0, 2.0, 1.0, 1.0]]), 2)
    assert top.tolist() == [[1, 0]] and tie.tolist() == [True]
    # 4. cos rows are unit norm; every grid cell builds.
    W = weights(c, tf="log", idf="lucene", norm="cos")
    assert np.allclose(np.sqrt(np.asarray(W.multiply(W).sum(1)).ravel()), 1)
    for t, i, n in itertools.product(TF, IDF, NORM):
        weights(c, tf=t, idf=i, norm=n)
    print("selftest PASS")


def main():
    from nanosearch import DATA_DIR, OUT_DIR
    from nanosearch.indexes import DEFAULT_INDEX_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "check", "run"])
    ap.add_argument("--split", default="eval")
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    ap.add_argument("--out_dir", default=os.path.join(OUT_DIR, "exp08"))
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "check":
        from nanosearch.baselines import load_questions
        from nanosearch.indexes import load_corpus, passage_text
        docs = load_corpus(os.path.join(a.data_dir, "corpus.jsonl"))
        with open(os.path.join(a.index_dir, "doc_ids.json")) as f:
            doc_ids = json.load(f)
        qs = load_questions(os.path.join(a.data_dir, f"{a.split}.jsonl"))
        ok = check_against_bm25s(Counts([passage_text(d) for d in docs]), doc_ids, qs, a.index_dir)
        sys.exit(0 if ok else 1)
    else:
        run(a.split, a.out_dir, a.data_dir, a.index_dir)


if __name__ == "__main__":
    main()
