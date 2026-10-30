"""BM25 (bm25s) and dense (sentence-transformers + FAISS flat) indexes over the corpus.

Both retrievers expose the same interface, which the baselines and the environment share:
    search(query, k) -> list[doc_id]
    search_batch(queries, k) -> list[list[doc_id]]

Passages are indexed as "title. text": HotpotQA titles carry the entity name, which is often
the only lexical anchor for the second hop.

Build:  python -m nanosearch.indexes            (writes to $GRPOP_ROOT/core-v1/indexes)
"""

import argparse
import json
import os

import numpy as np

from nanosearch import DATA_DIR, INDEX_DIR

DEFAULT_CORPUS = os.path.join(DATA_DIR, "corpus.jsonl")
DEFAULT_INDEX_DIR = INDEX_DIR
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
# bge v1.5 recommends this instruction on the query side only.
DENSE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_corpus(path=DEFAULT_CORPUS):
    with open(path) as f:
        docs = [json.loads(line) for line in f]
    return docs


def passage_text(doc):
    return f"{doc['title']}. {doc['text']}"


class BM25Retriever:
    def __init__(self, index_dir=DEFAULT_INDEX_DIR):
        import bm25s
        import Stemmer
        self.retriever = bm25s.BM25.load(os.path.join(index_dir, "bm25"))
        self.stemmer = Stemmer.Stemmer("english")
        with open(os.path.join(index_dir, "doc_ids.json")) as f:
            self.doc_ids = json.load(f)

    @staticmethod
    def build(docs, index_dir=DEFAULT_INDEX_DIR):
        import bm25s
        import Stemmer
        stemmer = Stemmer.Stemmer("english")
        tokens = bm25s.tokenize([passage_text(d) for d in docs], stopwords="en",
                                stemmer=stemmer, show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=False)
        retriever.save(os.path.join(index_dir, "bm25"))

    def search_batch(self, queries, k):
        import bm25s
        tokens = bm25s.tokenize(queries, stopwords="en", stemmer=self.stemmer,
                                show_progress=False)
        idx, _ = self.retriever.retrieve(tokens, k=k, show_progress=False, n_threads=1)
        return [[self.doc_ids[i] for i in row] for row in idx]

    def search(self, query, k):
        return self.search_batch([query], k)[0]


class DenseRetriever:
    def __init__(self, index_dir=DEFAULT_INDEX_DIR, device=None):
        import faiss
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(DENSE_MODEL, device=device)
        self.index = faiss.read_index(os.path.join(index_dir, "dense.faiss"))
        with open(os.path.join(index_dir, "doc_ids.json")) as f:
            self.doc_ids = json.load(f)

    @staticmethod
    def build(docs, index_dir=DEFAULT_INDEX_DIR, device=None, batch_size=64):
        import faiss
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(DENSE_MODEL, device=device)
        emb = model.encode([passage_text(d) for d in docs], batch_size=batch_size,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False).astype(np.float32)
        index = faiss.IndexFlatIP(emb.shape[1])  # exact; cosine since embeddings are unit norm
        index.add(emb)
        faiss.write_index(index, os.path.join(index_dir, "dense.faiss"))

    def search_batch(self, queries, k):
        q = self.model.encode([DENSE_QUERY_PREFIX + x for x in queries],
                              normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False).astype(np.float32)
        _, idx = self.index.search(q, k)
        return [[self.doc_ids[i] for i in row] for row in idx]

    def search(self, query, k):
        return self.search_batch([query], k)[0]


def get_retriever(name, index_dir=DEFAULT_INDEX_DIR, **kw):
    if name == "bm25":
        return BM25Retriever(index_dir)
    if name == "dense":
        return DenseRetriever(index_dir, **kw)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    os.makedirs(args.index_dir, exist_ok=True)

    docs = load_corpus(args.corpus)
    with open(os.path.join(args.index_dir, "doc_ids.json"), "w") as f:
        json.dump([d["doc_id"] for d in docs], f)
    BM25Retriever.build(docs, args.index_dir)
    print(f"bm25: indexed {len(docs)} passages")
    DenseRetriever.build(docs, args.index_dir, device=args.device)
    print(f"dense ({DENSE_MODEL}): indexed {len(docs)} passages")

    # Sanity: a passage's own title should retrieve it.
    for name in ("bm25", "dense"):
        r = get_retriever(name, args.index_dir, **({"device": args.device} if name == "dense" else {}))
        hits = r.search_batch([d["title"] for d in docs[:50]], k=9)
        rate = np.mean([d["doc_id"] in h for d, h in zip(docs[:50], hits)])
        print(f"{name}: title self-retrieval@9 on first 50 docs = {rate:.2f}")


if __name__ == "__main__":
    main()
