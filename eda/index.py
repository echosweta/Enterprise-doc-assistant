"""Hybrid index: dense embeddings (numpy) + BM25 keyword scores, with metadata filtering."""
from __future__ import annotations

import math
import os
import pickle
import re
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from . import config
from .ingest import Chunk

STOPWORDS = set(
    "a an and are as at be by can do does for from how i in is it my of on or our the to was we what when where which who why will with you your "
    "get got have has had if that this there their them they me us about into than then so many much long often soon should would could also any".split()
)


def stem(tok: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            tok = tok[: -len(suf)]
            break
    if len(tok) > 4 and tok.endswith("e"):  # so "leave" and "leaves" both become "leav"
        tok = tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())]


def content_terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and t not in {stem(w) for w in STOPWORDS}]


# ---------------------------------------------------------------- embedders
class HashingEmbedder:
    """Offline fallback: signed feature hashing of unigrams + bigrams. No model download needed.
    Use a real embedding model (sentence-transformers) for production quality."""

    name = "hashing-384"
    dim = 384

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = [t for t in tokenize(text) if t not in STOPWORDS]
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = zlib.crc32(f.encode())
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class STEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.query_prefix = "Represent this sentence for searching relevant passages: " if "bge" in model_name else ""

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([self.query_prefix + text])[0]


def get_embedder():
    if os.getenv("FORCE_HASH_EMBEDDINGS") == "1":
        return HashingEmbedder()
    try:
        return STEmbedder(config.EMBED_MODEL)
    except Exception:
        return HashingEmbedder()


# --------------------------------------------------------------------- BM25
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf), dtype=np.float32)
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


# -------------------------------------------------------------------- index
class Index:
    def __init__(self, chunks: list[Chunk], embedder=None):
        self.chunks = chunks
        self.embedder = embedder or get_embedder()
        texts = [f"{c.title}. {c.section}. {c.text}" for c in chunks]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.matrix = self.embedder.encode(texts)
        self.embedder_name = self.embedder.name

    def save(self, path: Path = config.INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25, "matrix": self.matrix, "embedder": self.embedder_name}, f)

    @classmethod
    def load(cls, path: Path = config.INDEX_PATH) -> "Index":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls.__new__(cls)
        obj.chunks, obj.bm25, obj.matrix, obj.embedder_name = d["chunks"], d["bm25"], d["matrix"], d["embedder"]
        obj.embedder = get_embedder()
        if obj.embedder.name != obj.embedder_name:
            raise RuntimeError(f"Index built with {obj.embedder_name} but runtime embedder is {obj.embedder.name}. Rebuild the index.")
        return obj
