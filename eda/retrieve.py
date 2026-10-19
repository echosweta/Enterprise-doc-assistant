"""Metadata filtering -> dense + BM25 -> reciprocal rank fusion -> optional cross-encoder rerank."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .index import Index, content_terms
from .ingest import Chunk

RRF_K = 60
_reranker = None


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense: float
    bm25: float


@dataclass
class Retrieval:
    hits: list[Hit]
    confidence: float
    mode: str


def is_allowed(chunk: Chunk, role: str, include_old: bool = False) -> bool:
    role_ok = "all" in chunk.allowed_roles or role in chunk.allowed_roles
    return role_ok and (chunk.is_latest or include_old)


def _rank(scores: np.ndarray, mask: np.ndarray, n: int, positive_only: bool = False) -> list[int]:
    s = np.where(mask, scores, -np.inf)
    order = np.argsort(-s)[:n]
    return [int(i) for i in order if np.isfinite(s[i]) and (not positive_only or s[i] > 0)]


def rrf(rankings: list[list[int]]) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + pos + 1)
    return fused


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(config.RERANK_MODEL)
    return _reranker


def coverage(query: str, chunks: list[Chunk]) -> float:
    """Fraction of the query's content terms that appear in the retrieved text."""
    q = set(content_terms(query))
    if not q:
        return 0.0
    text = set(content_terms(" ".join(f"{c.title} {c.section} {c.text}" for c in chunks)))
    return len(q & text) / len(q)


def retrieve(index: Index, query: str, role: str = "employee", k: int = config.TOP_K,
             mode: str = "hybrid", rerank: bool = False, include_old: bool = False) -> Retrieval:
    from .index import tokenize

    mask = np.array([is_allowed(c, role, include_old) for c in index.chunks])
    qvec = index.embedder.encode_query(query)
    dense = np.einsum("ij,j->i", index.matrix, qvec)
    bm25 = index.bm25.scores(tokenize(query))

    n = config.CANDIDATES
    dense_rank = _rank(dense, mask, n)
    bm25_rank = _rank(bm25, mask, n, positive_only=True)

    if mode == "dense":
        order = dense_rank
        fused = {i: float(dense[i]) for i in order}
    elif mode == "bm25":
        order = bm25_rank
        fused = {i: float(bm25[i]) for i in order}
    else:
        fused = rrf([dense_rank, bm25_rank])
        order = sorted(fused, key=fused.get, reverse=True)[:n]

    if rerank and order:
        try:
            scores = _get_reranker().predict([(query, index.chunks[i].to_context()) for i in order])
            order = [i for _, i in sorted(zip(scores, order), key=lambda x: -x[0])]
            fused = {i: float(s) for s, i in zip(sorted(scores, reverse=True), order)}
        except Exception:
            pass  # reranker unavailable -> keep fused order

    order = order[:k]
    hits = [Hit(index.chunks[i], fused.get(i, 0.0), float(dense[i]), float(bm25[i])) for i in order]
    if not hits:
        return Retrieval([], 0.0, mode)
    top_dense = max(0.0, min(1.0, hits[0].dense))
    conf = 0.5 * top_dense + 0.5 * coverage(query, [h.chunk for h in hits[:2]])
    return Retrieval(hits, round(conf, 3), mode)
