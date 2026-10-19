"""End-to-end: question -> retrieval -> confidence gate -> generation -> response."""
from __future__ import annotations

import time

from . import config
from .generate import INSUFFICIENT, generate, used_citations
from .index import Index
from .retrieve import retrieve

NOT_FOUND = "I couldn't find this in the company documents you have access to. Please contact HR or the IT helpdesk."


def get_index(rebuild: bool = False) -> Index:
    if not rebuild and config.INDEX_PATH.exists():
        try:
            return Index.load()
        except RuntimeError:
            pass
    from .ingest import load_corpus

    idx = Index(load_corpus())
    idx.save()
    return idx


def ask(index: Index, question: str, role: str = "employee", mode: str = "hybrid",
        rerank: bool = False, threshold: float | None = None) -> dict:
    t0 = time.perf_counter()
    threshold = config.CONFIDENCE_THRESHOLD if threshold is None else threshold
    r = retrieve(index, question, role=role, mode=mode, rerank=rerank)

    base = {"confidence": r.confidence, "role": role, "retrieved": [h.chunk.doc_id for h in r.hits]}
    if not r.hits or r.confidence < threshold:
        return {**base, "answered": False, "answer": NOT_FOUND, "citations": [], "reason": "low_confidence",
                "latency_ms": int((time.perf_counter() - t0) * 1000)}

    g = generate(question, r.hits)
    if g["answer"].strip().startswith(INSUFFICIENT):
        return {**base, "answered": False, "answer": NOT_FOUND, "citations": [], "reason": "model_refused",
                "latency_ms": int((time.perf_counter() - t0) * 1000)}

    return {**base, "answered": True, "answer": g["answer"], "model": g["model"],
            "citations": used_citations(g["answer"], r.hits),
            "latency_ms": int((time.perf_counter() - t0) * 1000)}
