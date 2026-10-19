"""Evaluate retrieval, refusal behaviour and (optionally) answer correctness.

Usage:
  python -m eval.run_eval                 # compare bm25 / dense / hybrid
  python -m eval.run_eval --rerank        # also hybrid + cross-encoder rerank
  python -m eval.run_eval --generate      # also check answers contain expected facts
  python -m eval.run_eval --sweep         # confidence-threshold sweep
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eda import config
from eda.pipeline import ask, get_index
from eda.retrieve import retrieve

QUESTIONS = Path(__file__).resolve().parent.parent / "data" / "eval" / "questions.json"


def retrieval_metrics(index, qs, mode, rerank=False, k=5):
    answerable = [q for q in qs if q["expected_doc"]]
    hits_at_k, rr, lat = 0, 0.0, []
    for q in answerable:
        t0 = time.perf_counter()
        r = retrieve(index, q["question"], role=q["role"], k=k, mode=mode, rerank=rerank)
        lat.append((time.perf_counter() - t0) * 1000)
        docs = [h.chunk.doc_id for h in r.hits]
        if q["expected_doc"] in docs:
            hits_at_k += 1
            rr += 1.0 / (docs.index(q["expected_doc"]) + 1)
    n = len(answerable)
    lat.sort()
    return {"mode": mode + ("+rerank" if rerank else ""), f"recall@{k}": hits_at_k / n, "mrr": rr / n,
            "p50_ms": lat[len(lat) // 2], "p95_ms": lat[int(len(lat) * 0.95) - 1]}


def gate_metrics(index, qs, threshold, mode="hybrid", rerank=False):
    """Answer rate on answerable questions vs refusal rate on unanswerable/access-denied ones."""
    ans = [q for q in qs if q["expected_doc"]]
    unans = [q for q in qs if not q["expected_doc"]]
    answered_ok = refused_ok = 0
    for q in ans:
        r = retrieve(index, q["question"], role=q["role"], mode=mode, rerank=rerank)
        answered_ok += bool(r.hits and r.confidence >= threshold)
    for q in unans:
        r = retrieve(index, q["question"], role=q["role"], mode=mode, rerank=rerank)
        refused_ok += not (r.hits and r.confidence >= threshold)
    return answered_ok / len(ans), refused_ok / len(unans)


def access_leaks(index, qs):
    leaks = 0
    for q in qs:
        r = retrieve(index, q["question"], role=q["role"])
        for h in r.hits:
            if "all" not in h.chunk.allowed_roles and q["role"] not in h.chunk.allowed_roles:
                leaks += 1
    return leaks


def md_table(rows):
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values()) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    qs = json.loads(QUESTIONS.read_text())
    index = get_index(rebuild=args.rebuild)
    print(f"Embedder: {index.embedder_name} | chunks: {len(index.chunks)} | questions: {len(qs)}\n")

    rows = [retrieval_metrics(index, qs, m) for m in ("bm25", "dense", "hybrid")]
    if args.rerank:
        rows.append(retrieval_metrics(index, qs, "hybrid", rerank=True))
    print("## Retrieval (answerable questions)\n" + md_table(rows) + "\n")

    para = [q for q in qs if q["type"] == "paraphrase"]
    print("## Paraphrase questions only (recall@5)")
    print(md_table([retrieval_metrics(index, para, m) for m in ("bm25", "dense", "hybrid")]) + "\n")

    print(f"## Access control leaks (should be 0): {access_leaks(index, qs)}\n")

    if args.sweep:
        sweep = []
        for t in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            a, r = gate_metrics(index, qs, t)
            sweep.append({"threshold": t, "answer_rate": a, "refusal_rate": r, "balanced": (a + r) / 2})
        print("## Confidence threshold sweep (pick the best 'balanced' value)\n" + md_table(sweep) + "\n")

    a, r = gate_metrics(index, qs, config.CONFIDENCE_THRESHOLD)
    print(f"## Gate at threshold {config.CONFIDENCE_THRESHOLD}: answers {a:.0%} of answerable, refuses {r:.0%} of unanswerable/forbidden\n")

    if args.generate:
        ok = total = 0
        for q in qs:
            res = ask(index, q["question"], role=q["role"])
            if q["expected_doc"] and q["must_contain"]:
                total += 1
                good = res["answered"] and q["must_contain"].lower() in res["answer"].lower()
                ok += good
                if not good:
                    print(f"  MISS: {q['question']!r} -> {res['answer'][:90]!r}")
        print(f"## Answer correctness (contains expected fact): {ok}/{total}")
        print("   (model:", res.get("model", "n/a"), ")")


if __name__ == "__main__":
    main()
