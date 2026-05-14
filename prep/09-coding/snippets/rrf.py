"""Reciprocal Rank Fusion for hybrid retrieval.

Companion to the question "Implement Reciprocal Rank Fusion (RRF) for hybrid retrieval"
in prep/09-coding/questions.md.

Run:
    python rrf.py
"""
from __future__ import annotations

from collections import defaultdict


def rrf(rankings: list[list[int]], k: int = 60, top_k: int = 10) -> list[tuple[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]


if __name__ == "__main__":
    dense = [3, 7, 1, 4, 5]
    bm25 = [1, 2, 4, 7, 9]
    fused = rrf([dense, bm25], k=60, top_k=5)
    print("fused top-5:")
    for doc_id, score in fused:
        print(f"  doc={doc_id} score={score:.6f}")
    # doc 7 and doc 1 both appear in top 3 of both rankings; they should rank high.
    fused_ids = [d for d, _ in fused]
    assert 1 in fused_ids and 7 in fused_ids
    print("rrf OK")
