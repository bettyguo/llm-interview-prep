# 05 — Retrieval & RAG

RAG is the modal AI-application pattern in industry. Loops at companies shipping AI products probe RAG depth heavily — and the depth is more than "embed and lookup."

## What you should walk in able to do

- Contrast **dual-encoder vs. cross-encoder** retrieval and explain why production systems use both (retrieve with dual, rerank with cross).
- Reason about **chunking strategies** (fixed, recursive, semantic, hierarchical) and their failure modes.
- Compare **HNSW, IVF, IVF-PQ** index choices on the precision/recall/latency frontier.
- Explain **hybrid retrieval** (dense + BM25 + reciprocal rank fusion).
- Explain **reranking** — cross-encoder rerankers, LLM-as-reranker — and the quality/latency tradeoff.
- Explain **query rewriting / HyDE / multi-query expansion / query decomposition** — when each helps.
- Evaluate a RAG system end to end: retrieval metrics (recall@k, nDCG, MRR), generation metrics (faithfulness, answer relevance), system-level metrics.
- Diagnose RAG failure modes: lost-in-the-middle, distractor sensitivity, multi-hop, contradictory contexts.

## Questions

See [`questions.md`](questions.md).

## References (aggregated)

See [`references.md`](references.md).
