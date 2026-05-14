# D4 — E-commerce semantic search

A worked drill following the [6-step structure](../README.md#the-canonical-6-step-structure).

---

### Q: Design a semantic search system for an e-commerce site (Amazon-like).

**Category:** system-design
**Difficulty:** senior
**Tags:** [search, e-commerce, hybrid-retrieval, learning-to-rank]

**Short answer.** **Hybrid retrieval** (BM25 + dense) → **cross-encoder rerank** → **multi-objective learning-to-rank** (relevance, conversion likelihood, business rules: in-stock, shipping speed, margin). Personalize via user features (purchase history, browsing). Calibrated for both relevance *and* conversion — pure relevance wins the SIGIR demo but pure conversion bias hides the products users actually need.

**Expansion / why this is the answer.**

**1. Clarify requirements.**
Functional:
- Free-text query → ranked product list.
- Handle typos, synonyms, exact-string matches (SKUs, brand names).
- Personalized to user.
- Multi-language / multi-region.

Non-functional:
- Latency: p99 < 150 ms.
- Throughput: 10k+ QPS at peak.
- Catalog: 100M+ items.
- Freshness: new products discoverable within 1 hour.

Clarifying Qs:
- "Do we sell physical goods, digital, both?"
- "Are we optimizing for click, add-to-cart, or revenue?"
- "How important is exact-keyword match vs. semantic match?"

**2. Define metrics.**

Online:
- **North-star**: revenue per session / GMV.
- **Conversion rate**: % searches → purchase.
- **CTR@k**: which positions clicked.
- **Add-to-cart rate** from search.
- **Reformulation rate**: high reformulation = bad search.

Offline:
- **Recall@k**: did the ground-truth purchased item appear in top-k retrieved?
- **nDCG@10**: position-weighted relevance.
- **MRR**.

Online ↔ offline: tricky because the "right answer" depends on user intent (browsing vs. buying), which the metric must capture.

**3. Data and labels.**

Sources:
- Query logs: query → clicked items → cart-added → purchased.
- Product catalog: title, description, attributes, images, brand, price, category, in-stock, ratings.
- User features: history, location, segments.

Labels:
- **Positive**: clicked, added-to-cart, purchased.
- **Negative**: skipped impression, dwell < 2s.
- **Hard negatives**: items shown but not clicked → useful for training rerankers.

Class imbalance: most impressions don't convert; weighted training.

Decontamination:
- Don't leak post-click features back into training.
- Time-ordered splits.

**4. Modeling.**

**4a. Retrieval (recall ~1k candidates from 100M+ items)**:
- **BM25 index** on product titles + descriptions + key attributes (brand, model number).
- **Dense embeddings**: dual-encoder; product = encode(title + description + attributes); query = encode(query). Train with contrastive loss on (query, purchased_product) pairs.
- **Hybrid fusion** (RRF or weighted score) over BM25 + dense.
- **Field-weighted BM25**: title 5×, description 1×, brand 3×.

Critical: exact-string queries (`iPhone 15 Pro 256GB`, `B07GS76YR`) need BM25; "comfortable running shoes for flat feet" needs dense.

**4b. Reranking (top ~100 candidates from retrieval)**:
- **Cross-encoder** (BERT-small or distilled): score (query, product-text) pairs.
- **Learning-to-rank** (LightGBM or DNN) with features:
  - Cross-encoder relevance score.
  - User-product affinity (user × category history).
  - Product features: price, rating, review count, in-stock, shipping speed.
  - Conversion rate over the last 30 days.
  - Margin (for tie-breaks, with caution).
- **Loss**: LambdaRank (pairwise) or listwise.

**4c. Re-rank business rules**:
- Diversity: don't return 10 of the same model.
- Stock-aware: out-of-stock items downranked.
- Sponsored items injected per business rules.
- Cold-start: new products get exploration boost.

**5. Serving.**

Latency budget: < 150 ms p99.
- BM25 (Elasticsearch / OpenSearch): ~20 ms.
- Dense ANN (HNSW): ~10 ms.
- Cross-encoder reranking 100 candidates: ~30 ms.
- LTR + business rules: ~5 ms.
- Network / orchestration: ~20 ms buffer.

Infrastructure:
- Search core (Vespa, Elasticsearch).
- Vector index (HNSW per shard).
- Feature store for personalization.
- Cross-encoder serving (GPU-batched).

Catalog freshness:
- New product indexed within minutes; embedding generated; HNSW upsert.

**6. Monitoring + iteration.**

Drift:
- Seasonality (holiday queries shift).
- New product categories appear.

Retraining:
- Embeddings: weekly.
- Reranker / LTR: daily.
- BM25: as new products are added; no retraining of the model.

Online evaluation:
- A/B every change.
- Slice by query intent (navigational, transactional, informational) — wins on one can hide regressions on another.
- Long-term: 7-day retention, return rate post-purchase.

Search-quality watchouts:
- **Filter bubbles**: heavy personalization can hide diversity. Maintain a baseline "objective" relevance signal.
- **Spam / SEO products**: low-quality products gaming the system.
- **Query understanding**: "running shoes" should not return socks (semantic-match failure).

---

**Common follow-ups.**

- "What if exact-product SKU search is missing dense-retrieval?" → BM25 catches it. The case for hybrid.
- "How do you handle long-tail queries (rare, only-once)?" → Dense retrieval handles novel queries; BM25 alone fails on misspellings.
- "How would you add LLM-based query rewriting?" → Reformulate ambiguous queries; use it conservatively (latency cost).

**Common mistakes.**

- Dense-only retrieval; misses exact-keyword queries.
- No reranking; top-k dense alone is noisy.
- Ignoring stock / business rules.
- Skipping personalization but claiming "personalized search."

**References.**

- [Robertson & Zaragoza — BM25 reference](https://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf) — BM25.
- [Karpukhin et al. — "DPR"](https://arxiv.org/abs/2004.04906) — dense retrieval for QA.
- [Burges — "From RankNet to LambdaRank to LambdaMART"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf) — LTR canonical reference.
- [Vespa / Elasticsearch docs](https://docs.vespa.ai/) — primary search infrastructure docs.
