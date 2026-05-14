# Retrieval & RAG — questions

Entries follow the [Q&A schema](../../CONTRIBUTING.md#the-qa-entry-schema).

---

### Q: Walk me through a production RAG pipeline end to end.

**Category:** concept
**Difficulty:** mid
**Tags:** [rag, pipeline, retrieval, reranking]

**Short answer.** **Ingest**: chunk documents → embed → write to a vector index (+ a text/BM25 index). **Query**: rewrite if needed → embed → retrieve top-k candidates (hybrid: dense + BM25 with RRF or another fusion) → rerank top-N candidates with a cross-encoder → assemble retrieved context into a prompt → call the LLM → optionally cite. **Eval**: track retrieval metrics (recall@k, nDCG), generation metrics (faithfulness, answer relevance), and end-to-end task metrics.

**Expansion / why this is the answer.**
- **Ingest**
  - **Document parsing**: PDFs/HTML/Word → text. Failure mode: lose structure (tables, code blocks) — use parsers that preserve it (Unstructured, LlamaParse).
  - **Chunking**: 200–800 tokens is the modal range. Strategy matters (see chunking question).
  - **Embedding**: a dense encoder (OpenAI text-embedding-3-large, Cohere embed-v3, BGE, GTE, E5).
  - **Storage**: vector index (HNSW / IVFPQ via FAISS, pgvector, Qdrant, Pinecone, Weaviate) + a sparse/text index (Elasticsearch, OpenSearch, Tantivy).
  - **Metadata**: document id, source URL, timestamp, ACL — for filtering and citation.
- **Query path**
  - **Query rewrite / expansion** (HyDE, multi-query): optional but helps on ambiguous or short queries.
  - **Hybrid retrieval**: dense (semantic) + sparse (BM25); fuse with Reciprocal Rank Fusion (RRF) or weighted score.
  - **Filtering**: by metadata (date, ACL, source).
  - **Top-k**: 20–100 candidates typically.
  - **Reranking**: cross-encoder (BGE-reranker, Cohere Rerank) cuts top-k → top-N (e.g. 5–10).
  - **Prompt assembly**: system prompt + retrieved passages (each with a [source-id]) + user query.
  - **LLM generation**: with instructions to cite [source-id] for any factual claim.
- **Eval** (see Topic 07): retrieval recall@k, MRR, nDCG; generation faithfulness (does the answer follow from the cited passages?); answer relevance.

**Common follow-ups.**
- "What's RRF?" → Reciprocal Rank Fusion (Cormack et al. 2009): `score(d) = Σ_lists 1/(k + rank_list(d))`. Robust to score-scale mismatches.
- "When do you skip reranking?" → Low latency budget, very narrow domain where dense retrieval is reliable.
- "How do you handle multi-turn?" → Often by query rewriting that incorporates the prior turn — "what is its CEO?" → "what is Acme's CEO?"

**Common mistakes.**
- Embed-and-pray: dense-only retrieval with no BM25 fallback. BM25 catches exact-string queries (product SKUs, names) that dense embeddings miss.
- Indexing too-large chunks; the model can't make use of context that fragmented signals across.
- No reranking: top-k dense alone has known precision issues at typical k.

**References.**
- [Lewis et al. — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG)](https://arxiv.org/abs/2005.11401) — the original RAG paper.
- [Cormack, Clarke, Buettcher — "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF.
- [Gao et al. — "Retrieval-Augmented Generation for Large Language Models: A Survey"](https://arxiv.org/abs/2312.10997) — survey.

---

### Q: Compare dual-encoder vs. cross-encoder retrievers. When do you use each?

**Category:** concept
**Difficulty:** mid
**Tags:** [dual-encoder, cross-encoder, retrieval, reranking]

**Short answer.** **Dual-encoder** (bi-encoder): embed the query and each document independently into a fixed vector; similarity is a dot product or cosine. Fast at index time (precompute all doc vectors), cheap at query time (one embed + ANN lookup) — *the* shape used for first-stage retrieval. **Cross-encoder**: feed `[query; doc]` jointly into a single encoder; output a relevance score. Much higher quality but `O(N_candidates)` forward passes per query — used for reranking the top-k from a dual-encoder.

**Expansion / why this is the answer.**
- **Dual-encoder**:
  - Examples: SBERT, BGE, E5, GTE, OpenAI text-embedding-3, Cohere embed-v3.
  - Quality: limited by the "two towers" not seeing each other; can miss subtle interactions.
  - Speed: query embed (one forward) + ANN search over `N_docs` precomputed vectors → milliseconds per query for millions of docs.
- **Cross-encoder**:
  - Examples: BGE-reranker, Cohere Rerank, Voyage rerank, cross-encoder/ms-marco-MiniLM-L-12-v2.
  - Quality: typically +5–15 nDCG points over dual-encoder alone in production benchmarks.
  - Speed: `N_candidates` forward passes (50–100 typical), each requiring the full query+doc pair → tens to hundreds of milliseconds.
- **Production pattern**: dual-encoder retrieves top-100, cross-encoder reranks to top-5.
- **ColBERT** (Khattab & Zaharia 2020): a middle ground — token-level "late interaction" embeddings; better quality than dual-encoder at a moderate compute increase. PLAID (Santhanam et al. 2022) makes it production-fast.

**Common follow-ups.**
- "Why don't you just use cross-encoder for retrieval?" → Cost: linear in document count per query; intractable for >10k-doc indices.
- "What's ColBERT-style late interaction?" → Per-token MaxSim over precomputed token embeddings; better than dual-encoder, much faster than cross-encoder. Storage ~10× a dual-encoder index.

**Common mistakes.**
- Treating dual-encoder embedding similarity as "the meaning" — it's a coarse approximation that misses fine-grained relevance.
- Trying to rerank with the same dual-encoder model (same architecture, same training); a cross-encoder is a *different* model trained on pairs.

**References.**
- [Reimers & Gurevych — "Sentence-BERT" (SBERT)](https://arxiv.org/abs/1908.10084) — dual-encoder.
- [Nogueira & Cho — "Passage Re-ranking with BERT"](https://arxiv.org/abs/1901.04085) — cross-encoder reranking.
- [Khattab & Zaharia — "ColBERT"](https://arxiv.org/abs/2004.12832) — late interaction.

---

### Q: Walk me through chunking strategies. Why does it matter?

**Category:** concept
**Difficulty:** mid
**Tags:** [chunking, retrieval, granularity]

**Short answer.** Chunking is choosing how to split documents into retrievable units. **Fixed-size**: split every N tokens — simple, baseline. **Recursive**: split on hierarchical separators (sections, paragraphs, sentences), falling back as needed — the modal default. **Semantic**: cluster sentences by embedding similarity, split at low-similarity boundaries. **Hierarchical**: small "child" chunks for retrieval precision + larger "parent" chunks for context (small-to-big). The right strategy depends on document type, query shape, and the LLM's context limit. Wrong chunking is the single biggest reason "my RAG doesn't work."

**Expansion / why this is the answer.**
- **Why it matters**:
  - Chunks too small: lose context; embedding can't represent the meaning; the LLM gets fragments.
  - Chunks too large: dilute the signal; the relevant fact is buried; recall@k drops because the index is coarse.
  - Bad chunk boundaries: split a code block, a table, or a sentence mid-thought.
- **Strategies**:
  - **Fixed-size** (e.g. 512 tokens, 50 overlap): trivial; ignores structure.
  - **Recursive character/token split** (LangChain default): hierarchical separators (`\n\n`, `\n`, `. `, ` `); guarantees a max size while respecting structure when possible.
  - **Sentence / paragraph based**: respect natural boundaries; size variance is the cost.
  - **Semantic chunking**: embed sentences, split at points of low cosine similarity (large semantic shift). Higher cost; modest quality gain.
  - **Hierarchical / parent-child (RAPTOR, Sarthi et al. 2024; "small-to-big")**: index small chunks for precision; on hit, return the parent chunk for context.
  - **Markdown- / code-aware**: preserve headings, code fences, tables.
  - **Token-budgeted** (for very long docs): split with a target token count *and* preserve metadata-aware boundaries.
- **Failure modes**:
  - Token-counted on chars: badly off for non-Latin scripts.
  - Splitting tables mid-row.
  - Stripping doc structure (headings) that gives crucial context.

**Common follow-ups.**
- "What's RAPTOR?" → Recursive Abstractive Processing for Tree-Organized Retrieval — build a tree by recursively summarizing clusters of chunks; retrieve at the level of granularity that fits the query.
- "How do you pick chunk size empirically?" → Sweep on a held-out eval set; the optimum is task- and doc-dependent (200–800 tokens common).

**Common mistakes.**
- Picking a chunk size without measuring.
- Forgetting overlap; sentences at chunk boundaries get lost.
- Ignoring chunk-level metadata (source, section title).

**References.**
- [Sarthi et al. — "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"](https://arxiv.org/abs/2401.18059) — hierarchical chunking.
- [LangChain — text-splitter docs](https://python.langchain.com/docs/concepts/text_splitters/) — practical chunking strategies.

---

### Q: Compare HNSW, IVF, and IVFPQ as ANN index types. What do they cost?

**Category:** concept
**Difficulty:** senior
**Tags:** [ann, hnsw, ivf, faiss]

**Short answer.** **HNSW**: hierarchical graph (small-world property); fast query, high recall, moderate memory. **IVF** (inverted file): partition vectors into Voronoi cells (k-means); search a few nearest cells. Smaller memory, lower recall at the same speed than HNSW for similar quality, but easier to update. **IVFPQ**: IVF + product quantization (compress vectors to bytes/codes). Drastically lower memory at the cost of recall (controlled with the number of refinement steps).

**Expansion / why this is the answer.**
- **Brute-force**: exact, `O(N)` per query; ground-truth baseline. Used at small scales (<1M vectors) or for re-scoring.
- **HNSW** (Malkov & Yashunin 2016):
  - Multi-layer graph; entry at top layer; greedy descent.
  - Pros: high recall (>95%) at low latency; widely available (FAISS, hnswlib, pgvector, Qdrant default).
  - Cons: memory = `O(M · N)` for `M` edges per node (typical M=16–32); slow to build at very large scale.
- **IVF** (FAISS):
  - K-means cluster centroids; assign each vector to its nearest centroid.
  - Search: probe the `nprobe` nearest centroids' vectors.
  - Pros: smaller memory; updateable.
  - Cons: recall-latency frontier is generally worse than HNSW at high recall targets.
- **IVFPQ** (Jégou, Douze, Schmid 2010): IVF + Product Quantization. Each vector's residual (after centroid subtraction) is quantized into `m` subvectors, each represented by 1 byte (256 codewords).
  - 384-dim float (1.5 KB) → 64-byte code: 24× compression.
  - Pros: massive memory savings — billions of vectors fit on a single machine.
  - Cons: recall drops noticeably without `m` tuning and a re-rank step (rerank top-N with original vectors).
- **ScaNN** (Google; Guo et al. 2020): asymmetric hashing + anisotropic quantization; often the recall-latency leader on internal Google benchmarks.
- **Decision rubric**:
  - <1M vectors, latency-sensitive: HNSW (or brute force on GPU).
  - 1M–100M vectors, RAM-limited: IVF or HNSW with tighter `M`.
  - 100M+ vectors, must fit in commodity RAM: IVFPQ or ScaNN.

**Common follow-ups.**
- "What's `efSearch` in HNSW?" → Search-time parameter; higher = more candidates explored = higher recall and higher latency.
- "What's the rerank step in IVFPQ?" → Retrieve top-K_coarse via PQ codes; recompute exact distances for the top-K_coarse; return the best K. Mostly fixes recall loss.

**Common mistakes.**
- Calling all ANN "HNSW" — there are many indexes, with different memory/quality/latency frontiers.
- Reporting recall numbers without specifying the index parameters (`efSearch`, `nprobe`).

**References.**
- [Malkov & Yashunin — "Efficient and robust approximate nearest neighbor search using HNSW"](https://arxiv.org/abs/1603.09320) — HNSW.
- [Jégou, Douze, Schmid — "Product Quantization for Nearest Neighbor Search"](https://hal.inria.fr/inria-00514462/document) — PQ.
- [Johnson, Douze, Jégou — "Billion-scale similarity search with GPUs" (FAISS)](https://arxiv.org/abs/1702.08734) — FAISS.

---

### Q: What is hybrid retrieval, and why does it usually beat pure dense retrieval?

**Category:** concept
**Difficulty:** mid
**Tags:** [hybrid-search, bm25, dense-retrieval, rrf]

**Short answer.** Hybrid retrieval combines dense (embedding) retrieval with sparse (BM25) retrieval, fusing their results via Reciprocal Rank Fusion (RRF) or weighted scoring. Dense retrieval captures semantic similarity ("car" matches "automobile"); BM25 captures exact-token signals ("ABC-12345" matches "ABC-12345"). Real queries have both — names, codes, jargon need exact matches; intent and rewording need semantic. Hybrid is consistently 5–15% better than dense alone on production-shape benchmarks.

**Expansion / why this is the answer.**
- **BM25** (Robertson & Zaragoza 2009): the canonical sparse-bag-of-words scoring function — built on TF-IDF with length normalization and term saturation. Implemented in Elasticsearch, OpenSearch, Tantivy.
- **Dense retrieval**: encoder → vector → ANN search. Captures meaning, fails on exact-string and rare-token queries.
- **Hybrid fusion**:
  - **RRF**: `score(d) = Σ_lists 1/(k + rank_list(d))` (k typical 60). Scale-free; no tuning of weighting.
  - **Weighted score fusion**: `α · dense_score + (1−α) · bm25_score`. Needs score normalization; α tuned per dataset.
  - **Learned fusion**: train a small model to combine the two.
- **When pure dense wins**: very abstract queries on a domain-tuned encoder, where exact strings don't matter.
- **When pure BM25 wins**: heavily-named-entity queries (legal cases, product SKUs, scientific identifiers).
- **The case for always going hybrid in production**: the failure modes are independent; combining is cheap insurance.
- **Modern variant**: **SPLADE** (Formal et al. 2021) — learned sparse retrieval that uses BERT to produce sparse vector representations; competitive with dense retrieval on its own, and complementary in hybrid.

**Common follow-ups.**
- "What's BM25's TF saturation parameter `k1`?" → Controls how quickly term-frequency contribution saturates; typical `k1=1.2`.
- "What's SPLADE?" → Sparse Lexical AnD Expansion — a learned, sparse retrieval model. Combines the explainability of sparse with some semantic capacity.

**Common mistakes.**
- Reporting "dense beats BM25" without measuring on a real-shape query mix that includes named-entity / exact-string queries.
- Mixing scores without normalization in weighted fusion.

**References.**
- [Robertson & Zaragoza — "The Probabilistic Relevance Framework: BM25 and Beyond"](https://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf) — BM25 reference.
- [Cormack et al. — RRF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — fusion.
- [Formal et al. — "SPLADE"](https://arxiv.org/abs/2107.05720) — learned sparse retrieval.

---

### Q: What is HyDE (Hypothetical Document Embeddings), and when does it help?

**Category:** concept
**Difficulty:** mid
**Tags:** [query-rewriting, hyde, retrieval]

**Short answer.** HyDE (Gao et al. 2022): rather than embedding the user's (often short, ambiguous) query, ask an LLM to generate a *hypothetical answer* to the query, then embed *that* and use it for retrieval. The hypothetical answer is "in the shape of" the documents we want to match, so its embedding lands closer to relevant passages. Helps most on short or under-specified queries where direct query-embedding misses.

**Expansion / why this is the answer.**
- The mechanism:
  1. User query: "What did the Apollo program achieve?"
  2. LLM is prompted: "Write a passage answering: 'What did the Apollo program achieve?'"
  3. LLM generates a paragraph as if it were a Wikipedia entry.
  4. Embed that paragraph (not the query); search the dense index.
- Why it works: the dense embedding model was trained on (query, document) pairs where the document is longer, more narrative, and more "answer-shaped" than the query. The hypothetical answer lands in the same regime.
- **Where HyDE helps**:
  - Short, abstract queries.
  - Conversational queries that drop context ("What about its CEO?").
  - Cross-lingual retrieval (the LLM can generate in the doc's language).
- **Where HyDE hurts**:
  - Queries with named entities the LLM might hallucinate.
  - High-precision exact-match queries (where BM25 wins).
  - Latency-sensitive paths (one extra LLM call per query).
- **Variants**:
  - **Multi-query**: ask the LLM to generate `k` different rewritings; embed all; union top-K.
  - **Query decomposition**: split a complex query into sub-queries, retrieve for each.

**Common follow-ups.**
- "Why not just use a more powerful query-aware embedding?" → Instruction-aware embedding models (e.g. E5-mistral-instruct) close some of this gap; HyDE is a free-lunch alternative if you don't have one.
- "What if the LLM hallucinates the hypothetical answer?" → Often fine — we're using it as a query, not as truth. The retrieved docs ground the actual answer.

**Common mistakes.**
- Using HyDE indiscriminately; on simple queries it's wasted latency.
- Forgetting to use the hypothetical's embedding, not its text, for search.

**References.**
- [Gao et al. — "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE)](https://arxiv.org/abs/2212.10496) — HyDE paper.

---

### Q: What is the "lost-in-the-middle" problem, and how do you mitigate it?

**Category:** concept
**Difficulty:** mid
**Tags:** [lost-in-the-middle, long-context, prompt-design]

**Short answer.** Liu et al. (2023) showed that LLMs use information at the start and end of a long context more reliably than information in the middle — a "U-shaped" attention pattern. Implication: in RAG, the order of retrieved passages matters. Mitigations: place the most-relevant passage at the start (or end), keep retrieved context tight (fewer passages, more reranking), and don't pad the context with marginally-relevant retrievals.

**Expansion / why this is the answer.**
- The empirical finding: across multiple long-context LLMs (GPT-3.5/4, Claude, open models), accuracy on a needle-in-haystack-style task drops when the relevant information is in the middle of the context.
- **The mechanism (hypothesized)**:
  - Models trained on shorter contexts; positional encodings are most reliable at the boundaries.
  - Attention pattern: early and late positions get more reliable attention from many tokens.
  - Even RoPE-based long-context fine-tunes show residual U-shape.
- **Mitigations in RAG**:
  - **Order passages by reranker score**: most relevant first.
  - Some practitioners reverse and put most-relevant *last* (some models attend most to the end). Test empirically.
  - **Re-rank aggressively**: 100 dense candidates → 5 reranked → only 5 in context.
  - **Bound context size**: don't dump 50 passages "to be safe." More context dilutes the signal.
  - **Highlight or label the most-relevant passage** ("This passage is most likely to contain the answer:") — modest empirical gains.
- **Modern caveat (2024–2026)**: frontier models trained explicitly on long-context (Claude 3.x, Gemini 1.5/2.0, GPT-4-turbo) have largely flattened the U-curve on synthetic needle-in-haystack tasks — but the effect remains for noisier, real-world RAG.

**Common follow-ups.**
- "What's needle-in-a-haystack?" → Insert one sentence in a long context, ask about it; measure recall as a function of position. Standard long-context eval.
- "How do you measure lost-in-the-middle in your own RAG?" → Inject a synthetic correct passage at varying positions; measure answer accuracy by position.

**Common mistakes.**
- Believing "long context = problem solved." It's necessary but not sufficient.
- Padding context with marginal retrievals on the theory that "more info is better."

**References.**
- [Liu et al. — "Lost in the Middle"](https://arxiv.org/abs/2307.03172) — the canonical paper.
- [Kamradt — Needle in a haystack benchmark](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — community eval framework.

---

### Q: How do you evaluate a RAG system end to end?

**Category:** concept
**Difficulty:** senior
**Tags:** [rag-eval, faithfulness, recall, ndcg]

**Short answer.** Separate the **retrieval** and **generation** axes, plus the **end-to-end task** axis. Retrieval: recall@k, MRR, nDCG over a labeled set of (query, relevant-passage) pairs. Generation: faithfulness (does the answer follow from the cited passages?), answer relevance (does it actually answer the query?), context precision (are retrieved passages used?). End-to-end: task accuracy on a held-out eval set, plus user-facing metrics (CSAT, deflection rate).

**Expansion / why this is the answer.**
- **Retrieval metrics** (need labeled data — at least pairs of `(query, gold passage)`):
  - **Recall@k**: did the gold passage appear in the top-k? Most-used in RAG.
  - **Precision@k**: of the top-k, how many were truly relevant?
  - **MRR (Mean Reciprocal Rank)**: `1/rank_of_first_relevant`. Good for "did the first hit matter?"
  - **nDCG (normalized Discounted Cumulative Gain)**: position-weighted; the IR-classical metric.
- **Generation metrics**:
  - **Faithfulness / groundedness**: every claim in the answer is supported by some retrieved passage. Measured with an LLM-as-judge or factual-entailment classifier (FActScore, RAGAS).
  - **Answer relevance**: did the answer address the query?
  - **Context precision / utilization**: fraction of retrieved passages used in the answer.
- **End-to-end / task metrics**:
  - **Task accuracy** on a held-out eval set with ground-truth answers.
  - **User-facing**: thumbs up/down, dwell time, repeat questions.
  - **Hallucination rate**: faithfulness's flip side.
- **Frameworks**: RAGAS (Es et al. 2023), TruLens, Phoenix, DeepEval — wrap LLM-as-judge into reusable metric pipelines.
- **Eval set construction**: hand-label 100–500 (query, gold-passage, gold-answer) triples; generation-rate via LLM is acceptable for early iterations but human-validate.

**Common follow-ups.**
- "What does RAGAS measure exactly?" → Faithfulness, answer relevance, context precision, context recall — all via LLM-as-judge prompts. Treat as a fast signal, calibrate against a smaller hand-graded set.
- "How do you measure retrieval recall when you don't have labels?" → You can construct synthetic labels: for each known doc, generate a question whose answer is in it; use that as the gold pair. Cheap, noisy, but useful for relative comparison.

**Common mistakes.**
- Measuring only end-to-end task accuracy — you can't tell whether to fix retrieval or generation.
- Trusting LLM-as-judge without calibration.

**References.**
- [Es et al. — "RAGAS: Automated Evaluation of Retrieval Augmented Generation"](https://arxiv.org/abs/2309.15217) — RAGAS.
- [Manning, Raghavan, Schütze — *Introduction to Information Retrieval*, §8](https://nlp.stanford.edu/IR-book/) — IR metrics canonical reference.
- [Min et al. — "FActScore"](https://arxiv.org/abs/2305.14251) — faithfulness measurement.

---

### Q: Compare instruction-aware vs. instruction-naive embedding models.

**Category:** concept
**Difficulty:** mid
**Tags:** [embeddings, instruction-tuned-embeddings, e5]

**Short answer.** **Instruction-naive** embeddings (early SBERT, BGE, GTE) embed a string into a single vector — same embedding regardless of intent. **Instruction-aware** embeddings (E5, E5-Mistral, Cohere embed-v3, OpenAI text-embedding-3) accept a task instruction prepended to the input — "Represent this query for retrieval:" or "Represent this passage:" — and produce different embeddings tailored to the task. Instruction-aware models match or beat naive ones on every standard benchmark (MTEB).

**Expansion / why this is the answer.**
- **The trick**: at training time, the dual-encoder is given a task description ("query for retrieval," "passage for retrieval," "classification") along with the text. At inference, the same prompt steers the embedding.
- Benefits:
  - One model for many tasks (retrieval, classification, clustering, code search).
  - Better quality per task by tuning the prompt.
  - Asymmetric tasks (query vs. doc) get tailored vectors.
- **MTEB** (Muennighoff et al. 2022): the standard benchmark; instruction-aware models top the leaderboard.
- **2026 picks**:
  - General retrieval: text-embedding-3-large (OpenAI), embed-v3 (Cohere), BGE-en-icl, E5-mistral-7b-instruct.
  - Code retrieval: voyage-code-2, CodeRankEmbed.
  - Multilingual: BGE-M3, Cohere embed-v3.

**Common follow-ups.**
- "Why does the same model handle both query and doc?" → Instruction differentiates them at embed time, so the vector geometry is task-aware. Saves an extra model.
- "How does this affect chunking?" → Doesn't directly — but the instruction-aware model expects a passage-shaped input; ensure chunks have enough context to be meaningful.

**Common mistakes.**
- Forgetting to use the instruction prefix at inference (often required for the model to produce calibrated embeddings).
- Mixing models — query embedded by one, doc by another — embedding spaces aren't aligned.

**References.**
- [Wang et al. — "Improving Text Embeddings with Large Language Models" (E5-Mistral)](https://arxiv.org/abs/2401.00368) — instruction-tuned embeddings.
- [Muennighoff et al. — "MTEB: Massive Text Embedding Benchmark"](https://arxiv.org/abs/2210.07316) — the benchmark.

---

### Q: What's the difference between fine-tuning and RAG? When would you choose each?

**Category:** concept
**Difficulty:** mid
**Tags:** [rag-vs-finetuning, knowledge-injection, deployment]

**Short answer.** **RAG** injects knowledge at *inference time* by retrieving and conditioning. **Fine-tuning** bakes knowledge (and behaviors) into the model's weights. RAG wins for **knowledge that changes** (docs, policies, products) and **factual recall** (the model is grounded by sources). Fine-tuning wins for **behaviors/style** (tone, format, refusals), **schemas** (consistent JSON output), and **fast inference** (no retrieval round-trip). Most production systems use both: fine-tune for style/format, RAG for knowledge.

**Expansion / why this is the answer.**
- **RAG strengths**:
  - Up-to-date knowledge without retraining.
  - Auditable: cite sources.
  - Easy to update: change the doc, re-embed.
  - Handles per-tenant data (RAG-against-customer-docs).
- **RAG limits**:
  - Latency of retrieval + context-stuffing.
  - Can't shape model behavior — only what it sees.
  - Garbage-in-garbage-out if retrieval is bad.
- **Fine-tuning strengths**:
  - Behavior shaping (tone, refusal, format consistency).
  - Schema compliance.
  - Latency: no retrieval call.
  - Specialized capabilities (code, medical reasoning).
- **Fine-tuning limits**:
  - Knowledge gets stale; expensive to update.
  - Can't easily handle per-tenant knowledge.
  - Catastrophic forgetting risk.
- **Empirical pattern (2024–2026)**: most production LLM apps are **RAG-on-pretrained**; advanced apps add a small fine-tune for style; only specialized domains (code, medical, legal) get heavyweight fine-tunes.
- **Hybrid**: fine-tune the *retrieval* model on domain queries (E5-like SFT); RAG against curated docs for knowledge.

**Common follow-ups.**
- "Why does RAG fail to teach the model 'how' to do something?" → It can give examples in-context but doesn't change weights; for true behavior shifts (e.g. always-emit-this-schema), fine-tune.
- "What about continued pretraining?" → Lets you bake domain knowledge in; orthogonal to RAG; expensive.

**Common mistakes.**
- Either-or framing — they're complements.
- Picking fine-tuning to add knowledge that will change next month.

**References.**
- [Ovadia et al. — "Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs"](https://arxiv.org/abs/2312.05934) — empirical comparison.
- [Soudani et al. — "Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge"](https://arxiv.org/abs/2403.01432) — when each wins.

---

### Q: What is multi-vector retrieval (ColBERT / parent-child indexing)? When is it worth the complexity?

**Category:** concept
**Difficulty:** senior
**Tags:** [multi-vector, colbert, parent-child, late-interaction]

**Short answer.** Multi-vector retrieval represents each document as multiple vectors instead of one — either per-token (ColBERT-style late interaction) or per-chunk in a hierarchy (parent-child / small-to-big). It improves retrieval precision because a single mean-pooled vector loses fine-grained signal. Cost: index size (multi× a single-vector index) and query complexity. Worth it when (a) retrieval precision at low-k is the binding constraint and (b) index storage is not.

**Expansion / why this is the answer.**
- **ColBERT-style** (Khattab & Zaharia 2020; ColBERTv2: Santhanam et al. 2022):
  - Per-token embeddings for both query and document.
  - Similarity = `Σ_q max_d (q_i · d_j)` (MaxSim).
  - Captures token-level "this query token matches this doc token" interactions that single-vector dense misses.
  - Production-fast via PLAID indexing.
- **Parent-child / small-to-big**:
  - Index small chunks for retrieval *precision*.
  - On hit, return the *parent chunk* (or summary, or full document) for context.
  - Implemented in LlamaIndex, common pattern.
- **Other multi-vector**:
  - **Multi-vector per document with different perspectives**: one vector per section, one summary vector, etc.
  - **Token-level dense + a sparse pass**: hybrid + late-interaction.
- **When worth it**:
  - High-precision low-k retrieval matters (e.g. legal lookup, code search).
  - You can afford the index size.
- **When not worth it**:
  - Tight latency / storage budget.
  - The single-vector quality is "good enough."

**Common follow-ups.**
- "What's PLAID?" → ColBERT's production indexing scheme — clusters + compressed token vectors; fast at large scale.
- "How does this interact with reranking?" → Multi-vector dense retrieval at scale can match a (dual-encoder + cross-encoder) pipeline's quality at lower latency.

**Common mistakes.**
- Implementing ColBERT-style without PLAID; naive multi-vector is O(N·M) — intractable.
- Confusing multi-vector retrieval with multi-query rewriting (different mechanisms).

**References.**
- [Khattab & Zaharia — "ColBERT"](https://arxiv.org/abs/2004.12832) — ColBERT.
- [Santhanam et al. — "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction"](https://arxiv.org/abs/2112.01488) — ColBERTv2.
- [LlamaIndex — Parent Document Retriever docs](https://docs.llamaindex.ai/en/stable/examples/retrievers/) — small-to-big pattern.

---

### Q: How would you decide between "use a long-context model" vs. "use RAG" for a 200-page document?

**Category:** concept
**Difficulty:** senior
**Tags:** [long-context, rag, tradeoffs]

**Short answer.** Long-context model: stuff the whole document into the prompt; model attends over all of it. RAG: chunk and retrieve. Long-context wins on **multi-document synthesis** and **comprehensive reasoning** when latency/cost is acceptable. RAG wins on **scale** (one query against millions of documents), **freshness** (update docs without re-running), **auditability** (cite sources), and **cost** (only retrieve what you need). For a 200-page document, single-query lookup-style tasks → RAG; comprehensive Q&A or summarization → long-context.

**Expansion / why this is the answer.**
- **200-page document** ≈ 100k–150k tokens. Most frontier models (Claude, Gemini 1.5+, GPT-4-turbo) handle this directly; many open models do not.
- **Long-context tradeoffs**:
  - Quality: better at multi-hop reasoning across the doc.
  - Cost: linear in tokens — every query reprocesses the whole doc.
  - Latency: significant TTFT; **prompt caching** can amortize across many queries on the same doc.
  - Lost-in-the-middle is still real even at high model capacity.
- **RAG tradeoffs**:
  - Quality: depends on retrieval; can miss the relevant chunk.
  - Cost: tiny per query.
  - Latency: low.
  - Freshness: easy.
- **Hybrid pattern**:
  - **RAG + larger context window**: retrieve more chunks (50+) into a long-context model. Some labs report this beats pure RAG.
  - **Cache the document, vary the query**: prompt-caching makes long-context economical for multi-query workflows.
  - **Summary-augmented RAG (RAPTOR)**: precompute hierarchical summaries; retrieve at the right level.
- **A decision frame**:
  - Does the user typically ask one focused question? RAG.
  - Does the user need a full-doc synthesis ("write a summary," "compare these clauses")? Long-context.
  - Latency-sensitive? RAG.
  - Document changes daily? RAG.

**Common follow-ups.**
- "What's the prompt-cache play here?" → If users query the same doc many times in a session, cache the doc once and pay decode-only on each query. Massive savings.
- "When does retrieval miss matter most?" → Multi-hop questions ("compare X across these two sections") that need passages from multiple chunks.

**Common mistakes.**
- Treating "model has 1M context" as "throw everything in." Cost and lost-in-the-middle still bite.
- Skipping the simple-question case where RAG is clearly cheaper.

**References.**
- [Anthropic — Long Context docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips) — practical guidance.
- [Liu et al. — "Lost in the Middle"](https://arxiv.org/abs/2307.03172) — still applicable at 1M context.

---

### Q: What are the common RAG failure modes, and how do you debug them?

**Category:** concept
**Difficulty:** senior
**Tags:** [rag-debugging, failure-modes, evaluation]

**Short answer.** Top failure modes: **(a) retrieval miss** (gold passage not in top-k); **(b) retrieval distraction** (irrelevant passage at top, model anchors); **(c) lost-in-the-middle** (correct passage retrieved but model ignores it); **(d) multi-hop** (answer requires combining passages, but no single retrieval has both); **(e) contradictory contexts** (retrievals disagree, model picks wrong); **(f) over-grounding** ("the docs don't say" when the model knew the answer); **(g) hallucination despite grounding**. Debug by separating retrieval vs. generation failure: run retrieval-only and check recall, then feed the *gold* passage to the generator and check if it answers correctly.

**Expansion / why this is the answer.**
- **Debug workflow**:
  1. Pick a failing query.
  2. **Retrieval audit**: were the relevant chunks retrieved? If no, your retrieval is the failure.
  3. **Oracle test**: hand-pick the right chunks; feed to the model. Does it answer correctly? If no, your generation/prompt is the failure.
  4. **Combined**: if both pass individually, you have a context-assembly or ordering problem (lost-in-the-middle).
- **By failure mode**:
  - **Retrieval miss**: improve embeddings, add BM25 / hybrid, better chunking, better query rewrite.
  - **Retrieval distraction**: stronger reranking; tighten top-k.
  - **Lost-in-the-middle**: order by reranker score; shrink context; place key passage early or last.
  - **Multi-hop**: query decomposition, multi-step retrieval, agent-style retrieve-think loops.
  - **Contradictory contexts**: confidence-aware merging; ask the model to flag disagreement.
  - **Over-grounding**: tune prompt — "answer using the documents, and if they don't cover it, say so explicitly"; sometimes too strict, model refuses easy questions.
  - **Hallucination despite grounding**: train/prompt for explicit citation, attribute every claim; penalize unsourced claims.
- **Production metrics**: track these separately, not in a single number. A monotone drop in recall@k tells you something different from a drop in faithfulness.

**Common follow-ups.**
- "How do you handle multi-hop in production?" → Either an agent loop (retrieve, reason, retrieve more) or a fixed query-decomposition step that issues multiple sub-retrievals.
- "How do you stop the model from over-grounding ('the docs don't say' for known-knowledge questions)?" → Allow the model to fall back to its parametric knowledge for general questions; restrict only domain-sensitive answers.

**Common mistakes.**
- Conflating retrieval and generation failure into one bug fix.
- Reading the model's output as the source of truth about its reasoning ("it said it couldn't find it" doesn't always mean retrieval failed).

**References.**
- [Barnett et al. — "Seven Failure Points When Engineering a Retrieval Augmented Generation System"](https://arxiv.org/abs/2401.05856) — failure taxonomy.
- [Liu et al. — "Lost in the Middle"](https://arxiv.org/abs/2307.03172) — ordering failure.

---

### Q: What's GraphRAG / knowledge-graph-augmented retrieval? When is it worth it?

**Category:** concept
**Difficulty:** senior
**Tags:** [graphrag, knowledge-graph, multi-hop]

**Short answer.** GraphRAG (Microsoft; Edge et al. 2024) augments standard chunk-based RAG with a knowledge graph extracted from the corpus: entities, relationships, and community summaries. For queries that require multi-hop reasoning or whole-corpus summarization ("what are the major themes across these reports?"), the graph structure gives better answers than chunk retrieval alone. Cost: significant ingest time and complexity. Worth it when you have a fixed, important corpus and care about whole-corpus reasoning.

**Expansion / why this is the answer.**
- **GraphRAG pipeline**:
  - Extract entities and relationships from each chunk using an LLM.
  - Build a graph: nodes = entities, edges = relationships, with mentions.
  - Detect communities (Leiden / Louvain).
  - For each community, summarize using an LLM.
  - **Local query**: retrieve entities + their neighborhoods + chunks.
  - **Global query**: route to community summaries; map-reduce over summaries.
- Where it wins: "what are the themes across the corpus?" — a question that no single chunk answers but the graph + community summaries do.
- Where it doesn't help much: focused lookup queries (vanilla RAG is fine).
- Tradeoffs:
  - Ingest cost: many LLM calls per chunk.
  - Maintenance: re-build the graph when docs change.
  - Complexity: more moving parts.
- **Other graph-RAG patterns**:
  - **LLM + KG hybrid**: ground the LLM with a curated knowledge graph (e.g. medical ontology); useful when the KG is authoritative.
  - **Vector + entity-link**: retrieve chunks, but also follow entity links between chunks for structure.

**Common follow-ups.**
- "Why not just use a larger context window?" → For local queries, yes. For "summarize the whole corpus," even a 1M-token context loses signal; community-summary indices win.
- "What's the cheapest way to get GraphRAG-like benefits?" → A simpler approach: precompute *summary indices* (per-doc, per-section) and route global queries to summary search.

**Common mistakes.**
- Treating GraphRAG as a drop-in upgrade. It's a different architecture with its own failure modes.
- Building the graph then never using it for the queries it helps with.

**References.**
- [Edge et al. — "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"](https://arxiv.org/abs/2404.16130) — GraphRAG.
- [Microsoft GraphRAG project](https://github.com/microsoft/graphrag) — implementation.

---

### Q: How do you handle citations and attribution in RAG?

**Category:** concept
**Difficulty:** mid
**Tags:** [citations, attribution, grounding]

**Short answer.** Two parts: (1) ensure the LLM *cites* its sources by prompt design — pass passages with explicit IDs (`[doc-12]`, `[chunk-3]`) and instruct it to attach IDs to factual claims; (2) ensure citations are *correct* — post-process by checking each cited claim against its passage (entailment / NLI / LLM judge). Anthropic's citations API and OpenAI's response-with-sources both implement variants of this; the trustworthy pattern is to *verify* citations server-side rather than trust the model's output.

**Expansion / why this is the answer.**
- **Why citations matter**:
  - User trust: users can verify claims.
  - Auditability: enterprise / legal / medical use cases require this.
  - Hallucination check: if every claim has a source, the system has a forcing function to ground answers.
- **Mechanisms**:
  - **Inline citations**: `[doc-12]` after each claim. Prompt: "After every factual claim, attach the source ID in brackets."
  - **Quote-extraction**: model returns a list of `(claim, supporting quote, source ID)`. Easier to verify.
  - **Citation-aware decoding**: server-side, the system tracks which source's KV cache was attended to most (research-grade; not in production widely).
- **Verification**:
  - For each cited claim, run an NLI / entailment check between the claim and the cited passage. Drop or down-weight unsupported claims.
  - LLM-as-judge for nuanced claims.
- **Anthropic Citations API**: Claude can return citations that link directly to passages in the supplied documents; the API validates that citations correspond to real passage spans.
- **Failure modes**:
  - Model cites a passage that doesn't support the claim ("citation hallucination").
  - Over-citing — every sentence has 5 citations, none of them useful.
  - Mid-sentence citation that breaks readability.

**Common follow-ups.**
- "What's the difference between groundedness and citation correctness?" → Groundedness: every claim is supported by *some* retrieved doc. Citation correctness: the *cited* doc actually supports the claim. You want both.
- "How do you measure citation correctness at scale?" → Sample-and-check; or full-coverage with a per-claim NLI classifier.

**Common mistakes.**
- Trusting that "the model cited it" means "the citation is valid."
- Citing the whole passage rather than the relevant span — fine for low-stakes, not for legal/medical.

**References.**
- [Bohnet et al. — "Attributed Question Answering: Evaluation and Modeling"](https://arxiv.org/abs/2212.08037) — attribution evaluation.
- [Anthropic — Citations API docs](https://docs.anthropic.com/en/docs/build-with-claude/citations) — primary docs.
- [Gao et al. — "RARR: Researching and Revising What Language Models Say, Using Language Models"](https://arxiv.org/abs/2210.08726) — verification.

---
