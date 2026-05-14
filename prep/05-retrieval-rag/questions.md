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
- [LlamaIndex — Retrievers documentation](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/) — small-to-big and other retrieval patterns.

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

### Q: What is agentic RAG, and when do you reach for it instead of vanilla RAG?

**Category:** concept
**Difficulty:** senior
**Tags:** [agentic-rag, multi-hop, react]

**Short answer.** Agentic RAG embeds retrieval inside an agent loop: the model issues a search, reads the results, decides whether to search again (with a refined query), and iterates until it has enough context. Useful for multi-hop questions ("Who founded the company that acquired X in 2019?"), under-specified queries, and exploratory research. Costs more (multiple LLM + retrieval rounds) and is harder to debug; vanilla RAG remains the default for single-hop factual lookup.

**Expansion / why this is the answer.**
- **Vanilla RAG**: one retrieval pass, one generation. Fixed pipeline.
- **Agentic RAG**: a loop —
  1. Model emits a search query (or decides not to search).
  2. Retriever returns passages.
  3. Model decides: "do I have what I need?" If yes, answer. If no, refine query and repeat.
- **Variants**:
  - **Self-RAG** (Asai et al. 2023): trained to emit special tokens controlling retrieval, supportiveness, and utility — fine-grained inline decisions.
  - **Corrective RAG / CRAG** (Yan et al. 2024): use a retriever-quality classifier; if retrieved docs are bad, fall back to web search or generation-from-scratch.
  - **ReAct-RAG**: ReAct loop with `search(query)` as a tool.
  - **Deep research-style** (OpenAI Deep Research, Anthropic Research, Perplexity Pro): heavy agentic; minutes of compute per query.
- **When to use agentic RAG**:
  - **Multi-hop**: chained sub-questions.
  - **Exploratory**: open-ended research where the user hasn't pinpointed a single fact.
  - **Long-tail / niche**: first retrieval often misses; need refinement.
- **When vanilla RAG wins**:
  - Single-hop factual queries.
  - Latency-sensitive.
  - Cost-sensitive (agentic round-trips multiply LLM calls).
- **Failure modes**:
  - Agent loops without making progress (no termination).
  - Compounding errors across hops.
  - Doubled hallucination risk (each hop is a chance to invent).

**Common follow-ups.**
- "How do you cap cost in agentic RAG?" → Hard step limit; budget on tool calls; fallback to vanilla on long runs.
- "What's the right base model?" → Strong tool-use + retrieval-aware behavior. Anthropic / OpenAI frontier; some OSS (Llama 3.1+ instruct) are competent.

**Common mistakes.**
- Treating agentic RAG as universally better — it's expensive and often unnecessary.
- No termination criterion → runaway loops.

**References.**
- [Asai et al. — "Self-RAG"](https://arxiv.org/abs/2310.11511) — Self-RAG.
- [Yan et al. — "Corrective Retrieval Augmented Generation"](https://arxiv.org/abs/2401.15884) — CRAG.
- [Anthropic — "Building effective agents"](https://www.anthropic.com/research/building-effective-agents) — agent + retrieval pattern.

---

### Q: How would you build a multi-lingual retrieval system?

**Category:** concept
**Difficulty:** senior
**Tags:** [multilingual, cross-lingual, retrieval, bge-m3]

**Short answer.** Use a multilingual embedding model (BGE-M3, Cohere multilingual-embed-v3, OpenAI text-embedding-3) that maps text from many languages into a shared embedding space — same-meaning queries in different languages retrieve the same documents. Pair with language-aware BM25 (per-language analyzers, tokenizers). For low-resource languages, consider machine-translating the query into a high-resource language before retrieval. Evaluate per-language; aggregate retrieval quality varies dramatically across languages.

**Expansion / why this is the answer.**
- **Multilingual embedding models** (top picks 2024–2026):
  - **BGE-M3** (Chen et al. 2024): handles 100+ languages; dense + sparse + multi-vector retrieval in one model.
  - **Cohere embed-v3 multilingual**: production-grade.
  - **OpenAI text-embedding-3**: multilingual, robust.
  - **mE5** (Wang et al. 2024): multilingual variant of E5.
- **Cross-lingual retrieval**: query in language A, retrieve docs in language B; multilingual embeddings make this work natively.
- **Sparse complement**:
  - Per-language BM25 with language-specific tokenizers / stemmers / stopwords.
  - For CJK / Thai / Burmese, character-or-syllable-level tokenization beats whitespace.
- **Hybrid + RRF**: especially valuable for multilingual where dense retrieval can struggle with named entities in low-resource languages.
- **Query expansion** for low-resource:
  - Machine-translate query into the doc's language (or vice versa).
  - Use multilingual LLM to paraphrase.
- **Evaluation gotcha**: per-language recall@k can vary by 30+ percentage points. Aggregate scores hide low-resource failures. Eval per-language.
- **Modern reranker**: Cohere Rerank-3 supports 100+ languages; BGE reranker M3 too.

**Common follow-ups.**
- "What's BGE-M3's three-mode retrieval?" → Dense, sparse, and multi-vector (ColBERT-style) in one model; combine via score fusion.
- "Does this work for code search?" → Use a code-specialized model (Voyage code-2, CodeRankEmbed); multilingual code embedding is a niche.

**Common mistakes.**
- Using an English-only embedding model and assuming it "works for everything."
- No per-language evaluation.

**References.**
- [Chen et al. — "BGE-M3"](https://arxiv.org/abs/2402.03216).
- [Wang et al. — "mE5"](https://arxiv.org/abs/2402.05672).

---

### Q: How would you design a code search system?

**Category:** system-design
**Difficulty:** senior
**Tags:** [code-search, retrieval, embeddings]

**Short answer.** Code search needs three complementary indexes: **(1) semantic** (code-trained embedding model retrieves "find me a function that does X"), **(2) lexical / symbol** (exact-name lookup for `getUserById`, AST-grep, tree-sitter), and **(3) graph** (call graph, import graph for "show me callers of `foo`"). Fuse with RRF or learned ranker. Code-specialized embeddings (Voyage code-2, CodeRankEmbed) substantially outperform general-text models on code corpora.

**Expansion / why this is the answer.**
- **Why code is different from text**:
  - Identifiers carry strong signal; exact-name lookup is high-precision.
  - Code structure (AST, type system) is rich; ignoring it loses signal.
  - Comments + code together: dual representations help.
- **Components**:
  1. **Semantic embedding index**:
     - Code-trained model (Voyage code-2, Cohere embed-v3-code, CodeRankEmbed, OpenAI text-embedding-3).
     - Chunk by function / class boundary, not arbitrary character windows.
     - Index per language; query routed by detected language.
  2. **Lexical / symbol index**:
     - Tree-sitter parse → AST.
     - Symbol-graph index (functions, classes, types).
     - `grep`/regex over identifiers.
     - LSP-style "find references."
  3. **Graph index**:
     - Call graph, import graph, type-hierarchy graph.
     - Enables "show me the chain of callers."
- **Fusion**:
  - Per-query, run all three.
  - Rerank with a cross-encoder or LLM judge.
- **Production systems**: Sourcegraph (Cody), GitHub Copilot Workspace, Cursor — each combines variants.
- **Eval**:
  - Per-query type (function-by-description, find-callers, search-by-symbol).
  - CodeSearchNet (Husain et al. 2019) for benchmarks; modern systems use private eval sets too.

**Common follow-ups.**
- "How would you handle very large monorepos?" → Sharded indexes; lazy embedding on file change; incremental index updates.
- "How does this integrate with a coding assistant?" → The coding assistant queries this system; results inform completions or chat answers.

**Common mistakes.**
- Pure-semantic retrieval without symbol lookup — fails on exact-name queries.
- Treating code chunks as arbitrary text chunks (misses structural boundaries).

**References.**
- [Husain et al. — "CodeSearchNet"](https://arxiv.org/abs/1909.09436).
- [Sourcegraph Cody docs](https://sourcegraph.com/docs/cody) — production case study.
- [Voyage AI — embedding model list](https://docs.voyageai.com/docs/embeddings) — code-2 embeddings.

---

### Q: Compare embedding-as-a-service providers (OpenAI / Cohere / Voyage / open-source).

**Category:** concept
**Difficulty:** mid
**Tags:** [embeddings, providers, mteb]

**Short answer.** **OpenAI text-embedding-3** (large/small): strong general purpose, broad-language; ubiquitous. **Cohere embed-v3**: instruction-aware, strong multilingual, dedicated reranker. **Voyage AI**: domain-specialized (voyage-code-2 for code, voyage-finance-2 for finance). **OSS** (BGE family, E5-Mistral, GTE, Stella): top of MTEB; free; require self-hosting. Pick on (a) MTEB / domain benchmark performance, (b) cost per million tokens, (c) self-host vs. managed, (d) embedding dimension (memory cost downstream).

**Expansion / why this is the answer.**
- **Closed APIs**:
  - **OpenAI**: text-embedding-3-large (3072 dim) / -small (1536 dim); great general purpose. Pay-per-token. Multilingual.
  - **Cohere**: embed-v3 (English / multilingual / light); instruction-aware (separate `input_type` for query vs. document). Dedicated Cohere Rerank-3.
  - **Voyage**: domain-specialized; voyage-large-2-instruct, voyage-code-2, voyage-finance-2; competitive on niche.
  - **Google**: text-embedding-gecko on Vertex.
- **Open source**:
  - **BGE family** (BAAI): bge-large-en-v1.5, bge-m3 (multilingual + multi-mode).
  - **E5 / E5-Mistral** (Microsoft): instruction-tuned; strong on MTEB.
  - **GTE** (Alibaba): gte-large; competitive.
  - **Stella** (Dunzhang): top of MTEB leaderboard in mid-2024.
  - **Nomic** (nomic-embed-text): long-context (8192).
- **Comparison axes**:
  - **MTEB score**: per-task; pick by your downstream task type (retrieval, classification, clustering).
  - **Embedding dim**: 384–4096; smaller = cheaper storage / faster ANN. OpenAI text-embedding-3 supports "shortening" — truncate to desired dim with minimal quality loss (Matryoshka representation).
  - **Cost**: closed APIs ~$0.02–0.13 per million tokens. OSS = your serving cost.
  - **Self-host**: OSS gives full control + no per-call cost; cost amortizes over scale.
- **Decision rubric**:
  - Bootstrapping fast: OpenAI or Cohere.
  - Domain-specific (code, finance, biomedical): Voyage or domain-fine-tuned OSS.
  - Sustained high-volume: self-host OSS with FP8/INT8 quantization.

**Common follow-ups.**
- "What's Matryoshka representation learning?" → Train embeddings such that the first `k` dimensions of a `d`-dim embedding are themselves usable embeddings (with quality degrading gracefully) — used in OpenAI text-embedding-3 and Nomic.
- "What's MTEB?" → Massive Text Embedding Benchmark; the standard for ranking embedding models. Check current leaderboard rather than trusting paper-time numbers.

**Common mistakes.**
- Picking by MTEB score alone — your domain may differ; benchmark on a slice of your data.
- Forgetting embedding-dim is a storage / serving cost downstream.

**References.**
- [Muennighoff et al. — "MTEB"](https://arxiv.org/abs/2210.07316).
- [Kusupati et al. — "Matryoshka Representation Learning"](https://arxiv.org/abs/2205.13147).
- [OpenAI text-embedding-3 announcement](https://openai.com/index/new-embedding-models-and-api-updates/).

---

### Q: What is Self-RAG / Corrective RAG, and what problem do they solve?

**Category:** concept
**Difficulty:** senior
**Tags:** [self-rag, crag, adaptive-rag]

**Short answer.** Vanilla RAG retrieves once and trusts the result. **Self-RAG** (Asai et al. 2023) trains the model to decide *when* to retrieve, *whether retrieved passages are relevant*, and *whether its own outputs are supported* — emitting special control tokens for each. **Corrective RAG (CRAG)** (Yan et al. 2024) adds a separate retriever-quality classifier; if retrievals are bad, falls back to web search or generation-without-RAG. Both address the "retrieval is sometimes worse than nothing" problem.

**Expansion / why this is the answer.**
- **Failure mode they target**:
  - Retrieval miss: retrieved passages are irrelevant; the model is misled.
  - Over-retrieval: model retrieves when it already knows the answer; wasted latency, sometimes adds noise.
  - Hallucination despite retrieval: model ignores the passages, makes things up.
- **Self-RAG** (training-time approach):
  - Model emits `[Retrieve]` / `[No Retrieve]` decisions.
  - For each retrieved doc, emits `[Relevant]` / `[Irrelevant]`.
  - For each generated statement, emits `[Supported]` / `[Partially Supported]` / `[Unsupported]`.
  - Trained on a critic-labeled dataset.
  - Inference: parses these tokens; can skip irrelevant docs, abstain when unsupported.
- **CRAG** (inference-time approach):
  - Separately-trained retriever-quality classifier scores each retrieved doc.
  - High → use docs.
  - Low → fall back to web search.
  - Ambiguous → augment with web search.
- **Adaptive RAG** (Jeong et al. 2024): a query classifier predicts "no RAG needed" / "single-hop" / "multi-hop" and routes.
- **Where these win**:
  - Diverse query distribution where one-size-fits-all retrieval fails.
  - Long-tail of queries where retrieval quality varies.
- **Where they lose**:
  - Latency cost (extra classifiers / decisions).
  - Training complexity.

**Common follow-ups.**
- "Is this strictly better than vanilla RAG?" → Not always; adds cost and complexity. Worth it when the retrieval-quality variance is high.
- "What's the relationship to agentic RAG?" → Self-RAG/CRAG are *inline* decision frameworks (not multi-hop loops). Agentic RAG is multi-turn.

**Common mistakes.**
- Treating these as replacements for retrieval-quality fixes (better chunking, better embeddings) — they're *complementary*.

**References.**
- [Asai et al. — "Self-RAG"](https://arxiv.org/abs/2310.11511).
- [Yan et al. — "Corrective Retrieval Augmented Generation"](https://arxiv.org/abs/2401.15884).
- [Jeong et al. — "Adaptive-RAG"](https://arxiv.org/abs/2403.14403).

---

### Q: What's the difference between sparse, dense, and multi-vector retrieval — concisely?

**Category:** concept
**Difficulty:** intro
**Tags:** [retrieval, sparse, dense, multi-vector]

**Short answer.** **Sparse** (BM25, SPLADE): represent each doc as a sparse vector over the vocabulary; matches on terms / lexical signal. **Dense** (SBERT, BGE, OpenAI embeddings): single fixed-dim vector per doc; matches on semantic similarity. **Multi-vector** (ColBERT, ColBERTv2): per-token vectors per doc; "late interaction" via MaxSim; captures fine-grained alignment. In production: hybrid sparse + dense by default; multi-vector when precision at low-k matters and storage budget allows.

**Expansion / why this is the answer.**
- **Sparse**:
  - BM25 (classical TF-IDF variant): sparse term-frequency vector + IDF weighting.
  - SPLADE (learned sparse): BERT produces sparse representations; competitive with dense.
  - Strength: exact-token signal, rare-word matching, named entities, identifiers.
- **Dense**:
  - Dual-encoder: fixed `d`-dim vector per doc.
  - Strength: semantic similarity, paraphrase, cross-lingual (with multilingual model).
  - Weakness: mean-pooling loses fine-grained token-level signal.
- **Multi-vector**:
  - Token-level embeddings (per token); similarity via `Σ_q max_d (q_i · d_j)` (MaxSim).
  - Strength: best of both worlds — semantic + token-precision.
  - Cost: index size ~10–30× a dense index.
- **Decision rubric**:
  - General-purpose retrieval: hybrid sparse + dense.
  - Precision at low-k matters, storage abundant: multi-vector (ColBERTv2 + PLAID).
  - Resource-constrained: dense alone, with strong embeddings.

**Common follow-ups.**
- "Why is dense sometimes worse than BM25 on a benchmark?" → Named entities, exact strings, very-out-of-domain queries.
- "What's PLAID?" → ColBERTv2's production indexing — clusters + compressed token vectors; fast and small enough for production.

**Common mistakes.**
- Calling SPLADE "dense" — it's learned but sparse.
- Conflating multi-vector with multi-query.

**References.**
- [Khattab & Zaharia — "ColBERT"](https://arxiv.org/abs/2004.12832).
- [Formal et al. — "SPLADE"](https://arxiv.org/abs/2107.05720).
- [Robertson & Zaragoza — BM25 reference](https://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf).

---

### Q: How do you handle very large documents (>100k tokens) in RAG?

**Category:** concept
**Difficulty:** mid
**Tags:** [large-documents, chunking, summary-index]

**Short answer.** Three strategies: (1) **chunk-and-retrieve** standard RAG with smaller chunks; (2) **hierarchical summary index**: per-section / per-chapter summaries indexed alongside detail chunks; route by query specificity. (3) **long-context model**: stuff the whole doc in the prompt (works only with 200k+ context models and is expensive). The right answer depends on query shape — focused lookup → RAG; cross-document synthesis → long context.

**Expansion / why this is the answer.**
- See T5 base "long-context-vs-RAG" entry for the broader frame.
- **Summary indexes** (RAPTOR-style):
  - Per-paragraph chunks at the leaves.
  - Per-section summaries one level up.
  - Per-chapter summaries above that.
  - Per-document summary at the top.
  - Query → which level to retrieve from.

**Common follow-ups.**
- "What's the tradeoff with hierarchical summary?" → Ingest cost (lots of LLM calls); query routing complexity.

**Common mistakes.**
- One chunk size for all documents.

**References.**
- [Sarthi et al. — "RAPTOR"](https://arxiv.org/abs/2401.18059).

---

### Q: What is the role of chunk-level metadata in RAG?

**Category:** concept
**Difficulty:** mid
**Tags:** [metadata, filtering, rag]

**Short answer.** Metadata (source URL, document type, publication date, ACL group, language, custom tags) attached to each chunk enables: (a) **pre-filter** retrieval (only show docs the user is authorized to see); (b) **time-window filtering** (recent only); (c) **citation generation** (link the answer to the source); (d) **hybrid retrieval scoring** (boost recent / authoritative). Critical in production RAG; vanilla "embed and search" misses it.

**Expansion / why this is the answer.**
- Metadata fields typically tracked:
  - `doc_id`, `chunk_id`.
  - `source_url`, `title`.
  - `created_at`, `updated_at`.
  - `acl_groups`: who can see this.
  - `language`.
  - `domain` / `category`.
  - Custom: `tags`, `priority`, `confidence`.
- Vector indexes (Qdrant, Weaviate, pgvector) support metadata filtering at query time.
- ACL filtering is **critical** — wrong-access in RAG is a security incident.

**Common follow-ups.**
- "How do you handle per-tenant data?" → ACL filtering on every query.
- "Filtering performance?" → Pre-filter (compute ANN over the filtered subset) is more expensive than post-filter; tradeoffs.

**Common mistakes.**
- Storing no metadata; embedding the whole text and losing the per-chunk handles.

**References.**
- [Qdrant docs — filtering](https://qdrant.tech/documentation/concepts/filtering/).

---

### Q: How would you build a retrieval system for code (function-level)?

**Category:** concept
**Difficulty:** senior
**Tags:** [code-search, code-retrieval, function-level]

**Short answer.** Three sources: (1) **semantic search** via code-aware embeddings; (2) **symbol search** via tree-sitter or LSP — exact name lookups for `getUserById`; (3) **call-graph traversal** for related functions. Chunk at function/class boundary, not character count. Use a code-trained embedder (Voyage code-2, CodeRankEmbed) — general-text embedders perform poorly on code. Fuse with RRF.

**Expansion / why this is the answer.**
- See T5 base "code search system" entry for fuller treatment.
- Function-level chunking specifically:
  - Tree-sitter or LSP to identify function boundaries.
  - Each chunk: function signature + docstring + body.
  - Metadata: file path, language, lines.

**Common follow-ups.**
- "Why not chunk by line count?" → Splits functions; breaks symbol resolution.
- "Cross-language search?" → Multi-lingual code embedding model.

**Common mistakes.**
- Pure-text embedder on code — fails on exact-name and structure queries.

**References.**
- [Voyage code-2](https://docs.voyageai.com/docs/embeddings).

---

### Q: What is "MMR" (maximal marginal relevance) reranking, and when do you use it?

**Category:** concept
**Difficulty:** mid
**Tags:** [mmr, diversity, reranking]

**Short answer.** MMR (Carbonell & Goldstein 1998): a reranking criterion that balances *relevance* with *diversity* — pick the next result to maximize relevance to the query minus similarity to already-selected results. Useful when you want diverse top-K (different perspectives, multi-faceted query) rather than near-duplicates of the top-1.

**Expansion / why this is the answer.**
- Formula: `MMR(c) = λ · sim(c, query) − (1−λ) · max_{d ∈ S} sim(c, d)` where `S` is the already-selected set.
- `λ` ∈ [0, 1]: 1 = pure relevance; 0 = pure diversity.
- Typical use: top-10 with `λ = 0.7` — mostly relevance, some diversity.
- **When useful**:
  - Search results page: show multiple perspectives.
  - RAG: avoid 5 near-duplicate chunks from the same source.
- **When not**: pure single-answer Q&A where you want the most relevant chunk.

**Common follow-ups.**
- "How does MMR compare to clustering?" → MMR picks during retrieval; clustering picks after.
- "λ tuning?" → Held-out eval set.

**Common mistakes.**
- Skipping MMR-like deduplication on the same-source chunks.

**References.**
- [Carbonell & Goldstein — "MMR"](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf).

---

### Q: What is "Cohere Rerank" / cross-encoder rerank as a service?

**Category:** concept
**Difficulty:** intro
**Tags:** [reranking, cohere, voyage]

**Short answer.** A managed cross-encoder rerank API (Cohere Rerank, Voyage rerank, Jina): provide a query and a list of candidate documents; receive a relevance-sorted list with scores. Saves you the work of hosting your own cross-encoder. Trade-off: latency hit per call (50–200 ms typical); cost. Production RAG often uses these.

**Expansion / why this is the answer.**
- **API call shape**: `rerank(query, documents) → ranked_list_with_scores`.
- **Vendors**:
  - **Cohere Rerank-3**: multilingual, fast.
  - **Voyage rerank-2-lite**: general purpose.
  - **Jina Reranker**: open-source-friendly.
- **Self-host option**: BGE-reranker (open source, BAAI).
- **When to use API**:
  - Don't want to manage GPU serving for the reranker.
  - Modest scale.
- **When to self-host**:
  - Sustained high volume.
  - Privacy / on-prem.

**Common follow-ups.**
- "Cost-quality tradeoff?" → API: $0.001–0.01 per query. Self-hosted: amortized.
- "Why dedicated rerank?" → Quality wins over dual-encoder-only retrieval; necessary for production RAG.

**Common mistakes.**
- Skipping the rerank step; top-k dense is too noisy.

**References.**
- [Cohere Rerank docs](https://docs.cohere.com/docs/rerank-overview).

---

### Q: What's the difference between "passage retrieval" and "document retrieval"?

**Category:** concept
**Difficulty:** intro
**Tags:** [passage-retrieval, document-retrieval, granularity]

**Short answer.** **Document retrieval**: return whole documents; user navigates within. **Passage retrieval**: return the specific span (paragraph, section) that answers the query. Modern RAG is passage retrieval — chunks are passage-sized. Document retrieval is the classical IR setup (web search returning URLs).

**Expansion / why this is the answer.**
- **Document retrieval**: TF-IDF / BM25 on full-doc indices; web search.
- **Passage retrieval**: index each chunk; return ranked passages.
- **RAG specifically uses passage retrieval** because the LLM ingests the passages as context.
- **Hybrid pattern**: passage retrieval; on user request for more context, expand to the parent document.

**Common follow-ups.**
- "Why not always passages?" → Doc-level is appropriate for web-search use; passage is for question-answering.

**Common mistakes.**
- Conflating the two; they have different evaluation metrics.

**References.**
- [Manning, Raghavan, Schütze — *IR*](https://nlp.stanford.edu/IR-book/) — classical IR.

---

### Q: How does Cohere's Rerank-3 differ from a generic cross-encoder?

**Category:** concept
**Difficulty:** mid
**Tags:** [cohere-rerank, cross-encoder, multilingual]

**Short answer.** Cohere Rerank-3 is multilingual, instruction-aware, and trained on diverse retrieval pairs. Generic cross-encoders (e.g. `cross-encoder/ms-marco-MiniLM-L-12-v2`) are typically English-only and trained on a single dataset. Cohere Rerank-3 also returns calibrated relevance scores (not just rankings). For production multilingual RAG, the difference is meaningful.

**Expansion / why this is the answer.**
- **Generic cross-encoder**:
  - MS MARCO trained; English; specific style.
  - Lightweight; fast.
- **Cohere Rerank-3**:
  - 100+ languages.
  - Instruction-aware (knows it's reranking).
  - Calibrated scores.
- **BGE-reranker** (open-source): competitive on English; multilingual variants exist.

**Common follow-ups.**
- "When to use Rerank-3 vs BGE?" → Multilingual / instruction-aware → Cohere; English-only / on-prem → BGE.

**Common mistakes.**
- Reranking with an English-only model on multilingual content.

**References.**
- [Cohere Rerank documentation](https://docs.cohere.com/docs/rerank-overview).
- [BGE-reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3).

---

### Q: What is "dense passage retrieval" historically vs. modern embeddings?

**Category:** concept
**Difficulty:** mid
**Tags:** [dpr, dense-retrieval, history]

**Short answer.** DPR (Dense Passage Retrieval, Karpukhin et al. 2020) was the canonical dense-retrieval paper: dual-encoder BERT trained with contrastive loss on question-passage pairs (Natural Questions, TriviaQA). Modern embeddings (E5, BGE, OpenAI text-embedding-3, Cohere embed-v3) follow the same architecture but with better training data, larger model size, instruction-tuning, and multilingual capability. DPR set the template; modern is the descendant.

**Expansion / why this is the answer.**
- **DPR (2020)**:
  - BERT-base dual-encoder.
  - Trained on (question, positive passage) + in-batch negatives.
  - Outperformed BM25 on open-domain QA.
- **Modern improvements**:
  - Larger backbones (LLM-based; E5-Mistral).
  - Better data curation (MTEB-tuned mixtures).
  - Hard negatives mining.
  - Instruction prefix ("Query: ..." vs. "Document: ...").
  - Multilingual.
- **Quality gap**: DPR vs. modern is large; MTEB leaderboard shows the progress.

**Common follow-ups.**
- "Is DPR still used in production?" → Rarely; modern embeddings dominate.
- "Why the architecture didn't change much?" → Dual-encoder is fundamentally right; the gains came from data and scale.

**Common mistakes.**
- Citing DPR as "best-in-class" — it's foundational but superseded.

**References.**
- [Karpukhin et al. — "DPR"](https://arxiv.org/abs/2004.04906).
- [Wang et al. — "E5"](https://arxiv.org/abs/2212.03533).

---

### Q: What is "hard negative mining" for retrieval training?

**Category:** concept
**Difficulty:** senior
**Tags:** [hard-negatives, contrastive, training]

**Short answer.** Hard negatives: documents that are *similar to but not the gold answer* for a query. Training with hard negatives (not just random ones) teaches the retriever to make fine distinctions. Standard mining: use BM25 or a weak retriever to find candidates near the gold; pick the top-K non-gold as negatives. Modern recipe used by E5, BGE, etc.

**Expansion / why this is the answer.**
- The intuition: random negatives are too easy; the retriever needs to learn near-misses.
- **Mining process**:
  - For each `(query, positive)` pair:
    - Retrieve top-K candidates with a weaker retriever.
    - Filter out the positive.
    - Use the remaining top-K as hard negatives.
- **Multiple rounds**: as the retriever improves, re-mine; the new model finds new hard negatives.
- **Empirical**: hard negatives improve MTEB scores significantly over random negatives.
- **Caveats**: false negatives — sometimes the "hard negative" is actually a valid answer; quality control needed.

**Common follow-ups.**
- "What's a 'false negative' in this context?" → A doc labeled as negative that's actually relevant.
- "How many hard negatives per query?" → 7–63 typical; in-batch + mined.

**Common mistakes.**
- Random-only negatives in training.

**References.**
- [Xiong et al. — "Approximate Nearest Neighbor Negative Contrastive Learning" (ANCE)](https://arxiv.org/abs/2007.00808).

---

### Q: How does RAG handle "out of distribution" queries?

**Category:** concept
**Difficulty:** mid
**Tags:** [ood, refusal, rag]

**Short answer.** OOD queries (the corpus doesn't contain the answer) should result in: (a) retrieval returning low-relevance results; (b) the LLM detecting low support; (c) explicit refusal: "I couldn't find that in the documentation." Implementations: low-confidence threshold from the reranker; LLM-judge verification step; explicit "supported / not supported" output from the model.

**Expansion / why this is the answer.**
- **Failure mode**: model invents an answer (hallucination) when retrieval returns poor matches.
- **Detection**:
  - **Reranker score threshold**: if top-1 below threshold, OOD.
  - **LLM self-evaluation**: prompt the model to say whether the retrieved docs support an answer.
  - **Self-RAG-style** training: model emits explicit `[Unsupported]` tokens.
- **Action on OOD**:
  - Refuse: "I don't have information about that."
  - Escalate to human / web search.
- **User trust**: explicit refusal is better than confident-wrong.

**Common follow-ups.**
- "How do you tune the threshold?" → Held-out OOD set; measure refusal accuracy vs. over-refusal.
- "Why is over-refusal bad?" → User experience: model says "I don't know" when it actually could answer.

**Common mistakes.**
- No OOD detection; model confabulates.

**References.**
- [Asai et al. — "Self-RAG"](https://arxiv.org/abs/2310.11511).

---

### Q: What is "context distillation" / context compression for RAG?

**Category:** concept
**Difficulty:** mid
**Tags:** [context-compression, distillation, llmlingua]

**Short answer.** Compress retrieved context before passing to the LLM. Methods: (a) **LLMLingua** (Jiang et al. 2023) — use a small LLM to predict per-token importance; drop low-importance tokens; (b) **summarization** — small model summarizes each chunk; (c) **selective extraction** — only keep sentences directly relevant to the query. Trade-off: cost-vs-quality; can lose nuance.

**Expansion / why this is the answer.**
- **LLMLingua / LongLLMLingua**:
  - Small LLM scores tokens for "perplexity contribution."
  - Drop low-contribution tokens.
  - 2–10× compression at modest quality cost.
- **Summary-based**:
  - Each retrieved chunk summarized to N tokens.
  - Pass summaries to the main LLM.
- **Use cases**:
  - Very long contexts that don't fit.
  - Cost-sensitive workloads.
- **When to skip**:
  - Long-context models with prompt caching handle this well.

**Common follow-ups.**
- "Quality cost?" → 5–15% on standard RAG benchmarks; depends on compression ratio.

**Common mistakes.**
- Compressing when the model has plenty of context budget.

**References.**
- [Jiang et al. — "LLMLingua"](https://arxiv.org/abs/2310.05736).

---

### Q: What is "query routing" in RAG?

**Category:** concept
**Difficulty:** mid
**Tags:** [query-routing, multi-index]

**Short answer.** A classifier routes each query to the right index: e.g. "support questions" → product docs index; "math homework" → math textbook index; "code question" → code-search index. Cheap routing (rule-based or small classifier) lets you scale RAG across multiple specialized corpora without a single mega-index.

**Expansion / why this is the answer.**
- **Why route**:
  - Indices have different costs, freshness, and ACL needs.
  - Different embedding models for different domains (code vs. text).
  - Avoid contaminating results across domains.
- **Routing options**:
  - **Rule-based**: regex / keyword match (cheap; brittle).
  - **Classifier**: small ML model on labeled `(query, index)` pairs.
  - **LLM-based**: cheap LLM as router.
- **Multi-index parallel**: route to multiple indices and fuse (more expensive but handles ambiguous queries).

**Common follow-ups.**
- "When does multi-index fusion win?" → Ambiguous queries that span domains.
- "Router error mode?" → Mis-routing → wrong index → bad retrieval → bad answer.

**Common mistakes.**
- Static rules in a fast-evolving system; data-drift breaks them.

**References.**
- [Jeong et al. — "Adaptive-RAG"](https://arxiv.org/abs/2403.14403) — routing variant.

---

### Q: How would you eval retrieval quality without labeled data?

**Category:** concept
**Difficulty:** senior
**Tags:** [unsupervised-eval, synthetic-eval]

**Short answer.** Synthetic evaluation: for each known document, use an LLM to generate a question whose answer is in that document. Now you have a labeled `(query, gold_doc)` pair. Compute recall@k and MRR on the synthetic set. Noisy but cheap; useful for relative comparison of retrieval methods. Hand-validate a subset for calibration.

**Expansion / why this is the answer.**
- **The workflow**:
  - Sample N documents from your corpus.
  - For each: LLM generates a focused question whose answer is in that doc.
  - Now you have (synthetic question, gold doc).
  - Run retrieval; measure recall@k.
- **Quality**:
  - Synthetic queries don't match real-user queries perfectly.
  - But: useful for relative comparison ("does the new embedding beat the old?").
- **Calibration**:
  - Hand-validate 50–100 of the synthetic pairs.
  - Verify the LLM's generated questions are actually answered by the gold doc.

**Common follow-ups.**
- "Can you do this for production monitoring?" → Yes; continuous synthetic-eval signals retriever quality.
- "What's the alternative?" → Hand-labeled set (expensive); user-feedback signals (sparse).

**Common mistakes.**
- Trusting synthetic eval without calibration.

**References.**
- [Bonifacio et al. — "InPars"](https://arxiv.org/abs/2202.05144) — synthetic queries for retrieval training.

---

### Q: What is "fusion-in-decoder" (FiD)?

**Category:** concept
**Difficulty:** senior
**Tags:** [fid, encoder-decoder-rag, izacard]

**Short answer.** FiD (Izacard & Grave 2020): a RAG architecture for encoder-decoder models (T5). Each retrieved passage is encoded independently; their outputs are concatenated and attended over by the decoder. Decoupling passage encoding (parallel) from decoding (autoregressive) makes FiD scale better than putting all passages in one encoder.

**Expansion / why this is the answer.**
- **Standard concat-then-encode**: stuff all passages into the encoder; encode together; decode.
- **FiD**: encode each passage *independently*; decoder cross-attends to the concatenation of encoder outputs.
- **Advantages**:
  - Parallel encoding of passages.
  - Scales to many passages (10+).
  - Used in Atlas, RETRO variants.
- **For decoder-only LLMs**: FiD doesn't directly apply (no encoder); the equivalent is just putting passages in the prompt.

**Common follow-ups.**
- "Why not used much in 2026?" → Decoder-only LLMs dominate; FiD requires encoder-decoder.
- "Does FiD match modern RAG quality?" → On classical QA benchmarks, competitive; less so for modern long-context tasks.

**Common mistakes.**
- Citing FiD as the modern RAG default — it's an encoder-decoder pattern.

**References.**
- [Izacard & Grave — "FiD"](https://arxiv.org/abs/2007.01282).

---

### Q: What is "RAG fusion" / multi-query retrieval?

**Category:** concept
**Difficulty:** mid
**Tags:** [rag-fusion, multi-query]

**Short answer.** Multi-query RAG: from the user's query, generate `k` paraphrased queries with an LLM; retrieve top-N for each; merge with RRF; pass to the generator. Captures more semantic angles than a single query; helps on ambiguous or under-specified questions. Cost: an extra LLM call (cheap if small).

**Expansion / why this is the answer.**
- **The pipeline**:
  - User query → LLM generates 3–5 paraphrases / rewordings.
  - Retrieve top-N for each.
  - Union and RRF.
  - Rerank top-K.
- **Compared to HyDE**: HyDE generates a hypothetical *answer*; multi-query generates rewordings of the *question*.
- **When this wins**:
  - Vague / under-specified queries.
  - Domain language mismatch (user says "stomach pain"; docs say "abdominal discomfort").

**Common follow-ups.**
- "Combination with HyDE?" → Yes; do both; expensive but covers more.
- "Cost vs benefit?" → 1 extra LLM call (cheap with a small model); typically worth it.

**Common mistakes.**
- Treating multi-query as universally beneficial; on focused queries, it adds noise.

**References.**
- [Anthropic — Contextual Retrieval blog](https://www.anthropic.com/news/contextual-retrieval).

---

### Q: What is "contextual retrieval" / Anthropic's approach?

**Category:** concept
**Difficulty:** mid
**Tags:** [contextual-retrieval, anthropic]

**Short answer.** Anthropic's Contextual Retrieval (Sep 2024): before embedding each chunk, prepend a 50–100 token context describing where the chunk sits in its document ("This is from the 'Pricing' section of the Acme product manual..."). Then embed. Improves retrieval recall ~35% by giving the chunk-level vector enough context to be findable. Pair with hybrid (BM25 + dense) for additional gains.

**Expansion / why this is the answer.**
- **The problem**: small chunks lose context. "It supports up to 500 users." — what's "it"?
- **The fix**: prepend chunk-specific context generated by an LLM that sees the whole document.
- **Pipeline**:
  - Ingest: each chunk → LLM generates a 50–100 token context → prepend → embed.
  - Query: standard.
- **Cost**: LLM call per chunk at ingest; prompt-caching makes this affordable (the document is the cache).
- **Anthropic-reported gains**:
  - Pure dense: baseline.
  - Contextual: +35% retrieval failure rate reduction.
  - + BM25 + RRF: even better.
  - + reranking: best.

**Common follow-ups.**
- "Cost at scale?" → With prompt caching, the per-chunk LLM call is cheap (the document text is cached).
- "Why isn't this universal?" → New (2024); requires LLM ingest pipeline; many teams haven't migrated.

**Common mistakes.**
- Skipping the context-prefix; small chunks alone lose meaning.

**References.**
- [Anthropic — "Introducing Contextual Retrieval"](https://www.anthropic.com/news/contextual-retrieval).

---

### Q: How do you A/B test a RAG improvement in production?

**Category:** concept
**Difficulty:** senior
**Tags:** [ab-test, rag-eval, production]

**Short answer.** (1) Define the metric (deflection, CSAT, time-to-resolution). (2) Bucket users 50/50 to control (old RAG) vs treatment (new RAG). (3) Hold for sufficient sample size (calculated via power analysis). (4) Monitor guardrails (latency, error rate). (5) Decide: ship if metric improves and guardrails unchanged; rollback if either regresses; iterate. Use offline eval first to filter clear losers before online tests.

**Expansion / why this is the answer.**
- See T7 A/B testing entries for the broader frame.
- RAG-specific signals:
  - Retrieval recall (offline).
  - Faithfulness (offline, LLM-judge).
  - User thumbs up/down (online).
  - Refusal rate.
  - Session-level metrics.

**Common follow-ups.**
- "What's a typical RAG A/B duration?" → 1–4 weeks depending on traffic.
- "Multi-variant?" → Yes — multiple new variants vs. control; correct for multiple testing.

**Common mistakes.**
- Skipping the offline gate; A/B-testing clear losers wastes traffic.

**References.**
- [Kohavi et al. — *Trustworthy Online Controlled Experiments*](https://experimentguide.com/).

---

### Q: What is "negative caching" in RAG production?

**Category:** concept
**Difficulty:** mid
**Tags:** [negative-cache, ood, monitoring]

**Short answer.** Track queries that consistently fail retrieval (no high-confidence match in the index). Negative caching: store these queries; periodically review to determine: (a) Should they be added to the corpus? (b) Should the model refuse more politely? (c) Is the retrieval failing on the corpus, or is it truly out-of-scope? Feedback loop for content expansion.

**Expansion / why this is the answer.**
- **Why**: queries that the corpus can't answer should be visible; they're product-improvement opportunities or refusal-tuning opportunities.
- **Implementation**:
  - Log queries where top-K retrieval scores are all below threshold.
  - Periodic review (weekly).
  - Tag: "add to corpus," "out of scope," "phrasing issue (rewrite query)."
- **Closes the loop**:
  - Corpus growth.
  - Better refusal training.

**Common follow-ups.**
- "Privacy?" → Queries may contain PII; redact or aggregate before review.
- "Volume?" → Production logs of OOD queries can be huge; sample.

**Common mistakes.**
- Just ignoring the OOD queries; missing the product-improvement signal.

**References.**
- [Barnett et al. — "Seven Failure Points When Engineering a RAG System"](https://arxiv.org/abs/2401.05856).

---

### Q: Compare BM25, dense-only, ColBERT, and hybrid retrieval on a specific corpus.

**Category:** concept
**Difficulty:** senior
**Tags:** [retrieval-comparison, hybrid]

**Short answer.** Concrete pattern: on a corpus of help docs with mixed exact-name (products) and abstract (concepts) queries:
- **BM25**: nails exact-name queries; fails on paraphrases.
- **Dense-only** (BGE / E5): nails paraphrases; misses exact names.
- **ColBERT**: best precision at low K; storage cost.
- **Hybrid BM25 + dense + RRF**: combines strengths; production default. Add reranker on top of any: another 5–15 nDCG points.

**Expansion / why this is the answer.**
- See T5 base hybrid retrieval entry for fuller treatment.
- The summary table is the interview-grade quick answer.

**Common follow-ups.**
- "When does dense-only suffice?" → Homogeneous semantic corpus; no exact-name traffic.
- "When does BM25-only suffice?" → Highly structured corpus (legal cases, product SKUs).

**Common mistakes.**
- Dense-only because it's "modern" — leaves performance on the table.

**References.**
- [Robertson & Zaragoza — BM25](https://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf).
- [Cormack et al. — "RRF"](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).

---

### Q: How do you handle "the user wants A but the doc says B" contradictions in RAG?

**Category:** concept
**Difficulty:** senior
**Tags:** [contradictions, faithfulness]

**Short answer.** RAG's job: ground the answer in the retrieved docs. If the user's premise contradicts the docs (e.g. "my plan has feature X" but the docs say it doesn't), the model should explicitly flag the discrepancy: "Based on the documentation, plan X doesn't include feature Y; you may be thinking of plan Z." Don't agree with the user when the docs say otherwise.

**Expansion / why this is the answer.**
- **Failure modes**:
  - **Sycophancy**: model agrees with the user's wrong premise.
  - **Hallucination**: model invents a way to confirm the user's claim.
- **Correct behavior**:
  - Acknowledge the user's framing.
  - Cite what the docs say.
  - Resolve the discrepancy: "I think there may be confusion."
- **Training**: include preference pairs where confirming the user's wrong premise is the rejected option.

**Common follow-ups.**
- "How do you measure sycophancy?" → Adversarial eval set with deliberately-wrong user premises.
- "What if the docs are wrong?" → Outside the model's scope; escalate; flag to the corpus owner.

**Common mistakes.**
- Optimizing only on user satisfaction (CSAT) — encourages sycophancy.

**References.**
- [Sharma et al. — "Sycophancy in LLMs"](https://arxiv.org/abs/2310.13548).

---

### Q: What's "self-querying retrieval"?

**Category:** concept
**Difficulty:** mid
**Tags:** [self-query, structured-filtering]

**Short answer.** The LLM examines the user query and constructs a *structured query* over the index — both a semantic search term and metadata filters. For "show me articles about RAG from 2024," the LLM produces: `search_text="RAG", filter={"year": 2024}`. Combines semantic and exact filtering in one step.

**Expansion / why this is the answer.**
- The pipeline:
  - User query.
  - LLM extracts: semantic content + metadata constraints.
  - Combined query: vector search + metadata filter.
- **Helps when**: queries naturally include filters (dates, categories, languages).
- **Implementations**: LangChain's `SelfQueryRetriever`, custom code with structured-output LLM calls.

**Common follow-ups.**
- "What if the LLM mis-extracts filters?" → Validation step; fall back to pure semantic search.
- "Cost?" → One small LLM call per query.

**Common mistakes.**
- Hard-coding filter rules instead of LLM extraction — brittle.

**References.**
- [LangChain Self-Querying docs](https://python.langchain.com/docs/how_to/self_query/).

---

### Q: What is "step-back prompting" for RAG?

**Category:** concept
**Difficulty:** mid
**Tags:** [step-back, retrieval-augmentation]

**Short answer.** Step-back prompting (Zheng et al. 2023): before retrieving for a specific question, the LLM generates a more *general* version of the question ("What are the principles of X?" before "Why did X behave like Y in 2023?"). Retrieve both the specific and general queries; the general one often surfaces relevant background that direct retrieval misses.

**Expansion / why this is the answer.**
- **The technique**:
  - Original Q: specific.
  - LLM generates: "step back to the general question."
  - Retrieve for both; combine.
- **When it helps**: questions with implicit context the user didn't articulate.
- **Cost**: one extra LLM call + one extra retrieval round.

**Common follow-ups.**
- "Relation to multi-query?" → Step-back generates a *more general* query; multi-query generates *paraphrased* queries.
- "Empirical wins?" → Step-back paper shows gains on STEM-QA and complex reasoning.

**Common mistakes.**
- Step-back on already-general queries; redundant.

**References.**
- [Zheng et al. — "Take a Step Back"](https://arxiv.org/abs/2310.06117).

---

### Q: How do you handle "table-heavy" documents (e.g. financial reports) in RAG?

**Category:** concept
**Difficulty:** senior
**Tags:** [tables, document-structure, parsing]

**Short answer.** Two-step ingestion: (a) **structured parser** (LlamaParse, Unstructured, Azure Document Intelligence) extracts tables as structured data, not flat text; (b) **table-aware chunking**: each table becomes its own chunk; cells preserved; surrounding context (caption + adjacent paragraphs) included. At retrieval: separate index for tabular content. Generic chunkers destroy table structure; this is a common failure mode.

**Expansion / why this is the answer.**
- **The problem**: a PDF's table flattens to row-major or column-major text; cells lose alignment.
- **Modern parsers**:
  - **LlamaParse** (LlamaIndex): structured PDF → markdown tables.
  - **Unstructured.io**: open-source parser with table extraction.
  - **Azure Document Intelligence**: cloud service.
- **Chunking**:
  - Tables as discrete chunks.
  - Include caption + surrounding paragraphs as context.
- **Retrieval**:
  - Embeddings can be on the markdown-table; cross-encoder rerank.
  - LLM presented with the structured markdown table.
- **For numerical Q&A** (financial figures): consider a code-interpreter step over the structured table.

**Common follow-ups.**
- "PDFs with images of tables?" → OCR + table extraction; messier; current OCR systems vary.
- "Excel files?" → Direct parsing; one chunk per worksheet or per logical table.

**Common mistakes.**
- Generic chunking destroys tables.

**References.**
- [LlamaParse docs](https://docs.cloud.llamaindex.ai/llamaparse/getting_started).
- [Unstructured.io](https://unstructured.io/).

---

### Q: What is "agentic-chunking" / late chunking?

**Category:** concept
**Difficulty:** senior
**Tags:** [late-chunking, agentic-chunking, jina]

**Short answer.** **Late chunking** (Jina, Günther et al. 2024): embed the *full document* with a long-context encoder, then chunk *the embeddings* (not the text). Each chunk's embedding inherits global context from the whole document. Solves the "chunks lose context" problem at the embedding-model level rather than via prompt engineering (contextual retrieval).

**Expansion / why this is the answer.**
- **Standard chunking** (early): chunk text → embed each chunk independently. Each chunk's embedding has no document-level context.
- **Late chunking**: embed the full document → pool the embedding over each chunk's token span. Each chunk's pooled embedding reflects the document context.
- **Requirement**: long-context encoder (8k+).
- **Variants**:
  - Late chunking with a long-context encoder.
  - Contextual retrieval (Anthropic): prepend context text before embedding (still early chunking).
- **Comparison**: late chunking is model-side; contextual retrieval is data-side.

**Common follow-ups.**
- "Performance comparison?" → Both improve over naive chunking; late chunking is simpler at ingest but needs a long-context encoder.

**Common mistakes.**
- Conflating late chunking with contextual retrieval; different mechanisms.

**References.**
- [Jina AI — "Late Chunking"](https://arxiv.org/abs/2409.04701).

---

### Q: How do you handle "stale documents" in RAG?

**Category:** concept
**Difficulty:** mid
**Tags:** [staleness, doc-update, indexing]

**Short answer.** Track `updated_at` per doc; re-embed and re-index when content changes. Versioning: keep doc versions or replace. Stale-detection: periodically scan source for changes; pipeline triggers re-ingest. For time-sensitive queries, filter or boost by recency. For privacy/PII removal: explicit deletion from the index (and any caches).

**Expansion / why this is the answer.**
- **Re-ingest pipeline**:
  - Detect changed docs (file mtime, hash, source API webhook).
  - Re-embed changed chunks.
  - Replace in the index.
- **Versioning**:
  - Replace: simple; lose history.
  - Version with date: keep history; queries can filter by version.
- **For RAG with citations**: ensure the citation links to the version that was retrieved.
- **Deletion (GDPR, right-to-be-forgotten)**:
  - Remove from index.
  - Remove from any cached results.
  - Audit log of deletions.

**Common follow-ups.**
- "Incremental re-embedding?" → Yes — only re-embed changed chunks.
- "What about cached responses?" → Invalidate on doc update.

**Common mistakes.**
- No incremental updates; periodic full re-index wastes resources.

**References.**
- [LlamaIndex updating docs](https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_management/).

---

### Q: How does Anthropic's "Citations API" actually validate citations?

**Category:** concept
**Difficulty:** mid
**Tags:** [citations, anthropic, attribution]

**Short answer.** Anthropic's Citations API: when you provide documents, Claude can output structured citations that include the *specific passage* the citation refers to. The API validates that citations reference actual passages in the supplied documents — preventing the model from inventing fake citations. Output is a structured `[{text, citation: {document_index, start_char, end_char, source_text}}]`.

**Expansion / why this is the answer.**
- Input: list of documents + user query.
- Output: structured response with per-sentence citations.
- Each citation: which document, what character range in that doc.
- The API checks: does the citation range actually exist in the provided document? If not, the API rejects the citation.
- Doesn't validate *semantic correctness* (whether the citation actually supports the claim) — only structural existence.

**Common follow-ups.**
- "Does it prevent hallucinated citations?" → Structural ones, yes; semantic ones, partially (the model is trained to cite faithfully).
- "Versus parsing citations from free text?" → API is structured; less brittle.

**Common mistakes.**
- Trusting the model's free-text citations without API validation.

**References.**
- [Anthropic — Citations API docs](https://docs.anthropic.com/en/docs/build-with-claude/citations).

---

### Q: What's "RAG over private data" / on-prem RAG architecture?

**Category:** concept
**Difficulty:** senior
**Tags:** [on-prem, private-rag, enterprise]

**Short answer.** Enterprise on-prem RAG: all components run inside the organization's perimeter. Components: (a) document parsing on-prem; (b) embedding model self-hosted (BGE, E5, or a fine-tune); (c) vector store on-prem (Qdrant, Weaviate, pgvector); (d) reranker self-hosted (BGE-reranker); (e) LLM self-hosted (Llama, Mistral, Mixtral) or via private cloud (Azure OpenAI with VNet). Strict ACL on every layer.

**Expansion / why this is the answer.**
- **Requirements**:
  - No data leaves the perimeter.
  - Audit logs for compliance.
  - ACL respected at every layer.
- **Component choices**:
  - **Embedding**: BGE-large, E5-large, mxbai-embed-large (open-source).
  - **Vector store**: Qdrant, Weaviate, pgvector, Milvus.
  - **Reranker**: BGE-reranker.
  - **LLM**: Llama 3, Mistral, Mixtral, DeepSeek-V3 on internal GPUs.
- **Hybrid**: some teams use cloud LLM (Anthropic with HIPAA / SOC2 / private cloud) + on-prem retrieval.
- **ACL**:
  - At ingest: tag each chunk with allowed-roles.
  - At retrieval: filter by user's roles.
  - At LLM: ensure no out-of-ACL content in the prompt.

**Common follow-ups.**
- "Why on-prem over private cloud?" → Strictest data sovereignty; some regulated industries require it.
- "Performance on-prem vs. cloud?" → Often slower; offset by data-sovereignty wins.

**Common mistakes.**
- ACL only at the UI layer; the index must enforce it.

**References.**
- [Anthropic — Enterprise SOC 2 docs](https://www.anthropic.com/trust).
- [Qdrant docs — security](https://qdrant.tech/documentation/guides/security/).

---

### Q: What is "semantic-aware chunking" via embeddings?

**Category:** concept
**Difficulty:** mid
**Tags:** [semantic-chunking, sentence-similarity]

**Short answer.** Semantic chunking: embed each sentence; split documents at points where adjacent sentences are semantically dissimilar (low embedding cosine). Reflects topical shifts rather than fixed character counts. Implementations: LlamaIndex's `SemanticSplitterNodeParser`. Modest quality gains; higher ingest cost.

**Expansion / why this is the answer.**
- **The technique**:
  - Embed each sentence.
  - Compute adjacent sentence similarity.
  - Find local minima (topical boundaries).
  - Split there.
- **Tuning**:
  - Threshold (percentile of similarity scores) controls chunk size.
  - Buffer: keep adjacent sentences for context.
- **Cost**: more embedding calls at ingest.
- **Quality**:
  - Better when documents have clear topical structure (essays, articles).
  - Marginal on uniform text (code, manuals).

**Common follow-ups.**
- "Does it beat recursive splitting?" → Often by a small margin; tune on your corpus.

**Common mistakes.**
- Always using semantic chunking; cost rarely justifies it for uniform corpora.

**References.**
- [LlamaIndex Semantic Splitter docs](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/).

---

### Q: How do you handle multilingual queries against a single-language corpus?

**Category:** concept
**Difficulty:** mid
**Tags:** [cross-lingual, multilingual, retrieval]

**Short answer.** Two approaches: (1) **Multilingual embedding model** that maps query and doc to the same space regardless of language; query in any language matches docs in the corpus language. (2) **Machine translation** at query time: detect the query language; translate to corpus language; retrieve normally. The first is cleaner; the second works when you only have an English embedder.

**Expansion / why this is the answer.**
- **Multilingual embeddings**: BGE-M3, Cohere multilingual-embed-v3, OpenAI text-embedding-3 all handle this natively.
- **Translation pipeline**:
  - Detect query language (langid).
  - Translate to corpus language with NMT.
  - Retrieve as normal.
- **For LLM answering**:
  - If answering in user's language: translate retrieved passages OR rely on the LLM's multilingual capability.

**Common follow-ups.**
- "Translation quality?" → Modern NMT is good; rare-language pairs less reliable.
- "Where do biases creep in?" → Translation may shift meaning; named entities especially.

**Common mistakes.**
- English-only embedder + non-English queries; recall tanks.

**References.**
- [Chen et al. — "BGE-M3"](https://arxiv.org/abs/2402.03216).

---

### Q: What is "query understanding" / query classification in production search?

**Category:** concept
**Difficulty:** mid
**Tags:** [query-understanding, classification]

**Short answer.** Before retrieval, classify the query: (a) intent (navigational, informational, transactional); (b) entities mentioned; (c) detected language; (d) detected NSFW / abusive. Used to route to the right index, apply filters, or modify the search behavior. Standard component of production search systems.

**Expansion / why this is the answer.**
- **Components**:
  - Intent classifier (small ML model or LLM).
  - Named-entity recognition.
  - Language detection.
  - Spam / abuse filter.
- **Use cases**:
  - Navigational ("Acme login page"): direct to known URL, skip RAG.
  - Informational ("what is X?"): full RAG.
  - Transactional ("buy X"): different ranking signals.
- **LLM-based query understanding**: structured-output LLM call returning the analysis JSON.

**Common follow-ups.**
- "Latency impact?" → 10–100ms for the classifier; sometimes parallelizable with retrieval.
- "When does query understanding fail?" → Unusual phrasings, code-mixed languages.

**Common mistakes.**
- Skipping query understanding; one-size-fits-all retrieval misses easy wins.

**References.**
- [Manning, Raghavan, Schütze — *IR Book*](https://nlp.stanford.edu/IR-book/) — classical query analysis.

---

### Q: What is the role of a "query rewriter" before retrieval?

**Category:** concept
**Difficulty:** mid
**Tags:** [query-rewriting, multi-turn]

**Short answer.** Rewrites the user's query into a self-contained form suitable for retrieval — typically in multi-turn chats where the query references prior context ("what about its CEO?" → "what about Acme's CEO?"). Implemented by a small LLM. Critical for multi-turn RAG; vanilla retrieval doesn't handle context-dependent queries.

**Expansion / why this is the answer.**
- **The problem**: a user's follow-up question depends on prior turns; the retriever doesn't see those.
- **The fix**: a small LLM rewrites the query, expanding pronouns and references using the conversation history.
- **Examples**:
  - User: "Tell me about Acme Corp."
  - Model: [response]
  - User: "What's its revenue?"
  - Rewriter: "What's Acme Corp's revenue?"
- **Implementation**: cheap LLM call; can be cached for retry scenarios.

**Common follow-ups.**
- "What if the rewriter makes a wrong inference?" → Fall back to original query if confidence is low; LLM-judge step.

**Common mistakes.**
- Naive retrieval on the raw follow-up; "What about its CEO?" returns nothing useful.

**References.**
- [Yu et al. — "Few-Shot Generative Conversational Query Rewriting"](https://arxiv.org/abs/2006.05009).

---

### Q: How do you handle PDFs with mixed text, tables, figures in RAG ingest?

**Category:** concept
**Difficulty:** senior
**Tags:** [pdf-ingest, multimodal, parsing]

**Short answer.** Use a structured-PDF parser that handles each element separately: (a) text → standard chunking; (b) tables → preserved as markdown / structured rows; (c) figures → OCR / image-caption / store separately; (d) layout → preserve hierarchy. Modern services: LlamaParse, Unstructured, Azure Document Intelligence, AWS Textract. Generic text-only extractors lose 30–50% of useful content on table/figure-heavy docs.

**Expansion / why this is the answer.**
- The challenge: PDFs are layout-rich; flattening to text loses information.
- **Component-wise extraction**:
  - **Text**: paragraphs, headings, lists.
  - **Tables**: structured rows / markdown tables; preserve cell alignment.
  - **Figures**: image; OCR if text-on-image; image-captioning model for descriptions.
  - **Equations**: LaTeX or math markdown.
  - **References / citations**: structured.
- **Modern tooling**:
  - **LlamaParse**: cloud service; multimodal; good with tables.
  - **Unstructured.io**: open-source.
  - **PyMuPDF / pdfplumber**: lightweight, OK for simple PDFs.
  - **Azure Document Intelligence** / **AWS Textract**: cloud-grade.

**Common follow-ups.**
- "Cost?" → Cloud parsers: $0.01–0.10 per page.
- "When can you skip structured parsing?" → Pure text PDFs (academic papers without figures).

**Common mistakes.**
- Default text extraction → tables and figures destroyed.

**References.**
- [LlamaParse](https://docs.cloud.llamaindex.ai/llamaparse/getting_started).

---

### Q: What is "RAG triad" of metrics (groundedness + answer relevance + context relevance)?

**Category:** concept
**Difficulty:** mid
**Tags:** [rag-triad, evaluation, faithfulness]

**Short answer.** TruLens / RAGAS-style RAG eval often centers on three metrics: **Context relevance** (are retrieved passages relevant to the query?); **Groundedness / faithfulness** (does the answer follow from the context?); **Answer relevance** (does the answer address the question?). Together they cover the three failure modes: bad retrieval, hallucination despite good retrieval, off-topic generation.

**Expansion / why this is the answer.**
- The triad:
  1. **Context relevance**: filter step. Is the retrieved context relevant?
  2. **Groundedness**: generation step. Does the answer stay within what context says?
  3. **Answer relevance**: end-to-end. Does the answer address the user?
- **All three need to be high** for a good RAG response.
- **Measured by LLM-judge** in production; hand-calibration on a held-out set.

**Common follow-ups.**
- "What does TruLens do?" → Open-source RAG evaluation framework implementing the triad.

**Common mistakes.**
- Optimizing only end-to-end accuracy; can't diagnose what's failing.

**References.**
- [Es et al. — "RAGAS"](https://arxiv.org/abs/2309.15217).
- [TruLens documentation](https://www.trulens.org/).

---

### Q: How does Anthropic's "tool-use as retrieval" pattern differ from RAG?

**Category:** concept
**Difficulty:** senior
**Tags:** [tool-use-as-retrieval, agentic-rag]

**Short answer.** Agentic pattern: instead of pre-retrieving passages and stuffing them in the context, expose a `search_knowledge_base(query)` tool to the model; the model decides when (and what) to search. The model can iterate: search, read results, search again with a refined query. More flexible than vanilla RAG; higher latency and cost.

**Expansion / why this is the answer.**
- **Vanilla RAG**: retrieve once → generate.
- **Tool-use as retrieval**: model decides when to search via tool calls.
- **Benefits**:
  - Model only searches when needed (skip on easy queries).
  - Can issue multiple refined queries.
  - Handles multi-hop naturally.
- **Costs**:
  - Multiple LLM round trips.
  - Longer latency.

**Common follow-ups.**
- "When does tool-use beat vanilla RAG?" → Multi-hop, open-ended, exploratory queries.
- "Production use?" → Claude with tool use; OpenAI Assistants API; agentic frameworks.

**Common mistakes.**
- Treating tool-use as universally better; for single-hop Q&A, vanilla RAG is faster.

**References.**
- [Anthropic — Tool use docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview).

---
