# D6 — LLM-powered customer-support assistant (RAG)

A worked drill following the [6-step structure](../README.md#the-canonical-6-step-structure).

---

### Q: Design a customer-support assistant that answers user questions using your company's help docs.

**Category:** system-design
**Difficulty:** senior
**Tags:** [rag, customer-support, llm-app, citations]

**Short answer.** Production RAG: ingest help-docs into a hybrid (BM25 + dense) index → query rewrite + retrieve top-100 → cross-encoder rerank to top-5 → assemble prompt with retrieved passages → call an LLM with strict citation requirements → verify citations server-side → log all turns for evaluation. Guardrails: refuse out-of-scope, escalate to humans when confidence is low, never invent product details. Evaluate end-to-end on a curated set with deflection rate, faithfulness, and CSAT.

**Expansion / why this is the answer.**

**1. Clarify requirements.**
Functional:
- User asks a question (text); system returns an answer grounded in the help docs.
- Cite which doc supports each claim.
- Handle multi-turn (follow-up questions).
- Escalate to human agent when uncertain or for sensitive topics (billing disputes, legal).

Non-functional:
- Latency: p99 < 5 s for first-token (it's a chat UX); streaming preferred.
- Throughput: depends on traffic — 100s to 10000s of concurrent sessions.
- Knowledge scale: 1k–100k help-doc pages.
- Update cadence: docs change frequently; the index must update within hours.
- Languages: depends on product.

Clarifying Qs:
- "What topics are in scope? (Anything from billing to API docs?)"
- "Is this a chat product or one-shot Q&A?"
- "Is human-agent handoff a hard requirement?"
- "What's the tolerated hallucination rate?"
- "What's the budget per session?"

**2. Define metrics.**

Online:
- **North-star**: deflection rate (% sessions resolved without human handoff).
- **CSAT**: user-reported satisfaction.
- **Escalation rate**: lower is better (but not zero).
- **Time-to-resolution** vs. baseline human-only.

Offline:
- **Retrieval recall@5**: did the right doc make it into the prompt?
- **Faithfulness**: every claim in the answer supported by retrieved docs.
- **Answer relevance**: did the answer address the question?
- **Refusal accuracy**: appropriate refusals vs. over-refusal.
- **Citation accuracy**: are cited docs the ones actually supporting claims?

Online ↔ offline: maintain a labeled set of `(question, gold-doc, gold-answer)`. Track offline weekly; A/B online.

**3. Data and labels.**

Sources:
- Help-docs corpus.
- Internal KB articles.
- Past support tickets (for question patterns; not for verbatim answers — privacy).
- ACL metadata if some docs are user-segment-specific.

Eval set:
- 200–500 hand-crafted `(question, gold doc, gold answer)` triples from real ticket patterns.
- Multiple plausible answers per question; hand-validated.

Decontamination:
- If using LLM API, no PII in prompts unless required and with consent.
- If fine-tuning, scrub PII from training data.

**4. Modeling.**

**4a. Ingest**:
- Doc parsing (markdown, HTML, PDF) preserving structure (headings, code blocks, tables).
- Chunking: recursive, 400–600 tokens, 50–100 overlap; respect doc boundaries.
- Embedding: an instruction-tuned encoder (E5, BGE, or OpenAI / Cohere).
- Index: hybrid (BM25 + HNSW) in pgvector / Qdrant + Elasticsearch.
- Metadata: doc id, version, last-updated, ACL, language.

**4b. Query path**:
- (Optional) Query rewriting / disambiguation for multi-turn ("its CEO" → "Acme's CEO"). LLM-based, cheap model.
- Hybrid retrieve top-100 (50 dense + 50 BM25, RRF fuse).
- Cross-encoder rerank to top-5.
- Filter by ACL / language.

**4c. Generation**:
- System prompt: scope policy, citation requirement, refuse-when-out-of-scope.
- Format: `[doc-1]\n<passage>\n[doc-2]\n<passage>\n... User question: ...`
- Model: a strong base model (Claude 4, GPT-5, Gemini 2.x); option to fall back to a cheaper model for short queries.
- Instructions: cite [doc-N] after each factual claim.
- Stream tokens to the user.

**4d. Verification**:
- After generation, parse citations and validate each against retrieved passages (lightweight NLI or LLM-judge).
- Unsupported claims → either drop or flag.
- Out-of-scope → "I can help with X but not Y; let me get a human."

**4e. Escalation**:
- Confidence-based: if retrieval returned nothing scoring above a threshold, or if the answer fails verification, escalate.
- Sensitive topics → always escalate.

**5. Serving.**

Latency budget:
- Retrieve + rerank: 200–400 ms.
- LLM streaming first-token: 500 ms – 2 s.
- Total before first user-visible content: < 2 s p99.

Infrastructure:
- Vector + BM25 store.
- Reranker (cross-encoder) on a small GPU pool.
- LLM via API (with prompt caching for the system prompt) or self-hosted via vLLM.
- Logging + observability stack (Phoenix, LangSmith, internal).

Cost:
- API: $0.001–0.05 per session depending on tokens.
- Self-host: amortized over scale; lower at high QPS.

**6. Monitoring + iteration.**

Drift:
- Doc updates → re-embed + re-index.
- New question patterns from ticket logs → expand eval set.

Retraining:
- The model isn't retrained; the retriever might be fine-tuned monthly.
- Re-evaluate on the gold set weekly.

Evals:
- Daily: spot-check 1% of sessions with LLM-judge.
- Weekly: human-grade ~50 sessions.
- Quarterly: full re-run on the 500-sample eval set.

Alerting:
- Deflection rate drop > X pp → page on-call.
- Faithfulness drop → freeze prompt/model changes.
- Latency p99 > SLO → page.

User feedback:
- Thumbs up/down on every answer; aggregate into a daily quality signal.
- Negative thumbs feed into the eval-set expansion.

---

**Common follow-ups.**

- "What if a user asks a non-support question?" → Refuse and redirect to documentation or general help.
- "How do you handle product-specific terminology?" → It's already in the help-docs; the embeddings learn it. For very niche terms, consider a small terminology fine-tune of the retriever.
- "What about multi-language?" → Multilingual embedding (BGE-M3, Cohere multilingual) + per-language docs index.

**Common mistakes.**

- Skipping reranking; top-5 dense alone is noisy.
- No citation verification; trusting the model's own citation claims.
- No escalation path; the bot fights you to the death.
- Building it without an eval set; no way to know if it's working.

**References.**

- [Lewis et al. — RAG](https://arxiv.org/abs/2005.11401) — RAG paradigm.
- [Es et al. — "RAGAS"](https://arxiv.org/abs/2309.15217) — RAG eval framework.
- [Anthropic — Building effective agents](https://www.anthropic.com/research/building-effective-agents) — production patterns.
- [Anthropic — Citations API docs](https://docs.anthropic.com/en/docs/build-with-claude/citations) — verified citation.
