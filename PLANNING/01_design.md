# Phase 1 — DESIGN

> Status: complete. The taxonomy and the Q&A schema below are the product.

This phase locks down (a) the canonical topic taxonomy a candidate must master, (b) the per-entry schema that makes every answer trustworthy, (c) the time-boxed study plans with persona variants, (d) the repo architecture, (e) the README wireframe, (f) the tooling design with the reference-required validation rule, (g) the contribution flow, and (h) the verification plan.

---

## 1. The prep taxonomy

The 12-topic spine. Order is the recommended study order (foundation up; system design near the end as the integrating discipline).

### T1 — ML & DL Fundamentals
Concepts:
- Bias–variance, overfitting/underfitting, regularization (L1/L2/dropout/early stopping/weight decay).
- Supervised vs. unsupervised vs. self-supervised; classification vs. regression.
- Loss functions (MSE, cross-entropy, hinge, KL, focal); when each is appropriate.
- Optimization (SGD, momentum, RMSProp, Adam, AdamW, Lion); learning rate schedules (cosine, warmup, one-cycle).
- Activation functions (ReLU, GELU, SwiGLU, sigmoid, tanh); vanishing/exploding gradients.
- Normalization (BatchNorm, LayerNorm, RMSNorm, GroupNorm) — what each normalizes over and why.
- Initialization (Xavier/Glorot, He/Kaiming) and why initialization matters.
- Classical algorithms (linear/logistic regression, trees & GBMs, SVMs, kNN, naive Bayes, k-means, PCA) — when each wins.
- Evaluation metrics (accuracy, precision/recall, F1, ROC-AUC, PR-AUC, calibration, log-loss, RMSE, MAE).
- Cross-validation, train/val/test splits, leakage, class imbalance handling.

Representative-question depth: foundational (definitions, contrasts, when-to-use); also derivations (logistic regression gradient, cross-entropy from MLE).

### T2 — Transformers & LLM Internals
Concepts:
- Self-attention math: Q, K, V; scaling by √d_k; softmax; multi-head.
- Positional encodings: sinusoidal, learned, ALiBi, RoPE (and YaRN/NTK extensions).
- Encoder–decoder vs. encoder-only vs. decoder-only; when each is used.
- Normalization placement (pre-norm vs. post-norm) and training stability implications.
- MoE: gating, top-k routing, load balancing loss, expert capacity, fine-grained experts (DeepSeek-V3 style).
- Attention variants: MHA, MQA (multi-query), GQA (grouped-query); KV-cache implications.
- Long-context techniques: FlashAttention (1/2/3), paged attention, ring attention.
- Tokenization: BPE, WordPiece, SentencePiece, byte-level BPE; tokenizer side effects (numbers, code, multilingual).
- Scaling laws: Kaplan (2020) vs. Chinchilla (Hoffmann 2022) and why the Chinchilla revision mattered.
- Emergent abilities: claim, critique (Schaeffer 2023), and what an interviewer wants you to know.

### T3 — Training & Fine-Tuning
Concepts:
- Pretraining objectives: next-token prediction, masked LM, span corruption, T5-style.
- Data curation: dedup (MinHash/SimHash), filtering, decontamination from eval sets, mixing ratios.
- Instruction tuning (SFT): how it differs from pretraining, dataset shape, common pitfalls.
- Preference optimization: RLHF (PPO), DPO, IPO, KTO, GRPO; what each optimizes; when each wins.
- Reward modeling: pairwise preference, Bradley-Terry, reward hacking, reward over-optimization.
- Parameter-efficient fine-tuning: LoRA, QLoRA, DoRA, prefix tuning, adapters; rank choice, scaling.
- Distillation: response, feature, attention; teacher–student tradeoffs.
- Distributed training: data parallel, ZeRO (1/2/3), tensor parallel, pipeline parallel, sequence parallel, FSDP.
- Mixed precision (fp16 vs bf16) and numerical stability; gradient accumulation; gradient checkpointing.
- Curriculum / data ordering effects.

### T4 — Inference & Serving
Concepts:
- KV cache: what it stores, memory cost, GQA/MQA effect, paged attention (vLLM).
- Quantization: weight-only (INT8/INT4, GPTQ, AWQ); weight+activation (SmoothQuant); FP8.
- Speculative decoding: draft model, target model, acceptance probabilities; Medusa, EAGLE variants.
- Continuous / inflight batching vs. static batching.
- Throughput vs. latency tradeoffs: prefill vs. decode phases; first-token vs. inter-token latency.
- Serving stacks: vLLM, TensorRT-LLM, SGLang, TGI; tradeoffs.
- Cost math: GPU type × batch × tokens-per-second × utilization → $/Mtok.
- Long-context inference tricks: prompt caching (Anthropic), KV reuse, prefix caching.

### T5 — Retrieval & RAG
Concepts:
- Embeddings: dual-encoder vs. cross-encoder; pooling strategies; instruction-aware embeddings.
- Chunking strategies: fixed, recursive, semantic, hierarchical; tradeoffs.
- Indexes: HNSW, IVF, IVFPQ, ScaNN; precision/recall/latency tradeoffs.
- Hybrid search: dense + BM25; reciprocal rank fusion.
- Reranking: cross-encoder rerankers, LLM-as-reranker.
- Query rewriting, HyDE, multi-query expansion, decomposition.
- Evaluation: retrieval (recall@k, nDCG, MRR), generation (faithfulness, answer relevance), end-to-end.
- Failure modes: lost-in-the-middle, distractor sensitivity, multi-hop, contradictory contexts.

### T6 — Agents & Harnesses
Concepts:
- Tool use: schema design, parallel tool calls, error recovery.
- Planning patterns: ReAct, Plan-and-Execute, Reflexion, Tree-of-Thoughts.
- Multi-step reliability: failure-mode taxonomy; verifier patterns; rollback.
- Harness design: context-window management, summarization, memory.
- Agent evaluation: SWE-bench (Verified/Live), TAU-bench, GAIA, AgentBench; what each measures.
- Cost/latency control in agentic flows.
- Multi-agent vs. single-agent tradeoffs; when multi-agent actually helps.

### T7 — Evaluation & Calibration
Concepts:
- Holdouts and contamination: how to know whether a benchmark is leaked.
- LLM-as-judge: bias, position bias, length bias, self-preference; mitigations.
- Pairwise vs. single-grade evaluation; reference-free vs. reference-based.
- Calibration: ECE, reliability diagrams; temperature scaling.
- Hallucination measurement: TruthfulQA, FActScore, FaithBench; limits.
- Eval set design: representativeness, sample size, statistical significance.
- Production evals: shadow traffic, A/B, online metrics vs. offline metrics.

### T8 — ML System Design (interview drills)
Format: each drill is a worked design walkthrough following a canonical 6-step structure:
1. Clarify requirements (functional + non-functional)
2. Define metrics (online + offline)
3. Data and labels
4. Modeling (candidates, features, architecture, training data, training procedure)
5. Serving (latency, scale, freshness, infrastructure)
6. Monitoring + iteration (drift, retraining cadence, A/B)

Drills (representative; each becomes a full doc):
- D1. YouTube/TikTok video recommendation
- D2. Ad click-through-rate prediction
- D3. Content moderation pipeline (multi-modal)
- D4. Semantic search for e-commerce
- D5. Personalized news feed ranking
- D6. **LLM-powered customer-support assistant (RAG)** — AI-era
- D7. **AI coding-assistant / autocomplete** — AI-era
- D8. **Multi-turn LLM agent for an enterprise workflow** — AI-era
- D9. Spam / fraud detection
- D10. ML feature store design

### T9 — ML/AI Coding Questions
Concepts and implementations:
- NumPy: implement attention, softmax with numerical stability, layer norm, batch norm.
- PyTorch: a transformer block from scratch, a training loop, gradient accumulation, mixed-precision training.
- From-scratch algorithms: k-means, kNN, logistic regression, decision tree split, gradient descent on a quadratic.
- Sampling: nucleus, top-k, temperature, beam search.
- BPE tokenizer implementation (toy).
- Cosine similarity, top-k retrieval over an embedding matrix.
- Common bugs: argmax instead of softmax, in-place ops breaking gradients, broadcasting traps.

### T10 — Research Discussion & Paper Deep-Dives
Concepts:
- How to talk about your own work (the 30-second / 5-minute / 30-minute versions).
- How to read a paper live in 15 minutes (interview drill).
- How to handle "what's a paper you read recently?"
- Critique structure: contribution → method → eval → threats to validity → followups.
- Bench-suite literacy: HumanEval, MBPP, GSM8K, MATH, MMLU/MMLU-Pro, GPQA, BIG-Bench Hard, AGIEval, ARC-AGI, SWE-bench.

### T11 — Behavioral & Communication
Concepts:
- STAR structure for behavioral answers.
- Common questions: tell-me-about-a-time conflict, ambiguous-spec, missed-deadline, mentored-someone, hardest-bug.
- AI-specific behavioral: time you chose not to use ML; time a model failed in prod; how you debugged hallucinations; how you communicated model risk.
- Working with PMs, research scientists, infra teams.
- The hiring-bar dimension: how interviewers map answers to scope/impact/ownership signals.

### T12 — The Study Plan
- Default 8-week plan (the meat of the artifact for new grads and switchers).
- 4-week plan (experienced engineer with prior ML exposure).
- 1-week cram (interview imminent).
- A daily template and a weekly review checkpoint.
- A self-assessment quiz at the start of each topic — "are you ready, or do you need a deeper pass?"

---

## 2. The Q&A entry schema

Every question lives in a `prep/<topic>/<slug>.md` file, OR is one entry inside a bundled topic file. Either way it conforms to this schema:

```markdown
### Q: <the question, phrased exactly as it would be asked in an interview>

**Category:** concept | derivation | system-design | coding | behavioral
**Difficulty:** intro | mid | senior | staff
**Tags:** [transformers, attention, kv-cache]

**Short answer.** <1–3 sentences that would be a passable interview opener.>

**Expansion / why this is the answer.**
<The structured walkthrough. Math, contrasts, the *because*. Use bullet sub-structure freely.
This is the part that makes the difference between memorizing a bullet and being able to defend it.>

**Common follow-ups.**
- <follow-up the interviewer is likely to ask next>
- <another>

**Common mistakes.**
- <candidate mistake that signals shallow understanding>
- <another>

**References.**
- [Title of source 1](https://url1) — what it supports.
- [Title of source 2](https://url2) — what it supports.
```

### Field rules

- `Short answer` is what gets exported to the future Anki flashcard front side. Keep it tight.
- `Expansion` is what gets exported to the Anki back side, possibly truncated.
- `References` is **mandatory** for category ∈ {concept, derivation, system-design}. The validator rejects entries lacking it for those categories.
- `Coding` entries replace `References` with an `Implementation` block containing a working code snippet.
- `Behavioral` entries replace `References` with a `Signal` block describing what the interviewer is listening for.

### Why the schema looks like this

It is purposely close to the *shape of a real interview exchange*: the interviewer asks, you give a tight opener, they nod or probe, you expand, they follow up. The schema trains the candidate in that rhythm.

---

## 3. The study plan structure

### Default — 8-week plan

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | T1 ML & DL Fundamentals | Self-quiz score ≥ 80% on T1; 1 mock |
| 2 | T2 Transformers & LLM Internals | Implement attention from scratch (T9.A1); write KV-cache explanation |
| 3 | T3 Training & Fine-Tuning + T9 (coding interleave) | LoRA-vs-full-fine-tune explainer in your own words; coding drills |
| 4 | T4 Inference & Serving + T5 RAG | Build a toy RAG locally (link to `build-your-own-ai` exercise) |
| 5 | T6 Agents & Harnesses + T7 Evaluation | Design an eval harness for a small task |
| 6 | T8 ML System Design (drills D1–D5) | One drill written up end to end |
| 7 | T8 ML System Design (drills D6–D10) + T10 Paper Deep-Dive | One paper deep-dive, 30-min talk version |
| 8 | T11 Behavioral + mocks + reviews | 3 mocks; gap-fill from prior weeks |

### 4-week plan (experienced engineer)

| Week | Focus |
|------|-------|
| 1 | T1 (fast review) + T2 (deep) |
| 2 | T3 + T4 |
| 3 | T5 + T6 + T7 |
| 4 | T8 + T10 + T11 |

### 1-week cram

| Day | Focus |
|-----|-------|
| Mon | T2 internals + T9 coding |
| Tue | T3 training |
| Wed | T4 inference + T5 RAG |
| Thu | T6 agents + T7 eval |
| Fri | T8 system design (one drill) |
| Sat | Behavioral + your-own-work |
| Sun | Mock + gap fill |

### Daily template

- 60 min: read 1 topic section
- 45 min: answer 3 representative questions out loud, self-graded
- 45 min: one coding drill OR one system-design drill (alternating)
- 15 min: behavioral journaling

### Weekly review

- Re-take the topic self-quiz.
- Re-answer 2 questions you struggled with last week.
- Update your "your-own-work narrative" doc.

---

## 4. Repo architecture

```
llm-interview-prep/
  README.md                    # the rendered prep guide + study plan
  LICENSE                      # CC-BY-4.0 (content) + MIT (tooling)
  CONTRIBUTING.md
  CODE_OF_CONDUCT.md
  CHANGELOG.md

  prep/                        # structured source of truth
    01-ml-dl-fundamentals/
      README.md                # topic intro + index
      questions.md             # the Q&A entries
      references.md            # the aggregated reference list
    02-transformers-llm-internals/
    03-training-fine-tuning/
    04-inference-serving/
    05-retrieval-rag/
    06-agents-harnesses/
    07-evaluation-calibration/
    08-ml-system-design/
      README.md
      drills/
        d01-video-recommendation.md
        d02-ad-ctr.md
        d03-content-moderation.md
        d04-ecommerce-semantic-search.md
        d05-news-feed-ranking.md
        d06-rag-customer-support.md
        d07-coding-assistant.md
        d08-enterprise-agent.md
        d09-fraud-detection.md
        d10-feature-store.md
    09-coding/
      questions.md
      snippets/
        attention.py
        kmeans.py
        ...
    10-research-discussion/
    11-behavioral/
    12-study-plan/

  study-plan/
    README.md
    default-8-week.md
    experienced-4-week.md
    cram-1-week.md
    daily-template.md

  tools/
    linkcheck.py
    validate_entries.py
    build.py
    requirements.txt

  assets/
    banner.svg                 # or banner.png if rendered
    social-card.svg
    MAKE_BANNER.md             # spec, in case banner is not rendered

  docs/
    LAUNCH.md
    PROFILE_SNIPPET.md
    FLASHCARD_MODE.md          # spec for the future Anki export
    MAINTENANCE.md

  PLANNING/
    00_think.md
    01_design.md
    03_verification_log.md     # one-row-per-answer table
    06_review.md

  .github/
    workflows/
      ci.yml
      linkcheck.yml            # scheduled
    PULL_REQUEST_TEMPLATE.md
    ISSUE_TEMPLATE/
      new-question.md
      wrong-answer.md
      broken-link.md
```

### Conventions

- Topic directories: 2-digit prefix preserves order.
- Each topic has a `README.md` (the human-readable intro), a `questions.md` (the bulk content), and a `references.md` (aggregated). For T8 system design, the questions become per-drill files.
- `build.py` aggregates `prep/` into the top-level `README.md` content sections — but the top-level README is hand-curated for the hero/study-plan portion and only the *topic table of contents* is auto-generated. This keeps the README's voice human while keeping topic links in sync.

---

## 5. README wireframe

Top to bottom:

```
[Banner image — assets/banner.svg]

# llm-interview-prep

[badges: license · last updated · linkcheck status · stars]

> Everything you need to walk into an AI, LLM, or ML engineering interview prepared —
> concepts, questions, worked answers, system-design drills, and a study plan,
> curated by an active AI researcher.

## Start here

| You are... | Plan |
|------------|------|
| Prepping cold (8 weeks out) | [8-week plan](study-plan/default-8-week.md) |
| Experienced (4 weeks out)   | [4-week plan](study-plan/experienced-4-week.md) |
| Interview next week         | [1-week cram](study-plan/cram-1-week.md) |

## Topics (the spine)

1. [ML & DL Fundamentals](prep/01-ml-dl-fundamentals/)
2. [Transformers & LLM Internals](prep/02-transformers-llm-internals/)
...
12. [The Study Plan](prep/12-study-plan/)

## Scope (IN / OUT)
- IN: ...
- OUT: ... (and the peer repos that own those)

## Peer resources
- [ai-engineer-roadmap](#) — breadth of the career path.
- [harness-engineer-roadmap](#) — agent/harness depth.
- [build-your-own-ai](#) — hands-on; doubles as interview prep.

## Curator
Betty Guo (Dongxin Guo / 郭东欣) — PhD candidate, Computer Science,
University of Hong Kong; advised by Prof. Siu-Ming Yiu.
ORCID: 0009-0000-2388-1072 · GitHub: bettyguo

## How this is kept correct
Every non-trivial answer carries a verifying reference to an authoritative source.
The validator refuses to build entries that don't.

## Contributing
PR template requires answer + reference + common-mistakes.

## Star history
[embed]

## License
- Content: CC-BY-4.0
- Tooling: MIT
```

Tone calibration: quiet, expert, organized. Not breathless. Not "ultimate guide." The work speaks.

---

## 6. Tooling design

### `tools/linkcheck.py`
- Walk every markdown file under `prep/`, `study-plan/`, `docs/`, and the root `README.md`.
- Extract every `[text](url)` link with HTTP(S) scheme.
- Issue HEAD (fall back to GET) with timeout 10s, retry once, allow 200/301/302/307/308.
- Output a report: `tools/linkcheck-report.md` with successes, redirects, and failures.
- Exit non-zero on any 4xx/5xx unless URL is in `tools/linkcheck-ignore.txt`.
- Scheduled via GitHub Actions weekly.

### `tools/validate_entries.py`
- Parse every `questions.md` under `prep/*/` and every per-drill file under `prep/08-ml-system-design/drills/`.
- Each entry must have: `### Q:` heading, then in order — `**Category:**`, `**Difficulty:**`, `**Tags:**`, `**Short answer.**`, `**Expansion`, `**Common follow-ups.**`, `**Common mistakes.**`, and one of `**References.**` / `**Implementation.**` / `**Signal.**` (per category rules).
- Reject any entry of category {concept, derivation, system-design} that lacks a non-empty `References` block.
- Reject any entry of category {coding} that lacks an `Implementation` block.
- Reject duplicate `Q:` headings.
- Reject malformed reference lines.
- Output: pass/fail with per-entry detail.
- Runs in CI on every PR.

### `tools/build.py`
- Aggregates per-topic question counts into a `# Stats` block in the README.
- Regenerates the per-topic `README.md` index pages (topic intro + question list).
- Regenerates `study-plan/README.md` index.
- Idempotent.

### `tools/requirements.txt`
- `requests` (linkcheck), nothing else; keep dependency footprint trivial.

---

## 7. Contribution flow

`PULL_REQUEST_TEMPLATE.md`:
```
## What this PR changes

- [ ] New question(s)
- [ ] Fix to existing answer
- [ ] Resource / reference addition
- [ ] Tooling / CI
- [ ] Study plan / docs

## For each new/changed Q&A entry

- [ ] The answer is correct to my best knowledge.
- [ ] The entry has a `References` block with at least one authoritative source (paper / official docs / textbook) — OR is a coding/behavioral entry.
- [ ] I ran `python tools/validate_entries.py` and it passes.
- [ ] I ran `python tools/linkcheck.py` and the new links resolve.

## Reviewer guide

Did the reviewer spot-check the answer against the cited reference? [yes/no]
```

`ISSUE_TEMPLATE/wrong-answer.md`:
- Topic, question text, what the entry says, what is wrong, the authoritative reference that supports the correction.

---

## 8. Verification plan

1. **Build-time verification (Phase 3).** `validate_entries.py` runs on every commit; passes only when every concept/derivation/system-design entry carries at least one reference and the reference URL parses.
2. **Per-answer review log (Phase 3).** `PLANNING/03_verification_log.md` is a flat table: `topic | question-slug | reference-used | verified-by | verified-on`. Every answer written must add a row.
3. **Link-resolves verification.** `linkcheck.py` runs as part of CI before commit and weekly thereafter.
4. **Correctness spot-check (Phase 5).** Pick 10% of answers at random (deterministic seed); re-verify each by re-reading the cited source and re-grading correctness. Log results in the verification log. Any failure triggers full re-audit of that topic.
5. **Hostile-reviewer pass (Phase 6).** Re-read as a hiring manager looking for any answer that's wrong, shallow, or would embarrass a candidate. Fix immediately.

---

## Phase 1 sign-off

Taxonomy, schema, study plan, architecture, tooling, contribution flow, and verification plan are all locked. Proceeding to Phase 2 (scaffold + tooling + skeleton + study plan).
