# Phase 0 — THINK

> Status: complete. Author: Betty Guo (Dongxin Guo / 郭东欣), PhD candidate, HKU CS, advised by Prof. Siu-Ming Yiu.
> Date authored: 2026-05-14.

This document validates the noun (`llm-interview-prep`), maps the competitive landscape with brutal honesty, defines the win condition for a contested category, and locks down the answer-correctness protocol that is this repo's single most important quality gate.

---

## 1. Noun validation — and the competition map

**Noun:** `llm-interview-prep`. The chosen GitHub-org/repo identity slot is `bettyguo/llm-interview-prep`.

**Verdict on the noun:** It is *valid but contested*. The category "AI/LLM/ML interview prep" is one of the highest-conversion-to-star niches on GitHub — interview anxiety is acute, the stakes are a job, and proven analogs (`coding-interview-university` ~346k stars, `tech-interview-handbook` ~139k, `system-design-primer` ~310k+) demonstrate that this niche reliably mints all-time-top-tier knowledge artifacts. But the AI sub-niche specifically has *several* existing entrants — so we cannot ship a thin entry and win by being early. The win condition is being unambiguously *the most comprehensive, the most correct, and the most maintained*.

### Competitive census (active GitHub players, as of 2026-05)

| Repo | Owner | Approx. positioning | Where it falls short |
|------|-------|---------------------|----------------------|
| `Machine-Learning-Interviews` | `alirezadir` | FAANG ML interview guide; well-respected | Older skew (classic ML/DL); LLM/agent/RAG depth is thin; few worked answers carry references; not actively versioned for 2026 LLM-era content |
| `machine-learning-interview` | `khangich` | Real FAANG questions, MVP study plan; commercially monetized course | Course-funnel-shaped; many sections are pointers not worked answers; modern LLM content sparse |
| `machine-learning-systems-design` | `chiphuyen` (Chip Huyen) | 27 open-ended MLSD questions; high authority | Intentionally a *prompt list*, not worked answers; pre-LLM-era framing for most of it |
| `MLQuestions` | `andrewekhalel` | ML/CV Q&A bullets | Bullet-style, terse, often no references; CV-skewed; little LLM coverage |
| `llms-interview-questions` | `Devinterview-io` | LLM Q&A | Vendor-marketing-shaped (Devinterview funnel); answers are shallow; not curated by a named expert |
| `LLM-Interview-Questions-and-Answers-Hub` | `KalyanKS-NLP` | 100+ LLM Qs | Bullet-list, no worked-answer depth, no system-design or coding interleave |
| `LLMInterviewQuestions` | `llmgenai` | LLM Qs from top companies | Lightly-maintained; uneven answer quality |
| `ai-engineering-interview-questions` | `amitshekhariitbhu` | AI engineering cheatsheet | Cheatsheet-shape, lacks worked depth |
| `Data-Science-Interview-Questions-Answers` | `youssefHosni` | DS interview Qs | DS-centric, not ML-engineering-centric, LLM section thin |
| `CrackingMachineLearningInterview` | `shafaypro` | ML interview roadmap | Listicle-shape; staleness on the LLM/agent/RAG side |

### Where the gap actually is

Five gaps no single existing repo fills together:

1. **Classic-to-modern unification.** Almost every existing repo is either *classic ML/DL* (alirezadir, khangich, chiphuyen) or *pure LLM* (Devinterview, KalyanKS, llmgenai). Almost none seriously covers the full stack a 2026 AI-engineer interview actually tests: classical ML fundamentals → DL fundamentals → transformer internals → training/fine-tuning → inference/serving → retrieval/RAG → agents/harnesses → evaluation → ML system design → coding → behavioral. Real loops sample all of these in a single day.
2. **Worked answers, not bullet lists.** The dominant pattern is "question → 2-line bullet." Interviews demand the *expansion*: why, how to discuss it, common follow-ups, common candidate mistakes. No top-10 repo does this consistently.
3. **References on every answer.** Almost no existing repo enforces a "every answer has a verifying authoritative reference" rule. This is the difference between "trustable in an interview" and "I memorized something wrong from a random README."
4. **Modern 2026 topics.** MoE architectures (Mixtral-style routing, DeepSeek-V3-style fine-grained experts), KV-cache tricks, speculative decoding, grouped-query / multi-query attention, FlashAttention variants, paged attention, GRPO/RLHF/DPO/KTO variants, agentic harness design, eval frameworks beyond MMLU — patchy at best across competitors.
5. **Maintained.** Multiple top-10 repos have not had a substantive content commit in 6+ months. Interview content drifts quickly; a stale repo loses authority within a hiring cycle.

### Strategic conclusion

The noun is contested, but the *standard of execution* in the contest is uneven. A repo that (a) unifies the full stack end-to-end, (b) writes genuine worked answers, (c) carries a verifying reference on every answer, (d) covers 2026 LLM/agent/MoE/RAG topics decisively, and (e) commits to a visible maintenance cadence wins this niche. None of the existing entrants do all five.

**This is what this repo will be.** A thin launch loses here — therefore Phase 2/3 must ship the full skeleton populated end-to-end on day one, not a scaffold-with-TODOs.

---

## 2. Audience and anxiety map

Four primary readers, all with maximal anxiety:

1. **The new grad targeting an AI/ML role.** Knows fundamentals from coursework, anxious about FAANG-style interviewing, doesn't know what's actually asked. Wants a structured plan and a list of questions they can practice against. Highest conversion-to-star (book-marks the resource).
2. **The SWE switching into AI/ML/LLM engineering.** Strong coding background, weak on transformer internals, RAG, fine-tuning. Has a 4–8-week prep window before applying. Wants the topical depth they don't have, in a study-plan shape.
3. **The ML researcher/PhD moving to industry.** Strong on ML/DL theory, weak on ML system design, production serving, evaluation in industry. Wants the system-design and the production-LLM sections specifically.
4. **The engineer with a specific interview loop next week.** Skims a fixed checklist of "the 50 questions I might get asked." Wants a study plan timeboxed at 1 week, plus a high-recall question bank.

The shared psychological state across all four is the same: *acute anxiety + a job at stake.* That is precisely the profile that drives starring (as a bookmark / safety blanket) and word-of-mouth ("this is the resource I used"). The audience does the marketing if the resource is genuinely trustable.

**Anti-pattern to avoid:** general-purpose "AI roadmap" framing. That is owned by the peer repo `ai-engineer-roadmap`. This repo is laser-focused on the interview gate at the end of that road.

---

## 3. Competitive structural study — the recipe

From dissecting `coding-interview-university` (~346k), `tech-interview-handbook` (~139k), `system-design-primer` (~310k+), and `JavaGuide`:

- **Title is the noun.** Not a brand, not a clever name. The repo title is exactly what someone is searching for ("system design primer," "coding interview university"). We do the same: `llm-interview-prep`.
- **Hero-prominent study plan.** All three top repos foreground a structured, time-boxed study plan in the first screen of the README. This is what users star *for*.
- **The README is the table of contents** — every topic listed, deep links into subdirectories. The reader can see the scope at a glance.
- **Trust signals up top.** Last-updated badge, license badge, contributor count, star history. `system-design-primer` adds Anki flashcards (a *delivery* feature, not just a content feature) — we should design toward this.
- **Comprehensive on day one.** None of the top three launched as a stub. The skeleton was always complete; iteration added depth.
- **Persona-based variant plans.** `coding-interview-university` offers short and long plans by background; this is a star-driver because it personalizes the artifact.
- **Self-contained.** Top repos minimize outbound link dependencies for the *answer itself* — outbound links are for *further reading*, not "click here to learn the answer." We honor this.

---

## 4. Scope boundary

### IN scope

- Classical ML fundamentals (bias-variance, regularization, classical algorithms, evaluation metrics) — at depth sufficient to answer FAANG-style Qs.
- Deep learning fundamentals (optimization, regularization, architectures, training dynamics).
- Transformer & LLM internals (attention variants, positional encodings, normalization, MoE, scaling laws).
- Training & fine-tuning (pretraining, instruction tuning, RLHF/DPO/GRPO, LoRA/QLoRA, distillation).
- Inference & serving (KV cache, paged attention, speculative decoding, quantization, batching, throughput vs. latency).
- Retrieval & RAG (embedding models, chunking, hybrid retrieval, reranking, eval).
- Agents & harnesses (tool use, planning, ReAct, multi-step reliability, eval).
- Evaluation & calibration (LLM-as-judge, holdouts, leakage, hallucination measurement).
- ML system design (interview-style drills for recsys, search, ads, content moderation, LLM-app system design).
- ML/AI coding questions (NumPy/PyTorch-style implementation challenges relevant to AI roles).
- Research discussion / paper deep-dives (how to talk about your own work, reading & critiquing a paper live).
- Behavioral & communication for AI roles.
- A multi-week study plan with persona variants.

### OUT of scope

- A general ML/DL course (that's `ai-engineer-roadmap`'s territory; we link to it).
- General SWE/algorithms interview prep (owned by `coding-interview-university`/`tech-interview-handbook`; we cross-link).
- General system design beyond ML (owned by `system-design-primer`; we cross-link).
- Vendor-specific certifications (AWS ML Specialty, etc.).
- A "how to land an AI job" career-coaching narrative (we are interview prep, not career coaching).
- Building real AI systems (that's `build-your-own-ai`; we link sideways).

---

## 5. Answer-correctness protocol — THE quality gate

This is the single most important section in the entire planning doc. In a category where the entire value proposition is *trust*, a wrong answer is the worst possible failure mode — worse than a broken link, worse than a typo, worse than thin coverage. A wrong answer under a real-name academic identity (HKU PhD, advised by Prof. Yiu) is a reputational injury.

### The protocol

Every non-trivial Q&A entry must:

1. **Have a `reference` field** — at minimum one authoritative source that supports the claim. The validator (`tools/validate_entries.py`) refuses to build any entry that lacks one.
2. **Cite from the authoritative class.** Authoritative = (a) the original paper, (b) primary documentation from the framework owner (PyTorch, HuggingFace official docs, OpenAI/Anthropic docs), (c) a widely-cited textbook (Murphy, Goodfellow, Bishop, Speech & Language Processing), or (d) a peer-reviewed survey. *Blog posts are acceptable as supplementary references, not as sole authority on a contested fact.*
3. **State limits.** Where a claim is contested or version-dependent (e.g., "FlashAttention is X× faster" — depends on hardware, version, sequence length), the answer must say so. Better to say "depends, here's the tradeoff" than to over-claim.
4. **Be verified at write time.** The Phase 3 verification log (`PLANNING/03_verification_log.md`) tracks every answer against its reference. Spot-check sample: a random 10% of answers re-verified in Phase 5.

### Editorial standards

- No fabricated benchmarks, no hallucinated paper titles, no invented hyperparameter defaults. If unsure, look it up; if can't verify, do not publish.
- Distinguish "consensus" from "popular but contested." E.g., scaling laws have a Chinchilla revision that updated the original Kaplan recipe — say so.
- Distinguish "true on small scale" from "true at frontier scale." Many interview-meme facts are demo-scale folklore.
- For algorithmic / mathematical claims (loss formulas, gradient derivations, complexity bounds), include the form, not just words.

### What "trivial" means (and what's exempted from references)

- Definitions of universally-known terms (e.g., "overfitting is when the model fits noise") do not require a paper citation.
- Code-pattern questions (NumPy/PyTorch implementations) require a *correct working implementation*, not a citation — but the implementation must be tested mentally / on paper for correctness.
- Behavioral / soft questions ("how do you handle disagreement with a senior?") are subjective and exempt from the reference requirement.

The validator will treat entries with `category: behavioral` or `category: coding` as exempt from the reference field but require an `implementation` field for coding.

---

## 6. Constellation map

This repo sits as **one of four standalone peers**, not a child of any:

```
+--------------------------+    +--------------------------+
| ai-engineer-roadmap      |    | harness-engineer-roadmap |
| (breadth of the path)    |<-->| (agent/harness depth)    |
+--------------------------+    +--------------------------+
            ^                                ^
            |                                |
            v                                v
+--------------------------+    +--------------------------+
| llm-interview-prep       |<-->| build-your-own-ai        |
| (the gate at the end)    |    | (hands-on, doubles as    |
| **THIS REPO**            |    |  interview prep)         |
+--------------------------+    +--------------------------+
```

- Reader arriving here: "I have an AI/ML/LLM interview coming up." Stays here.
- Cross-links sideways (in a `Peer resources` section in the README, not embedded mid-flow):
  - `ai-engineer-roadmap` — "if you want the full career-path scope, see this peer."
  - `harness-engineer-roadmap` — "if you specifically need agent/harness interview depth, see this peer."
  - `build-your-own-ai` — "for hands-on practice that doubles as interview prep, see this peer."

Each peer repo is described as a standalone equal. No nesting language. No "official sister repo." Just "peer resources, owned separately, recommended."

---

## 7. Risk log

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | Contested-noun risk: existing players already have stars and inbound traffic | High | Decisive comprehensiveness (full topic stack), genuine worked answers with references, and visible maintenance cadence. Launch big, not stub. |
| 2 | **Wrong-answer reputational risk** | **Critical** | Phase 0's answer-correctness protocol. `validate_entries.py` enforces references. Verification log. Spot-check in Phase 5. **A wrong answer under HKU/Prof. Yiu attribution is unacceptable.** |
| 3 | Thin-launch risk: stub README + TODOs in every section | High | Phase 2 ships complete skeleton; Phase 3 ships content in *every* topic before checkpoint. No empty topic on launch day. |
| 4 | Staleness risk: LLM topics drift fast (new model families, new techniques every quarter) | Medium | Commit to a maintenance cadence (stated in README), changelog, "last updated" badge, GitHub Actions on a schedule to run linkcheck. Quarterly content review reminder. |
| 5 | Over-scoping risk: turning into a general ML course | Medium | Strict adherence to Phase 0 IN/OUT. Every entry must serve interview prep specifically — has a representative question shape. |
| 6 | Tone risk: appearing thirsty / SEO-blogspam | Medium | No clickbait. No "Top 10 Tricks." Question-and-worked-answer format throughout. Quiet, expert-curator tone. |
| 7 | Length/maintainability risk: a single 10,000-line README becomes unmaintainable | Medium | Source of truth lives in `prep/` with structured per-topic files; README is the rendered index. Build script regenerates. |
| 8 | Dependency-on-external-resources risk: outbound links become primary product | Low | Self-contained answers. Outbound links are *further reading*, never *the answer itself*. |

---

## 8. Open questions (proceed using best judgment; log assumptions)

1. **Anki / flashcard delivery on day one?** `system-design-primer` ships Anki. I will ship the *source schema* in Phase 1 that makes Anki export feasible, but the Anki bundle itself will be a Phase 5 spec (`docs/FLASHCARD_MODE.md`) rather than launch-day content. **Assumption: defer Anki bundle to post-launch.**
2. **Coding questions in `.py` files or markdown?** Both. `prep/coding/` has answer expansions in markdown that include the code block; a parallel `prep/coding/snippets/` holds runnable `.py` files for testing. **Assumption: dual-format.**
3. **How many questions per topic?** No hard count; depth over count. Target: enough that a candidate with a one-week prep window finishes feeling exam-ready in that topic. **Assumption: 15–40 questions per topic, depending on topic breadth.**
4. **Persona variants of the study plan?** Yes — default 8-week plan, plus 4-week "experienced engineer switching to ML" plan, plus 1-week "interview next week" cram plan. **Assumption: three plans.**
5. **External-paper PDFs hosted in-repo?** No — link out only. Avoids licensing problems and keeps the repo small. **Assumption: link-out-only.**
6. **Banner image generated by external tool or specified in `MAKE_BANNER.md`?** Specified in `assets/MAKE_BANNER.md` for a designer to render. **Assumption: spec only.**
7. **Reference style — full URLs in entries, or footnote-numbered?** Full URLs inline in entries, plus a `references.md` per topic that aggregates them. Easier for readers, easier for linkcheck. **Assumption: dual.**

---

## Phase 0 sign-off

The noun is valid, contested, and winnable by execution quality. The answer-correctness protocol is the single non-negotiable gate. The repo will launch comprehensive end-to-end. Proceeding to Phase 1 (Design).
