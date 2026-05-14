# Launch playbook

The contested-noun thesis: AI/ML/LLM interview prep already has multiple GitHub entrants (alirezadir, khangich, Devinterview, etc.). The win condition is unambiguous comprehensiveness + correctness + maintenance. The launch must signal all three within 10 seconds of arrival.

## Timing

Interview-prep repos spike hardest **right before hiring seasons**:

- **Late Aug – Sep**: new-grad fall recruiting kicks off.
- **Jan – Feb**: post-bonus / new-budget hiring at large tech.
- **Apr – May**: spring internship + new-grad summer recruitment.

Recommend launching ~2 weeks before one of these windows so the SEO and word-of-mouth have time to compound.

Avoid: launching the week of major conference deadlines (NeurIPS, ICML) — attention is elsewhere.

---

## Show HN

**Title** (under 80 chars): `Show HN: llm-interview-prep – a comprehensive guide for AI engineering interviews`

**Body**:
```
Hi HN — I'm Betty Guo, a final-year CS PhD at the University of Hong Kong. I built this
because I noticed nothing on GitHub unified the full 2026 AI-interview stack in one
place — classical ML, transformer internals, training/fine-tuning, inference/serving,
RAG, agents, evaluation, ML system design, and behavioral — all with worked answers
and verifiable references.

What's in it:
- 12 topics × ~15 worked questions each (~147 total).
- 10 ML system design drills following a canonical 6-step structure (recsys, ad CTR,
  RAG, coding agent, fraud, feature store, etc.).
- 3 study plans: 8-week default, 4-week experienced, 1-week cram.
- Every concept/derivation/system-design answer carries an authoritative reference
  (paper, official docs, or textbook). A validator enforces it.
- Runnable code snippets for the attention / k-means / RRF / softmax coding questions.

Released under CC-BY-4.0 (content) + MIT (tooling). PRs welcome — especially questions
you were actually asked in a recent loop.

Link: https://github.com/bettyguo/llm-interview-prep
```

---

## X / Twitter thread

**Tweet 1** (lead with the study plan + system-design hook):
```
Built a comprehensive AI / LLM / ML engineering interview prep guide.

147 worked questions across 12 topics + 10 system-design drills + 3 study plans
(8-week / 4-week / 1-week cram).

Every answer has an authoritative reference. Open-source, CC-BY-4.0.

🧵
```

**Tweet 2** (the study plan):
```
The 8-week study plan is the spine.

Week 1: fundamentals
Week 2: transformers (highest-yield)
Week 3: training + coding
Week 4: inference + RAG
Week 5: agents + eval
Week 6-7: system design (10 drills)
Week 8: behavioral + mocks
```

**Tweet 3** (system design as the standalone hook):
```
The system-design section walks you through 10 worked drills:
TikTok recs, ad CTR, content moderation, e-commerce search, news feed,
RAG customer support, coding assistant, enterprise agent, fraud,
feature store.

Each follows a 6-step structure you can use on any interview.
```

**Tweet 4** (correctness contract):
```
The thing that makes this different from other prep repos: every
non-trivial answer carries an authoritative reference. The validator
refuses to build entries without one.

A wrong answer in an interview-prep repo actively harms users. Zero
unsourced answers.
```

**Tweet 5** (call to action):
```
Released under CC-BY-4.0 (content) + MIT (tooling).

PRs welcome — especially questions you were actually asked in a recent
interview loop. The community-contributed Qs are the highest-signal
additions.

https://github.com/bettyguo/llm-interview-prep
```

---

## r/MachineLearning post

**Title**: `[P] llm-interview-prep — a comprehensive 12-topic prep guide for AI/ML engineering interviews`

**Body**:
```
Hi r/MachineLearning,

I built a comprehensive prep guide for AI/ML/LLM engineering interviews because
I noticed the existing options on GitHub are either ML-classic or pure-LLM,
rarely both, and most are bullet-list summaries without worked answers.

What's in it:
- 12 topics × ~15 questions: ML/DL fundamentals → transformer internals →
  training (RLHF/DPO/GRPO/LoRA) → inference/serving → RAG → agents → eval →
  ML system design → coding → research discussion → behavioral.
- 10 ML system design drills with a canonical 6-step structure.
- Three study plans (8-week / 4-week / 1-week cram).
- Every concept/derivation/system-design answer cites an authoritative source.

Released under CC-BY-4.0. Adapt for cohorts / bootcamps / study groups freely.

Curator: Betty Guo, final-year CS PhD at HKU (advised by Prof. Siu-Ming Yiu).

Link + repo: https://github.com/bettyguo/llm-interview-prep

Happy to take feedback on what's missing or wrong.
```

---

## r/cscareerquestions / r/learnmachinelearning post

Adapt the r/ML post; lighter on the technical depth, more emphasis on the study plan and the "I have an interview next week" cram plan. r/cscareerquestions readers are typically less ML-deep but more interview-anxious — lead with the time-boxed plan rather than the question count.

---

## Newsletter outreach (one paragraph each)

**For Sebastian Raschka's Ahead of AI (Substack)**:
> Hi Sebastian — I'm Betty Guo, finishing my PhD at HKU. I built a comprehensive AI/ML/LLM
> interview prep guide that unifies the classical ML side with the 2026 LLM side. 147
> worked questions, 10 system-design drills, three study plans, every non-trivial answer
> referenced. Released CC-BY-4.0. If it fits a future issue, I'd be honored: [link]. Either
> way, would love your read.

**For Import AI (Jack Clark)**:
> Hi Jack — I'm Betty Guo (CS PhD, HKU). Released llm-interview-prep, a 147-question prep
> guide spanning ML fundamentals, transformer internals, training (RLHF/DPO/GRPO),
> inference/serving (KV cache math, paged attention, etc.), RAG, agents, evaluation, and
> ML system design. Every non-trivial answer cites an authoritative source. CC-BY-4.0.
> [link].

**For AlphaSignal**:
> A new comprehensive AI/LLM/ML interview prep guide: 147 worked questions, 10 system-
> design drills, three time-boxed study plans, every non-trivial answer referenced.
> CC-BY-4.0. [link].

---

## What signals "trust this" in 10 seconds

Top of README:
- Strong banner.
- Three badges (license, CI, last-updated).
- One-sentence value proposition.
- Three study-plan picks ("Prepping cold / Experienced / Interview next week").
- Topic index.

What kills trust:
- Missing references.
- A single-sentence answer for what should be a paragraph.
- Bullet-list-only content.
- Stale "last updated" date.

The phase-5 hardening sweep below ensures none of these are present at launch.

---

## Hand-checklist before posting

- [ ] Verified all CI badges resolve.
- [ ] Verified the star-history embed renders.
- [ ] Verified every link in the README resolves.
- [ ] Tested a fresh-eyes read: does a stranger understand the value within 10 seconds?
- [ ] Confirmed the curator attribution is correct (Betty Guo, HKU, Prof. Yiu).
- [ ] First PR opened from a real second user (asking a colleague to add one question) so the contribution path is proven.
- [ ] Have a 3-paper-ready "what's a paper you read recently" answer for any inbound interviewer (since the repo is *about* interviews, it will attract them).

---

## After launch: the 30-day window

Most stars come in the first 30 days. Maintenance signal in that window matters:
- Respond to every issue within 48 hours.
- Merge or close every PR within 7 days.
- Post 1–2 follow-up content updates (new topics, new questions) on day 7 and day 21.
- Pin a "wanted questions" issue with the high-value gaps.

If the project survives the 30 days, the maintenance cadence (quarterly content review per `docs/MAINTENANCE.md`) takes over.
