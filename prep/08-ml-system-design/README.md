# 08 — ML System Design

The integrating discipline. The 45–60-minute interview where the candidate is asked to design an ML/AI system end to end. This is the section that, for a senior or staff candidate, often decides the loop.

## The canonical 6-step structure

Use this as your skeleton for every drill, every interview. Interviewers grade on coverage and reasoning, not memorization.

1. **Clarify requirements.**
   - Functional: what is the system supposed to do? Who uses it? How often?
   - Non-functional: latency budget, throughput, freshness, cost, privacy, compliance.
   - Ask 3–5 clarifying questions. Do not assume.

2. **Define metrics.**
   - Online (north-star + guardrails): the business metric the system moves, plus the things that should not regress.
   - Offline: the proxy metric you optimize during training/evaluation, with the assumed correlation to online.
   - State the gap between online and offline explicitly — interviewers love this.

3. **Data and labels.**
   - Sources, volume, labeling strategy (implicit vs. explicit, human vs. weak supervision), labeling cost, label noise.
   - Feature store; freshness; backfilling.
   - Bias / fairness considerations where applicable.

4. **Modeling.**
   - Candidate generation → ranking (two-stage where appropriate).
   - Architecture sketch, feature design, training data window, training procedure.
   - Cold-start handling.
   - For LLM-app drills: model size choice, fine-tuning vs. prompting vs. RAG, eval shape.

5. **Serving.**
   - Latency and scale targets reconciled with model choice.
   - Caching, prefilter, batch vs. real-time inference.
   - Feature retrieval at serve time, online/offline feature parity.

6. **Monitoring + iteration.**
   - Drift detection (input drift, label drift, concept drift).
   - Retraining cadence and trigger.
   - A/B harness, holdout, gated rollout.

## How to use the drills

Each drill walks through the 6-step structure for a specific system. Read one drill end to end to learn the rhythm; then do a fresh one timed at 45 minutes with no peeking. After 3–4 timed drills, the structure becomes muscle memory.

## The 10 drills

| # | Drill | Era |
|---|-------|-----|
| 1 | [Video recommendation (TikTok/YouTube)](drills/d01-video-recommendation.md) | classic |
| 2 | [Ad click-through-rate prediction](drills/d02-ad-ctr.md) | classic |
| 3 | [Multi-modal content moderation pipeline](drills/d03-content-moderation.md) | classic + modern |
| 4 | [E-commerce semantic search](drills/d04-ecommerce-semantic-search.md) | modern |
| 5 | [Personalized news feed ranking](drills/d05-news-feed-ranking.md) | classic |
| 6 | [LLM-powered customer-support assistant (RAG)](drills/d06-rag-customer-support.md) | **AI-era** |
| 7 | [AI coding-assistant / autocomplete](drills/d07-coding-assistant.md) | **AI-era** |
| 8 | [Multi-turn LLM agent for enterprise workflow](drills/d08-enterprise-agent.md) | **AI-era** |
| 9 | [Spam / fraud detection](drills/d09-fraud-detection.md) | classic |
| 10 | [ML feature store design](drills/d10-feature-store.md) | platform |

## References

Each drill carries its own references. The canonical readings that influenced this section are aggregated in [`references.md`](references.md).
