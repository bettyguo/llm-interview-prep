# D1 — Video recommendation (TikTok / YouTube)

A worked drill following the [6-step structure](../README.md#the-canonical-6-step-structure).

---

### Q: Design a video recommendation system for TikTok / YouTube Shorts.

**Category:** system-design
**Difficulty:** senior
**Tags:** [recsys, two-tower, ranking, candidate-generation]

**Short answer.** Two-stage funnel: **candidate generation** retrieves O(1k) candidates from a 100M+ video corpus via embeddings (two-tower model) and heuristics (recent, trending, friends-watched); **ranking** uses a richer deep model with per-impression features to score the candidates and pick the top N to show. Optimize a multi-objective utility (watch time, retention, like, share, complete-watch) with calibrated probabilities. Serve with a feature store + ANN index + low-latency model server, well under 100 ms.

**Expansion / why this is the answer.**

**1. Clarify requirements.**
Functional:
- Surface: home feed (For You). Single-stream or paged.
- Personalized to user; cold-start handling.
- Diverse content; not the same creator repeatedly.
- Real-time-ish: new videos should be discoverable within minutes.

Non-functional:
- Latency: p99 < 200 ms (server-side); p50 < 100 ms.
- Throughput: 100k+ QPS at peak.
- Catalog: 100M–1B videos; growing.
- User base: 100M+ DAU.
- Cost: serving cost must be a tiny fraction of revenue per session.
- Cold-start: brand-new users; brand-new videos.

Clarifying Qs to ask the interviewer:
- "How many videos to return per request — single or feed of 10?"
- "What's the freshness requirement on new content?"
- "Is creator-quality / safety part of the same ranker, or a separate pass?"
- "Do we have explicit signals (likes, shares) or only implicit (watch time)?"

**2. Define metrics.**

Online (north-star + guardrails):
- **North-star**: total watch-time per user (TikTok); session length (YouTube Shorts).
- **Guardrails**: 1-day / 7-day retention, complete-watch rate, like rate, creator-diversity Gini, % new-creator surfaces.
- **Watch-out**: optimizing watch-time alone can drive recommendation toward outrage / clickbait. Add quality/safety guardrails (community-violation rate, low-quality-creator share).

Offline:
- **Candidate-generation recall@k** (did the ground-truth watched video appear in the top-k retrieved?).
- **Ranker** AUC / GAUC (grouped AUC per user) on a labeled `(user, item, watched_yes/no)` set.
- **Calibration**: ECE on predicted P(watch) vs. observed rate — calibration matters when you combine multiple objectives.

Online ↔ offline gap: AUC improvements don't always translate to watch-time gains — many wins are noise, hence the A/B requirement.

**3. Data and labels.**

Sources:
- **Implicit watch logs**: every impression + watch-duration + complete-rate per user × video. Trillions of events / month.
- **Explicit signals**: likes, comments, shares, follows.
- **Negative signals**: skip < 2s (strong negative), report, block.
- **User features**: demographics (where available), device, locale, language, history (last-N videos, last-N creators), inferred-interests.
- **Item features**: creator, hashtags, music/track, duration, language, embeddings (vision + audio + text).
- **Context features**: time-of-day, day-of-week, session position, device.

Labels:
- **Watch-time-weighted positive**: not just "did they watch?" but how long, relative to video length.
- **Complete-watch**: binary.
- **Skip < 2s**: hard negative.
- **Like / share / follow**: weaker positive, but high-value.

Data freshness:
- Candidate-generation embeddings: refreshed nightly (or every few hours).
- Ranker: trained on the last 7–30 days of impression data; refreshed daily.
- Online learning for trending-content boosts.

Decontamination:
- Don't include the user's own uploads in retrievals.
- Don't re-show recently-shown videos for some window.

**4. Modeling.**

Two-stage funnel:

**4a. Candidate generation (retrieval)**
- **Two-tower model**: user tower (embeds user features + recent history) and item tower (embeds video features + content embeddings); train so `cos(user, watched_item)` > `cos(user, skipped_item)`.
- **Loss**: in-batch sampled softmax (Covington, Adomavicius, Sargin 2016, "Deep Neural Networks for YouTube Recommendations" — variant); or contrastive.
- **Output**: a user vector per request; ANN search against the precomputed item index.
- **Index**: HNSW or IVF over hundreds of millions of items.
- **Additional candidate sources** (don't just rely on dense retrieval):
  - Trending in user's locale.
  - Recently-followed creators' latest videos.
  - Co-watched (collaborative-filtering signal).
  - Random exploration (5–10% slot for diversity).
  - Friend / social-graph videos.
- **Output**: union of all sources → ~1k candidates.

**4b. Ranking**
- **Model class**: deep neural net — typically a wide-and-deep / DLRM-style architecture (Naumov et al. 2019), or a transformer-over-history with cross-features.
- **Inputs**: full user features, full item features, cross features (user-creator affinity, user-hashtag affinity), context.
- **Output**: multi-head — `P(watch ≥ 50%)`, `P(complete)`, `P(like)`, `P(share)`, etc.
- **Loss**: multi-task; weighted sum of per-head losses; per-head label distribution and weighting.
- **Combining heads**: `score = Σ w_i · P_i` with weights tuned for the north-star.
- **Cold-start**:
  - New user: rely heavily on locale / demographic priors + exploration.
  - New video: use content embeddings only until enough engagement data accrues; consider exploration boosts ("creator bandit").

**4c. Re-ranking / business rules**
- Diversity (don't surface 5 of the same creator).
- Safety filters (apply *after* ranking).
- Promoted content slots.
- Cooldowns (don't re-show recent).

**5. Serving.**

Latency budget: p99 < 200 ms.
- **User features**: read from a feature store (sub-ms).
- **Candidate generation**: ANN search (~10 ms for HNSW at 100M scale).
- **Ranker forward pass**: batched over 1k candidates per request → ~30 ms with a small ranker.
- **Re-rank + business rules**: ~5 ms.
- **Feature consistency**: same code path computes features at train and serve time (Tecton/Feast-style feature platform).

Infrastructure:
- ANN service (Faiss/Vespa/Vertex MS).
- Feature store (Tecton, Feast, internal).
- Online inference (TF Serving, Triton, KServe).
- Cold-start cache (per-locale top videos).

Scale design:
- Per-user fan-out across many candidate-generators in parallel; merge.
- Sharded ANN index.

**6. Monitoring + iteration.**

Drift:
- Input-feature drift: distribution shift over time; monitor and retrain.
- Label drift: positive-rate changes with platform growth.
- Concept drift: what users want changes.

Retraining cadence:
- Ranker: daily; same-day promotion is risky — gated rollout.
- Embeddings: weekly retraining; daily fine-tune.

Online evaluation:
- A/B on every change.
- Hold-out group for long-term effects.
- Watch for guardrail regressions (the model that gets +5% watch-time but drops creator diversity by 20% is not a win).

Feedback loop concern:
- The model trains on what it shows; what it shows depends on the model.
- Mitigations: exploration slots, randomized holdout, off-policy correction (Chen et al. 2019).

---

**Common follow-ups.**

- "How do you handle the cold-start problem?" → Content-based features for new videos; demographic / locale priors for new users; exploration slots.
- "What if a creator games engagement?" → Quality / abuse-detection model; lower-rank or remove low-quality content; surfaced as a separate moderation pipeline.
- "How do you balance multi-objective trade-offs?" → Weighted sum, with weights from offline experiments validating against the north-star online metric.
- "What about feedback loops?" → Off-policy correction (Chen et al. 2019), exploration, A/B holdouts.

**Common mistakes.**

- Pitching one giant model instead of the two-stage funnel.
- No diversity / fairness consideration.
- Skipping cold-start.
- No monitoring / retraining cadence story.
- Optimizing watch-time without guardrails.

**References.**

- [Covington, Adomavicius, Sargin — "Deep Neural Networks for YouTube Recommendations"](https://research.google/pubs/pub45530/) — the canonical two-stage funnel paper.
- [Naumov et al. — "Deep Learning Recommendation Model for Personalization and Recommendation Systems" (DLRM)](https://arxiv.org/abs/1906.00091) — modern ranker architecture.
- [Chen et al. — "Top-K Off-Policy Correction for a REINFORCE Recommender System"](https://arxiv.org/abs/1812.02353) — feedback-loop correction.
- [Cheng et al. — "Wide & Deep Learning for Recommender Systems"](https://arxiv.org/abs/1606.07792) — wide-and-deep.
