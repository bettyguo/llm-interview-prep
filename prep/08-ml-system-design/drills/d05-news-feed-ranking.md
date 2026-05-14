# D5 — Personalized news feed ranking

A worked drill following the [6-step structure](../README.md#the-canonical-6-step-structure).

---

### Q: Design a personalized news feed (Facebook / Twitter / LinkedIn style).

**Category:** system-design
**Difficulty:** senior
**Tags:** [feed-ranking, multi-objective, freshness]

**Short answer.** Pull candidate posts from friends/follows/recommended sources → score each with a multi-task model predicting `P(read, like, comment, share, hide)` → combine into a single utility weighted by business objectives → diversify and inject. Differs from search/recommendations in that freshness is paramount (posts decay in hours) and the *candidate pool* is user-specific (friends' posts).

**Expansion / why this is the answer.**

**1. Clarify requirements.**
Functional:
- Show a paged feed of posts per user.
- Mix of friends / follows + recommended + ads.
- Personalized; novel; not repetitive.

Non-functional:
- Latency: p99 < 250 ms for the next-page request.
- Catalog per user: typically hundreds to thousands of candidate posts (own network), millions if including recommended.
- Freshness: posts surface within minutes of being published.

Clarifying Qs:
- "Friends-only (Facebook) or follow-based (Twitter/IG)?"
- "What's the post-decay shape — minutes or hours?"
- "Do we show ads in-feed?"

**2. Define metrics.**

Online:
- **North-star**: meaningful interactions (likes + comments + shares + dwell time) — combined to avoid Facebook's historic "engagement-via-outrage" pathology.
- **Guardrails**: time-spent (engagement vs. addiction), reported posts, hide rate, unfollow rate, content-quality survey.
- **Long-term**: 7-day and 30-day retention.

Offline:
- AUC on each prediction head.
- GAUC (per-user AUC).
- Calibration.

Online ↔ offline: Facebook famously documented the gap (their internal research on meaningful-interactions metric trade-offs).

**3. Data and labels.**

Sources:
- Impression logs per post per user.
- Engagement logs: like, comment, share, hide, report, dwell time.
- Post features: author, content (text, image, video), hashtags, topics, embeddings.
- User features: history, network, demographic, recent activity.

Labels:
- Multi-head: read (dwell > threshold), like, comment, share, hide, report.
- Negative: skipped, scrolled-past quickly.

Class imbalance: heavy. Most posts in feed are not engaged with.

Decontamination: time-respecting splits; no future features.

**4. Modeling.**

**4a. Candidate generation**:
- For each user, source posts from:
  - Friends / follows in the last N days.
  - Recommended sources (followed-by-similar-users; engaged-with-similar-content).
  - Trending in user's network.
- Total candidates: O(1k–10k) per user.

**4b. Scoring**:
- Multi-task DNN. Inputs: user features, post features, cross features (user-author affinity, user-topic affinity), context.
- Outputs: `P(like), P(comment), P(share), P(hide), P(dwell > T)`.
- Loss: per-head BCE, weighted.

**4c. Combining into a feed score**:
- `feed_score = Σ w_i · P_i(positive) − Σ v_j · P_j(negative)`.
- Weights tuned for the north-star metric. Hide and report are heavy negatives.
- Calibrated probabilities matter because the weighting is direct.

**4d. Re-rank / freshness / diversity**:
- Time-decay: `score · exp(-Δt / τ)` with τ in hours.
- Diversity: penalize many posts from the same author / topic.
- Ad and recommended-content insertion.

**5. Serving.**

Latency: p99 < 250 ms.
- Candidate generation: pull from friend-graph + recommended pool (cached per user).
- Scoring: forward pass on ~1k candidates.
- Ranker on GPU; batched.
- Feature store with online + offline features.

Caching:
- Per-user feed cached on session start; refreshed on new-post arrival or on user pull.

**6. Monitoring + iteration.**

Drift:
- Activity patterns by time-of-day, day-of-week, seasonal.
- New content categories (memes, formats).

Retraining:
- Daily ranker.
- Periodic full retrain when distribution shifts.

Online evaluation:
- A/B with bucketed users.
- Long-running holdouts (weeks) to track creep.

Pitfalls:
- **Engagement-bait**: outraged users engage more; the model can learn to surface outrage. Mitigate by including survey-based quality signals in the loss.
- **Filter bubbles**: heavy personalization narrows the user's content. Inject exploration / diverse content.
- **Misinformation**: amplification risk. Separate classifier + downrank pipeline.

---

**Common follow-ups.**

- "How would you reduce engagement-bait?" → Add quality-survey signals; downrank "engagement-bait classifier" output; include comments-from-friends signal (genuine engagement).
- "What if users hate the feed but engage with it (addiction model)?" → A real tension; treat session length and self-reported value separately; let "meaningful interactions" win.
- "How do you A/B test feed changes?" → Bucketed users; long horizons; holdout for novelty effect.

**Common mistakes.**

- Single-objective optimization (one metric, then a Goodhart's-Law story).
- No freshness handling.
- Skipping the dis-engagement signals (hide, unfollow, report).

**References.**

- [Eksombatchai et al. — "Pixie: A System for Recommending 3+ Billion Items"](https://arxiv.org/abs/1711.07601) — Pinterest's feed-recommendation backbone.
- [Facebook AI — News Feed and meaningful interactions](https://ai.meta.com/) — official engineering blog posts on feed ranking.
- [Covington et al. — YouTube two-tower paper](https://research.google/pubs/pub45530/) — analogous two-stage funnel.
