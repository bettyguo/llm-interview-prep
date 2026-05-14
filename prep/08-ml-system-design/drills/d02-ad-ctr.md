# D2 — Ad click-through-rate prediction

A worked drill following the [6-step structure](../README.md#the-canonical-6-step-structure).

---

### Q: Design a CTR-prediction system for an ad auction.

**Category:** system-design
**Difficulty:** senior
**Tags:** [ctr, ad-ranking, calibration, online-learning]

**Short answer.** Predict `P(click | user, ad, context)` for each candidate ad. Use a deep CTR model (Wide & Deep, DeepFM, or DCN-V2) over high-cardinality categorical features (with embeddings) + dense features + cross features. Calibrated probabilities are critical because the ad auction multiplies bid × P(click) — uncalibrated scores translate directly into wrong winners. Serve under 50 ms per impression; retrain hourly to daily; A/B every change.

**Expansion / why this is the answer.**

**1. Clarify requirements.**
Functional:
- Score `P(click | user, ad, context)` for every candidate ad in an auction.
- The auction picks the winner based on `bid × P(click)` (or a richer utility).
- Returns winner ad + price.

Non-functional:
- Latency: p99 < 50 ms (auctions chained with publisher page load).
- Throughput: 1M+ QPS at peak.
- **Calibrated probabilities** (not just rankings) — auction math depends on actual probabilities.
- Real-time-ish updates: a new ad campaign should be biddable within minutes.

Clarifying Qs:
- "Search ads, display ads, or sponsored social?"
- "Bidding model — CPC, CPM, CPA?"
- "Are we predicting click or downstream conversion?"
- "How many candidates per auction?"

**2. Define metrics.**

Online:
- **North-star**: revenue / RPM (revenue per mille impressions).
- **Guardrails**: user satisfaction proxies (bounce rate, complaints, post-click dwell), advertiser ROI, fraud rate.

Offline:
- **AUC** / **GAUC**: ranking quality.
- **LogLoss** / NLL: calibration-sensitive proxy.
- **Calibration**: ECE — for CTR specifically, the *level* of predicted P(click) matters, not just ordering.
- **Population stability**: PSI on features.

Online ↔ offline gap: AUC and LogLoss correlate with RPM but not perfectly; always A/B.

**3. Data and labels.**

Sources:
- **Impression logs**: every ad shown, every click. Trillions/month.
- **User features**: device, OS, locale, demographics (where legal), prior behavior aggregations, recent ad interactions.
- **Ad features**: advertiser, creative type, category, image/text embeddings, campaign metadata.
- **Context**: page / app, ad slot, time-of-day, browser.
- **Cross features**: user-advertiser affinity, user-category affinity.

Labels:
- Binary: click vs. no-click (per impression).
- For downstream metrics, post-click conversion events (delayed labels — typically logged within 24h).

Volume / class imbalance:
- Typical CTR: 0.1–5% — severe imbalance.
- Negative sampling or weighting required.

Decontamination & leakage:
- Don't use features computed after the impression event (post-click conversion, user's later actions).
- Time-respecting splits.

**4. Modeling.**

Architecture (any of, all with embeddings over high-card categoricals):
- **Wide & Deep** (Cheng et al. 2016): wide part for memorization, deep part for generalization.
- **DeepFM** (Guo et al. 2017): factorization machines for 2nd-order interactions + DNN.
- **DCN-V2** (Wang et al. 2021): explicit higher-order feature crosses.
- Modern: transformer over user history + cross attentions to ad embedding.

Embeddings:
- High-cardinality categorical (ad id, user id, etc.) → dense embeddings, learned end-to-end.
- Sharing: cross-embedding for similar fields (e.g., user country and ad country).

Loss:
- Binary cross-entropy on (impression, clicked).
- Often add a calibration term (Platt scaling or isotonic regression post-hoc).

Calibration:
- Critical for auctions. Post-process model probabilities with isotonic regression or Platt scaling on a held-out set.
- Drift-monitor calibration over time; recalibrate if ECE rises.

Cold-start:
- New ad: rely on creative embeddings + advertiser priors + exploration boost.
- New advertiser: even higher exploration boost; learn fast.

Counterfactual / bias correction:
- The training data is biased — only the impressions that won an auction were shown. Use sample weights or off-policy correction (inverse-propensity weighting).

**5. Serving.**

Latency: p99 < 50 ms.
- Pre-compute user and ad embeddings; serve them.
- Per-impression: lookup features (sub-ms), forward pass (sub-10 ms on a small model), calibrate, return.
- Sharded model serving; horizontal scale.

Feature store:
- Online features (real-time aggregates) + offline features (long-term aggregates).
- Same code computes features at train and serve time (train-serve consistency).

Online learning:
- Some shops use FTRL-Proximal (McMahan et al. 2013, Google's "Ad Click Prediction: a View from the Trenches") for online learning.
- Modern: nightly batch retrain + periodic warm-start; risky to do continuous online learning at scale.

**6. Monitoring + iteration.**

Drift:
- Input drift on key features.
- Concept drift: user preferences, market changes.
- Label drift: organic CTR varies seasonally.

Retraining:
- Daily for the ranker.
- Hourly for some online-learning features (recent CTR aggregates).

Online evaluation:
- A/B with bucketed users.
- Power analysis: tiny RPM lifts need huge sample sizes.
- Long-term holdout to track creep.

Common pitfalls:
- **Position bias**: top-slot ads always get more clicks; the model must not learn "this is the best ad" from position alone. Mitigations: position as a feature at train, neutral position at serve; or counterfactual training.
- **Survivorship bias**: ads that performed badly were paused; the data is biased.

---

**Common follow-ups.**

- "How do you handle delayed labels (post-click conversion)?" → Delayed-feedback models (Chapelle 2014).
- "What's FTRL?" → Follow-the-Regularized-Leader; an online-learning algorithm popular for ad CTR (McMahan et al. 2013).
- "Why does calibration matter more here than in recsys?" → The auction *multiplies* by P(click); poor calibration directly distorts revenue.

**Common mistakes.**

- Optimizing AUC, ignoring calibration.
- Forgetting position bias.
- Skipping the cold-start story.
- No online-evaluation discipline.

**References.**

- [Cheng et al. — "Wide & Deep"](https://arxiv.org/abs/1606.07792) — Wide & Deep.
- [Guo et al. — "DeepFM"](https://arxiv.org/abs/1703.04247) — DeepFM.
- [Wang et al. — "DCN V2"](https://arxiv.org/abs/2008.13535) — DCN-V2.
- [McMahan et al. — "Ad Click Prediction: a View from the Trenches"](https://research.google/pubs/pub41159/) — FTRL and production lessons.
- [Chapelle — "Modeling Delayed Feedback in Display Advertising"](https://research.criteo.com/wp-content/uploads/2018/09/chapelle_delayed_feedback.pdf) — delayed labels.
