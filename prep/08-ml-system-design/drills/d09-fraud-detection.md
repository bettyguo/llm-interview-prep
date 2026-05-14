# D9 — Fraud detection (payments / account takeover)

A worked drill following the [6-step structure](../README.md#the-canonical-6-step-structure).

---

### Q: Design a fraud-detection system for payments.

**Category:** system-design
**Difficulty:** senior
**Tags:** [fraud, anomaly-detection, imbalanced-classification, real-time]

**Short answer.** Real-time scoring per transaction: a gradient-boosted-tree model (XGBoost / LightGBM) over engineered features (velocity, device, geography, amount distribution, graph features). Decision pipeline: auto-allow at low risk, auto-block at high risk, send-for-review in the middle. Severe class imbalance (~0.1%), so calibrate carefully and use precision-recall metrics. Adversarial: fraudsters adapt; the model must too — retrain frequently and run a parallel anomaly-detection model for novel attacks.

**Expansion / why this is the answer.**

**1. Clarify requirements.**
Functional:
- Score each transaction `P(fraud)` in real time.
- Decision: allow / block / step-up auth / manual review.
- Account-takeover detection: separate model class.

Non-functional:
- Latency: p99 < 100 ms (transaction is held).
- Throughput: thousands of TPS at peak.
- Class imbalance: severe (~0.1% fraud rate).
- Adversarial: fraudsters actively evade.
- Compliance: PCI-DSS, regional regulations.

Clarifying Qs:
- "Card-present, card-not-present, or both?"
- "Are chargebacks the label source?"
- "What's the latency budget for the hold?"
- "How do we balance fraud loss vs. false-decline customer pain?"

**2. Define metrics.**

Online:
- **Fraud loss rate**: $ fraud / $ transaction volume.
- **False-decline rate**: legitimate transactions blocked (drives customer pain).
- **Manual-review precision**: of reviewed transactions, % are truly fraud.
- **Chargeback rate**.

Offline:
- **PR-AUC** (the right metric under heavy imbalance, see T1 question).
- **Precision-at-K%** (the K% highest-risk transactions; what's their fraud rate?).
- **Recall at fixed precision**.

**3. Data and labels.**

Sources:
- Transaction logs.
- Chargeback events (delayed: 60–120 days).
- Confirmed-fraud labels from manual review.
- Account-takeover signals (password resets, device fingerprints).
- Third-party signals (IP reputation, device intelligence).

Features:
- **Transaction-level**: amount, currency, merchant category, device, geography.
- **Velocity**: count + amount in last 1h, 1d, 7d, 30d (per user, per card, per device).
- **Geographic**: distance from prior; impossible travel ("New York at 09:00 then Tokyo at 09:15").
- **Behavioral**: deviation from user's historical pattern.
- **Graph features**: connected to known-bad cards / devices / accounts.
- **Time-of-day, day-of-week**: legitimacy patterns.

Labels:
- Delayed: chargeback labels arrive weeks-months later.
- Confirmed-fraud (from manual review): faster but biased toward what review caught.
- Use **proxy labels**: combine, weighted by reliability.

Class imbalance: ~0.1% fraud. Negative sampling at train; report PR-AUC.

Decontamination:
- Strict time-respecting splits; never use future info.
- No leakage of post-transaction features.

**4. Modeling.**

**4a. Primary model: GBM (XGBoost / LightGBM)**:
- Strong on tabular with engineered features.
- Handles missing values, mixed types.
- Calibrated post-hoc (isotonic regression).

**4b. DL feature extractor (optional)**:
- For sequence features (user's transaction history over time), an LSTM / Transformer can produce embeddings used as additional features for the GBM.

**4c. Anomaly detection (parallel)**:
- Isolation Forest or autoencoder-reconstruction-error on aggregate features.
- Catches **novel** attack patterns the supervised model hasn't seen.
- Output combined with GBM via a meta-model or rule-based combiner.

**4d. Graph-based**:
- Build a graph of (user, device, card, merchant) connections.
- Graph neural network or community-detection on the graph to find rings of related fraud.

**4e. Decision policy**:
- `P(fraud) < 0.01` → allow.
- `0.01 ≤ P < 0.5` → step-up auth (3DS, OTP).
- `P ≥ 0.5` → manual review.
- `P > 0.95` → block.
- Thresholds tuned to balance fraud loss vs. customer pain.

**5. Serving.**

Latency: p99 < 100 ms.
- Feature store: pre-computed aggregates + real-time stream (Flink / Kafka Streams).
- GBM inference: ~5 ms.
- Anomaly model: similar.
- Decision logic: sub-ms.
- Network + serialization: ~50 ms buffer.

Real-time features:
- Velocity aggregates updated in stream (Flink, Kafka Streams).
- Approximate aggregations (Count-Min sketch) at scale.

Train-serve parity:
- Same code computes features at train and serve time.

**6. Monitoring + iteration.**

Drift:
- Fraud patterns shift fast; weekly retraining is common.
- Concept drift = the very thing fraudsters drive.

Retraining:
- Daily to weekly for the primary GBM.
- Anomaly model retrained less often.

Online evaluation:
- A/B with care — exposing fraud to a control group is costly.
- Champion-challenger: shadow-score with new model; compare against decisions taken.

Adversarial:
- Continuous red-team simulation of attack patterns.
- Feature-store backfilled when adding new features.

Reviewer feedback:
- Manual review confirms / overrides every escalated case.
- Confirmed-fraud labels feed back into training.
- Reviewer dashboards for new attack patterns.

---

**Common follow-ups.**

- "Why GBM over a neural net?" → Tabular + engineered features + small data per class — GBMs win (see T1, Grinsztajn et al. 2022). Deep nets help if you have very large data and sequential modeling.
- "How do you handle delayed labels?" → Train on chargebacks lagged by their typical lag; combine with faster proxies.
- "What's the false-decline tradeoff?" → Each false decline is a customer pain event; optimize a weighted cost function balancing fraud-loss-$ against customer-friction-$.

**Common mistakes.**

- Optimizing accuracy under 0.1% positive rate (model that always says "not fraud" is 99.9% accurate, useless).
- No threshold tuning / decision-policy story.
- No adversarial / continuous-retraining story.
- Treating chargebacks as immediate labels.

**References.**

- [Bahnsen et al. — "Feature Engineering Strategies for Credit Card Fraud Detection"](https://arxiv.org/abs/1602.06723) — fraud features.
- [Grinsztajn et al. — "Why do tree-based models still outperform deep learning on tabular data?"](https://arxiv.org/abs/2207.08815) — GBM-over-DNN on tabular.
- [Stripe Radar product overview](https://stripe.com/radar) — production case study.
- [Pourhabibi et al. — "Fraud detection: A systematic literature review"](https://doi.org/10.1016/j.dss.2020.113303) — survey.
