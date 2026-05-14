# D10 — ML Feature Store design

A worked drill following the [6-step structure](../README.md#the-canonical-6-step-structure).

---

### Q: Design an ML feature store for an organization with many models.

**Category:** system-design
**Difficulty:** senior
**Tags:** [feature-store, platform, train-serve-skew]

**Short answer.** A feature store has two tiers: an **online store** (low-latency key-value lookup at serve time) and an **offline store** (analytical store for training feature retrieval), both populated from the same **feature definitions** (transformations applied to source data). Single source of truth eliminates train-serve skew. Adds: point-in-time-correct backfills, feature versioning, monitoring, and discovery (a registry).

**Expansion / why this is the answer.**

**1. Clarify requirements.**
Functional:
- Define a feature once → make it usable for training and serving.
- Read features at low latency online.
- Read historically-accurate feature values for training (point-in-time correctness).
- Multi-team / multi-model sharing.

Non-functional:
- Online lookup p99 < 10 ms.
- Offline query throughput: handle ~TB-scale training jobs.
- Feature freshness: streaming features within seconds; batch features within hours.
- Versioning: schema evolution doesn't break models.
- Observability: per-feature staleness, distribution drift.

Clarifying Qs:
- "How many teams / models will use this?"
- "Is there a real-time / streaming source, or mostly batch?"
- "What's the existing data infra — Snowflake/BigQuery/Spark?"

**2. Define metrics.**

Platform:
- **Feature reuse**: across models. Higher = more value.
- **Online latency** p50/p99 per feature group.
- **Train-serve skew rate**: % of features differing between train and serve at the same logical time.
- **Feature freshness** (lag).

Model metrics (downstream):
- Indirect; the model team owns these.

**3. Data and labels.**

Sources:
- Raw events (clicks, purchases, logs).
- Operational databases.
- Streaming sources (Kafka).
- Third-party.

Inputs to features:
- Single events: enrich and store.
- Aggregations: count, sum, mean over time windows per entity.

Schema:
- Entity (user_id, item_id, session_id, etc.).
- Feature name.
- Value (typed).
- Timestamp (when the value became valid).

**4. Modeling.**

The "model" here is the system, not an ML model. Components:

**4a. Feature registry**:
- Versioned definitions of features (transformations).
- Owner, description, freshness requirement, ACL.
- Searchable.

**4b. Transformation engine**:
- Computes features from sources.
- Batch (Spark / Snowflake / BigQuery): aggregate over windows; daily backfills.
- Streaming (Flink / Kafka Streams): incremental updates for low-latency features.
- The same logical definition compiles to both batch and streaming (Tecton's approach).

**4c. Online store**:
- Low-latency key-value: Redis, DynamoDB, BigTable, Cassandra.
- Per-entity row indexed by entity-id.
- Hot path: read N features for entity X in one round-trip.

**4d. Offline store**:
- Columnar / analytical: Snowflake, BigQuery, Parquet on object store.
- Stores historical values with timestamps for point-in-time-correct training.

**4e. Point-in-time joins**:
- The hardest correctness problem.
- For each training sample at time T, retrieve feature values *as they were at time T-Δ* (Δ accounts for inference lag).
- Implementation: as-of join in SQL or a dedicated planner.
- This is what prevents *future leakage* into training.

**4f. Monitoring**:
- Per-feature: freshness, missing-value rate, distribution stats (mean, std, PSI vs. training).
- Alert on drift.

**5. Serving.**

Online read path:
- Model server requests features for entity X by name.
- SDK batches reads to the online store.
- Returns features (with default / missing-value handling).
- Latency budget: < 10 ms p99 for a few-dozen-feature read.

Write path:
- Streaming features write directly to online store on each event.
- Batch features write to both online (latest value) and offline (history) stores.

Train path:
- Training job queries offline store with point-in-time join.
- Returns historically-correct features.

**6. Monitoring + iteration.**

Per-feature monitoring:
- Freshness (lag since last update).
- Coverage (% entities with this feature populated).
- Distribution (PSI vs. baseline; alert on drift).
- Missing-value rate.

Schema evolution:
- New features: added; old models unaffected.
- Removing features: deprecate gracefully.
- Changing semantics: new feature version; models pin to a version.

Train-serve skew detection:
- Sample inputs at serve time; compare against the values used at training for the same entity-time.
- Persistent skew → model team alerted; bug in either the feature pipeline or the offline join.

Cost:
- Online store storage (hot data).
- Offline store storage (cold history).
- Transform compute (batch + streaming).
- Tradeoffs: keep aggregations only at the granularity used; don't store everything.

---

**Common follow-ups.**

- "What's point-in-time correctness?" → Train on the value a feature *had* at the moment of the training event, not its current value. Prevents future leakage.
- "How is this different from a normal data warehouse?" → Same data, but a feature store also serves online with low latency *and* enforces a unified compute path so train and serve match.
- "Build vs. buy?" → Tecton, Feast (OSS), Vertex AI Feature Store, SageMaker Feature Store. Buy at small scale; build/augment at very large scale or for specific latency needs.

**Common mistakes.**

- No point-in-time join → silent feature leakage.
- Different compute paths for train and serve → skew.
- No versioning → breaking changes break models.
- No monitoring → drift goes undetected.

**References.**

- [Tecton — Feature Store architecture](https://docs.tecton.ai/) — primary docs.
- [Feast project](https://feast.dev/) — open-source feature store.
- [Uber — Michelangelo platform overview](https://www.uber.com/blog/michelangelo-machine-learning-platform/) — feature platform case study.
