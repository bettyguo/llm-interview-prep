# D3 — Multi-modal content moderation

A worked drill following the [6-step structure](../README.md#the-canonical-6-step-structure).

---

### Q: Design a content moderation pipeline for a user-generated-content platform.

**Category:** system-design
**Difficulty:** senior
**Tags:** [moderation, multi-modal, safety, escalation]

**Short answer.** Tiered pipeline: **(a) automated classifiers** at ingest (text, image, audio, video) flag violations; **(b) auto-action** on high-confidence violations (CSAM, terrorism — must be near-perfect-recall); **(c) human review queue** for ambiguous cases, prioritized by severity × reach; **(d) appeals and feedback loop** to improve models. Optimize for high recall on critical categories with low false-positive rate on broader policy violations; calibrate per-category thresholds; never auto-remove user content at low confidence without human review (false positives are costly).

**Expansion / why this is the answer.**

**1. Clarify requirements.**
Functional:
- Detect policy violations across modalities (text, image, video, audio).
- Categories: CSAM, terrorism, gore, hate, harassment, self-harm, nudity, spam, scam, copyright.
- Actions: block, hide-from-feed, age-gate, downrank, send-to-review, label.
- Multi-tenant: per-region policy differences.

Non-functional:
- Throughput: millions of posts/min.
- Latency: ingest-time scoring < 200 ms; allow async processing for video.
- Recall floor on critical categories (CSAM near 100%).
- Precision floor on actioning to avoid wrongful takedowns.
- Auditability: every decision has an explanation.

Clarifying Qs:
- "Which platform? UGC scale matters a lot (Reddit vs. Facebook)."
- "Sync vs. async OK?"
- "Which jurisdictions / legal requirements (GDPR, DSA, US 230 dynamics)?"

**2. Define metrics.**

Online:
- **Per-category recall**: did we catch X% of true violations? Hardest to measure (need a labeled holdout from organic content).
- **Per-category precision**: of our removals, X% were correct.
- **Appeal-success rate**: high = our actions are wrong too often.
- **Mean time to action (MTTA)** for critical categories.
- **Prevalence**: % of organic content that's violating (track over time).

Offline:
- AUC per category.
- F1 at the operating threshold.
- Reviewed-agreement rate with human raters.

Online ↔ offline: hard. Real violations are rare; offline AUC can be misleading. Use real labeled traffic samples.

**3. Data and labels.**

Sources:
- **Labeled-by-reviewer**: human moderators label flagged content. Source of truth.
- **User reports**: noisy but high-volume.
- **Hash matches**: known-CSAM, known-malware via PhotoDNA / CSAM hash databases (NCMEC, IWF).
- **Adversarial / red-team**: synthesize-and-label new categories.

Labels:
- Multi-label (a post can be both hate and harassment).
- Severity scores per category.
- Time-of-labeling matters (policies evolve).

Class imbalance: violations are rare (often <1% per category). Heavy negative sampling at train.

Annotator agreement: report Cohen's κ; train on consensus labels.

**4. Modeling.**

Per-modality models (cascade or parallel):
- **Text**: encoder-only LM (DeBERTa, ModernBERT, or a specialized hate-speech model); multi-label classifier head per category.
- **Image**: CNN (EfficientNet) or ViT; per-category head.
- **Audio**: speech-to-text → text classifier; or audio fingerprinting for music/sound.
- **Video**: sample N frames + audio track + transcript → per-modality models + fusion model.
- **Multi-modal fusion**: CLIP-style joint embedding + classifier head; or late-fusion over per-modality scores.

For LLM-driven moderation (2024+ pattern):
- A multimodal LLM (Gemini, Claude, GPT-4o) with policy prompt: "Does this violate policy X?"
- Strengths: handles novel violations zero-shot, easy to update policy.
- Weaknesses: cost; LLM jailbreaks; calibration.
- Usually a hybrid: cheap classifiers as first pass, LLM for ambiguous cases.

Threshold calibration:
- Per-category, per-region threshold.
- CSAM: extremely low threshold (high recall, accept low precision; humans review every flag).
- Mild violations: higher threshold; tolerate some misses to avoid mass takedowns.

Escalation:
- High-confidence violation → auto-action.
- Medium → human review queue.
- Low → no action.

Adversarial robustness:
- Adversarial inputs (perturbed images, mis-spelled hate speech, "leetspeak"); regular adversarial training.

**5. Serving.**

Sync vs. async:
- Pre-publish scan: fast classifiers; blocks publishing on high-confidence violations.
- Post-publish: heavier models (video understanding) run async; can hide content retroactively.

Infrastructure:
- Inference: batched GPU-served classifiers; modality-specific.
- Hash-match: in-memory hash lookup against CSAM/known-violation DB.
- Human review queue: severity × reach prioritization.

Latency budget:
- Critical category sync: < 500 ms.
- General async: minutes acceptable.

**6. Monitoring + iteration.**

Per-category metrics dashboarded daily.

Drift:
- New attack patterns (memes, slang, deepfakes).
- Policy updates require re-labeling.

Retraining:
- Per-category, monthly to quarterly.
- For fast-moving categories (e.g. election misinformation), weekly.

Appeals:
- Every appeal that succeeds = a false positive. Feed back into training.
- Appeals dashboard tracks per-category appeal-success rate; rising rate signals model regression.

Red team:
- Continuously test for jailbreaks, adversarial inputs.

Human review:
- Worker well-being is critical; CSAM exposure has psychological cost. Limit shifts, provide counseling.

---

**Common follow-ups.**

- "How do you handle policy updates?" → Versioned policy; re-label past flagged content; retrain.
- "What about a deep-fake detection system?" → Hard; current best is detection-via-known-watermarks plus heuristic models that lag deepfake quality.
- "How do you handle jurisdictional differences?" → Per-region policy + per-region thresholds + per-region classifiers when content is severely region-specific.

**Common mistakes.**

- Single threshold across all categories.
- No human-in-the-loop for low-confidence cases.
- Skipping the appeals path.
- Ignoring annotator well-being.

**References.**

- [Schmidt & Wiegand — "A Survey on Hate Speech Detection using Natural Language Processing"](https://aclanthology.org/W17-1101/) — text-side survey.
- [Microsoft PhotoDNA](https://www.microsoft.com/en-us/photodna) — CSAM hash matching.
- [Meta — Community Standards Enforcement Report](https://transparency.meta.com/policies/community-standards/) — production transparency reporting on moderation outcomes.
- [Radford et al. — "CLIP"](https://arxiv.org/abs/2103.00020) — multimodal joint embedding.
