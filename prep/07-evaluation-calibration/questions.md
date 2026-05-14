# Evaluation & Calibration — questions

Entries follow the [Q&A schema](../../CONTRIBUTING.md#the-qa-entry-schema).

---

### Q: What is LLM-as-judge, and what are its biases?

**Category:** concept
**Difficulty:** mid
**Tags:** [llm-as-judge, bias, eval]

**Short answer.** LLM-as-judge uses a (typically larger) LLM to score model outputs — pairwise ("which is better, A or B?") or scalar ("rate this answer 1–10"). It's the dominant cheap-and-fast eval method in 2026 but has known biases: **position bias** (favors the first/last option), **length bias** (prefers longer answers), **self-preference** (a model favors its own outputs), and **style over substance**. Mitigate with randomized order, multiple judges, calibration against human-graded subsets, and reference-based grading where possible.

**Expansion / why this is the answer.**
- The pattern (Zheng et al. 2023, MT-Bench / Chatbot Arena):
  - Show the judge model two candidate responses (and the prompt).
  - Ask: which is better? Or score each on a rubric.
- **Documented biases**:
  - **Position bias**: GPT-4 prefers the first option ~60% of the time without randomization.
  - **Length bias**: longer = better (RLHF-baked-in correlation).
  - **Self-preference**: a model judges its own outputs more favorably (Panickssery et al. 2024).
  - **Style bias**: markdown formatting, confidence, "sounding good."
  - **Verbosity bias** related to length but specifically about adding unnecessary detail.
- **Mitigations**:
  - **Swap-and-average**: judge each pair twice with positions swapped; average.
  - **Multiple judges**: ensemble (e.g. Claude + GPT-4 + Gemini).
  - **Reference-based grading**: provide a gold answer; judge whether candidate matches it (less bias than open judging).
  - **Calibration**: maintain a human-graded set; track judge-vs-human agreement; retrain prompts if drift.
  - **Rubrics with explicit criteria** rather than free-form judgment.
- **When LLM-as-judge fails badly**:
  - Domain expertise the judge lacks.
  - Highly subjective tasks (creative writing).
  - Safety/compliance evaluation (humans needed).

**Common follow-ups.**
- "What's the typical judge-human agreement rate?" → ~80–85% on well-defined tasks (MT-Bench); lower on subjective ones.
- "Can a small model judge a big one?" → Sometimes; usually you want the judge ≥ the candidates' capability.

**Common mistakes.**
- Reporting judge scores without randomization or human calibration.
- Using the same model as candidate and judge.

**References.**
- [Zheng et al. — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"](https://arxiv.org/abs/2306.05685) — the canonical paper.
- [Panickssery, Bowman, Feng — "LLM Evaluators Recognize and Favor Their Own Generations"](https://arxiv.org/abs/2404.13076) — self-preference.

---

### Q: What is benchmark contamination, and how do you detect it?

**Category:** concept
**Difficulty:** mid
**Tags:** [contamination, decontamination, benchmarks]

**Short answer.** Contamination is the eval set leaking into the training data — the model is "remembering" answers, not solving the task. Detection: (a) **n-gram overlap** between training corpus and benchmark questions; (b) **perplexity / loss tests** (a contaminated benchmark has unusually low loss); (c) **paraphrase-and-test** (if scores drop on paraphrased questions, contamination); (d) **before/after-cutoff tests** (use benchmarks created after the model's training cutoff). Trust technical reports that document their decontamination methodology.

**Expansion / why this is the answer.**
- **Why it happens**: web-crawled training corpora include benchmark questions verbatim or in derivatives (study guides, GitHub gists, derived datasets).
- **Detection methods**:
  - **N-gram overlap** (Brown et al. 2020): check 13-gram exact matches between training docs and benchmark; remove contaminated training docs.
  - **Membership inference**: train a classifier to predict whether a sample was in training; high accuracy on benchmark items = contamination.
  - **Perplexity gap**: model has unusually low perplexity on benchmark text vs. similar held-out text.
  - **Paraphrase testing** (Yang et al. 2023): paraphrase the question; if the score drops sharply, the model was memorizing.
  - **Post-cutoff benchmarks** (e.g. SWE-bench Live, MMLU-Pro from later dates, GPQA): created after model cutoffs to be uncontaminated.
- **What technical reports document**:
  - GPT-4 / Claude / Gemini / LLaMA 3 all describe their decontamination process.
  - LLaMA 3 used 8-gram + 50% overlap thresholds.
- **Why it matters**: a model reporting 95% on MMLU could mean "very capable" or "memorized MMLU." Decontamination methodology is the difference.
- **In interviews**: signal awareness that high benchmark scores aren't always capability; ask about methodology when relevant.

**Common follow-ups.**
- "Has GPT-4 been shown contaminated on specific benchmarks?" → Yes — early reports identified some HumanEval contamination; the technical report addresses this.
- "What's a 'dynamic eval'?" → A benchmark that updates or randomizes per access, resistant to memorization.

**Common mistakes.**
- Treating high benchmark scores as direct capability evidence.
- Trusting "we decontaminated" without methodology.

**References.**
- [Magar & Schwartz — "Data Contamination: From Memorization to Exploitation"](https://arxiv.org/abs/2203.08242) — the canonical paper.
- [Yang et al. — "Rethinking Benchmark and Contamination for Language Models with Rephrased Samples"](https://arxiv.org/abs/2311.04850) — rephrase test.
- [Sainz et al. — "NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark"](https://arxiv.org/abs/2310.18018) — practical detection.

---

### Q: Explain calibration. How do you measure it?

**Category:** concept
**Difficulty:** mid
**Tags:** [calibration, ece, reliability-diagram]

**Short answer.** A model is **calibrated** if its predicted probabilities match empirical correctness: when it says "I'm 70% confident," it should be right ~70% of the time. Measure with **ECE (Expected Calibration Error)** — bin predictions by confidence, compute per-bin accuracy, weight-average the |confidence − accuracy| gap. Plot a **reliability diagram** (confidence vs. accuracy per bin). Modern LLMs are *poorly calibrated* by default (overconfident); RLHF makes calibration worse.

**Expansion / why this is the answer.**
- **Definition** (Guo et al. 2017): a classifier outputs probabilities `p`; an empirical bin of samples with `p ≈ 0.7` should have ~70% accuracy.
- **ECE**:
  - Bin predictions into `M` bins by confidence.
  - For each bin: `acc = correct/total`, `conf = avg predicted probability`.
  - `ECE = Σ_bins (|B_m|/N) · |acc_m − conf_m|`.
  - Lower is better; perfect calibration = 0.
- **Reliability diagram**: plot bin accuracy vs. bin confidence; perfect calibration is the y=x diagonal.
- **LLM calibration**:
  - **Pre-RLHF**: base LLMs are surprisingly well-calibrated for multiple-choice (the GPT-4 tech report reliability diagram, base model is near diagonal).
  - **Post-RLHF**: calibration degrades — RLHF makes the model more confidently wrong (Kadavath et al. 2022 also; OpenAI's GPT-4 system card explicitly notes this).
  - **Token-probability calibration**: at the next-token level, calibration is decent; at the "I'm confident in this answer" level, less so.
- **Mitigations**:
  - **Temperature scaling** (Guo et al. 2017): post-hoc, learn a single scalar `T` to apply to logits; minimizes NLL on a held-out set; preserves accuracy.
  - **Verbal probability prompts**: ask the model "with what probability is X true?" — sometimes better-calibrated than the raw token probabilities.
  - **Verbalized confidence** (Tian et al. 2023): train the model to say "I am 70% confident"; correlates better with correctness post-training.

**Common follow-ups.**
- "Why does RLHF degrade calibration?" → The reward model prefers confident-sounding answers; the policy learns to be confident.
- "What's the Brier score?" → `Σ (p − y)²` for binary tasks; a proper scoring rule that combines accuracy and calibration.

**Common mistakes.**
- Reporting accuracy without calibration; "high accuracy but overconfident" can be worse than "lower accuracy but well-calibrated."
- Conflating temperature scaling (post-hoc calibration) with sampling temperature (decoding).

**References.**
- [Guo et al. — "On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599) — ECE, temperature scaling.
- [Kadavath et al. — "Language Models (Mostly) Know What They Know"](https://arxiv.org/abs/2207.05221) — LLM calibration.
- [OpenAI — GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) — RLHF and calibration.

---

### Q: How do you measure hallucination?

**Category:** concept
**Difficulty:** mid
**Tags:** [hallucination, faithfulness, factscore]

**Short answer.** "Hallucination" splits into two: **factuality** (statement matches the world / a reference) and **faithfulness** (statement matches a provided context, e.g. in RAG). Measure factuality with reference-based metrics (FActScore, TruthfulQA, FactBench) — break the answer into atomic claims, check each against a knowledge source. Measure faithfulness with NLI / entailment models or LLM-as-judge over `(answer claim, retrieved passage)` pairs.

**Expansion / why this is the answer.**
- **The two flavors**:
  - **Factuality**: is the statement true in the world?
  - **Faithfulness**: is the statement supported by the provided context (RAG, prompted documents)?
- **Factuality measurement**:
  - **FActScore** (Min et al. 2023): decompose answer into atomic facts; for each, retrieve a Wikipedia passage; check support; aggregate as fraction supported.
  - **TruthfulQA** (Lin et al. 2022): a curated benchmark of 800+ questions designed to test whether models avoid common misconceptions.
  - **HaluEval** (Li et al. 2023): hallucination-labeled question/answer dataset.
  - **SimpleQA** (OpenAI, 2024): short-form factual questions; measures pure factual recall.
- **Faithfulness measurement** (in RAG):
  - **NLI-based**: classify each `(claim, retrieved passage)` pair as entailment / contradiction / neutral.
  - **LLM-as-judge**: prompt a strong model to score support.
  - **FaithBench** (Bao et al. 2024): a faithfulness benchmark for RAG summarization.
- **Production patterns**:
  - **Sample claims, hand-verify**: small set, cheap.
  - **Automated atomic-claim pipeline**: decompose → retrieve → score; RAGAS implements this.
  - **End-user signals**: thumbs-down, "this is wrong" labels.
- **Caveats**:
  - "Atomic claim" decomposition itself is noisy; an LLM does it.
  - Knowledge source may be incomplete (Wikipedia has gaps).
  - Time-bound facts ("the current CEO of X") are hardest.

**Common follow-ups.**
- "What's the difference between hallucination and confabulation?" → Often used interchangeably; some use "confabulation" for plausible-sounding-but-wrong; "hallucination" is the broader term in 2026.
- "Why is TruthfulQA controversial?" → It's adversarially designed; "good" answers sometimes amount to refusing to answer; models that confidently state correct mainstream answers can score lower than ones that hedge.

**Common mistakes.**
- Conflating factuality with faithfulness.
- Treating LLM-as-judge as ground truth for factuality without human spot-check.

**References.**
- [Min et al. — "FActScore"](https://arxiv.org/abs/2305.14251) — FActScore.
- [Lin et al. — "TruthfulQA"](https://arxiv.org/abs/2109.07958) — TruthfulQA.
- [Li et al. — "HaluEval"](https://arxiv.org/abs/2305.11747) — HaluEval.
- [Es et al. — "RAGAS"](https://arxiv.org/abs/2309.15217) — RAG faithfulness pipeline.

---

### Q: Walk me through designing an eval set for a new LLM-powered feature.

**Category:** concept
**Difficulty:** senior
**Tags:** [eval-design, sample-size, stratification]

**Short answer.** Define the **task** narrowly. Sample **representative inputs** from real / projected traffic (or hand-craft if no traffic yet). Stratify by **important slices** (input length, user segment, edge-case categories). For each, collect a **gold output** or **acceptance criterion**. Aim 100–500 examples for first-pass; expand as you find failure modes. Track **multiple metrics**, not one: correctness, refusal rate, latency, cost. Critically, **review the labels** — your eval is only as good as its labels.

**Expansion / why this is the answer.**
- **Steps**:
  1. **Task spec**: input → output, success criterion.
  2. **Sample inputs**:
     - From real traffic if you have it (log queries).
     - From similar product / public benchmarks adapted.
     - From hand-crafted edge cases.
  3. **Stratify**:
     - Length (short / medium / long).
     - User segments (new / power / enterprise).
     - Edge cases (multilingual, code, dates, names).
     - Failure-mode-targeted (jailbreaks, ambiguous queries, etc.).
  4. **Gold outputs**:
     - Hand-written for high-stakes.
     - LLM-generated then human-validated for scale.
     - Acceptance criterion (e.g. "answer contains the city name") for cheap.
  5. **Metrics**:
     - Correctness / accuracy.
     - Per-slice accuracy (the stratification pays off here).
     - Refusal / over-cautious rate.
     - Latency / cost.
     - Safety flags.
  6. **Cadence**: run the eval on every model change; track over time.
- **Statistical hygiene**:
  - Confidence intervals matter: with N=100, a 5pp swing is within noise.
  - Use a held-out test set separate from any tuning.
- **Eval rot**:
  - As the model gets good at the eval, plateau.
  - Refresh with new examples (hardest ones first).
- **Frameworks**: OpenAI Evals, Inspect (UK AISI), DeepEval, Helm.

**Common follow-ups.**
- "How do you handle subjective tasks?" → Pairwise human preferences; report agreement rate among labelers; calibrate LLM-as-judge against the human set.
- "What sample size do you need?" → Power analysis: depends on the effect you want to detect. 100 → ~10pp resolution; 1000 → ~3pp.

**Common mistakes.**
- Sampling from convenient queries (top of the log) rather than representative.
- Single number; no slices.
- Evolving the eval over time without versioning.

**References.**
- [Bowman & Dahl — "What Will it Take to Fix Benchmarking in Natural Language Understanding?"](https://arxiv.org/abs/2104.02145) — eval design principles.
- [OpenAI Evals project](https://github.com/openai/evals) — practical framework.

---

### Q: Compare pairwise comparison vs. single-grade evaluation. When do you use each?

**Category:** concept
**Difficulty:** mid
**Tags:** [pairwise, single-grade, eval-methodology]

**Short answer.** **Pairwise**: show two outputs; rate which is better. **Single-grade**: rate one output on a rubric (1–5 or pass/fail). Pairwise is **easier for humans/judges** (relative is easier than absolute) and **resistant to scale drift**; downside is `O(N²)` if you compare all pairs. Single-grade is **fast and absolute**; downside is rater drift and "what does '4 out of 5' mean across raters." Use pairwise for fine-grained model comparison (Chatbot Arena, RLHF preference collection). Use single-grade for monitoring an absolute quality bar in production.

**Expansion / why this is the answer.**
- **Pairwise** (Bradley-Terry preference model):
  - Show `(input, output_A, output_B)`; rater picks one or "tie."
  - From many pairwise comparisons, fit Elo / Bradley-Terry to derive a ranking.
  - **Chatbot Arena** (LMSYS): the canonical multi-model pairwise leaderboard.
  - Used in RLHF data collection.
- **Single-grade**:
  - Show `(input, output, rubric)`; rate 1–5 or assign a pass/fail.
  - Used in production monitoring (e.g. "% answers ≥ 4/5").
- **Tradeoffs**:
  - Pairwise: lower variance per comparison; harder to scale to many models (`O(N²)`).
  - Single-grade: scales linearly; rater drift over time.
- **Hybrid**: a fixed rubric *with* pairwise — "Which better satisfies criterion X?"
- **LLM-judge variants**:
  - Pairwise + position swap + average.
  - Single-grade with explicit rubric.
- **Choosing**:
  - Evolving model under development: pairwise vs. last-good.
  - Production SLO: single-grade pass rate.
  - Public leaderboard: pairwise (Arena style).

**Common follow-ups.**
- "Why is Chatbot Arena trusted?" → Crowdsourced pairwise from real users on real prompts; large N; resistant to gaming compared to closed benchmarks.
- "What's the Elo system in Arena?" → Pairwise win/loss → ratings via the same algorithm used for chess.

**Common mistakes.**
- Reporting single-grade scores without inter-rater agreement.
- Pairwise without position randomization.

**References.**
- [Zheng et al. — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"](https://arxiv.org/abs/2306.05685) — pairwise eval.
- [LMSYS Chatbot Arena](https://lmarena.ai/) — the production pairwise leaderboard.

---

### Q: Walk me through benchmark suites — MMLU, MMLU-Pro, GPQA, GSM8K, MATH, HumanEval, MBPP, SWE-bench, ARC-AGI. What does each measure?

**Category:** concept
**Difficulty:** mid
**Tags:** [benchmarks, mmlu, gpqa, humaneval]

**Short answer.** **MMLU**: 57-subject multiple-choice knowledge test. **MMLU-Pro**: harder, contamination-resistant successor with 10 choices. **GPQA**: graduate-level science questions. **GSM8K**: grade-school math word problems. **MATH**: competition math. **HumanEval**: Python function-completion from docstrings. **MBPP**: similar, mostly basic problems. **SWE-bench**: real-world code-patch tasks. **ARC-AGI**: visual-grid reasoning, resistant to scale.

**Expansion / why this is the answer.**
- **MMLU** (Hendrycks et al. 2020): 15.9k multiple-choice questions across 57 subjects (humanities, STEM, professional). The headline "knowledge" benchmark for years. Saturating in 2024.
- **MMLU-Pro** (Wang et al. 2024): 12k harder questions, 10 choices instead of 4; designed to resist memorization and reduce ceiling effects.
- **GPQA** (Rein et al. 2023): "Google-proof" graduate-level biology/physics/chemistry; 448 questions; designed so experts get ~65% and non-experts ~34%.
- **GSM8K** (Cobbe et al. 2021): 8.5k grade-school math word problems; tests multi-step reasoning. Largely saturated.
- **MATH** (Hendrycks et al. 2021): 12k competition math problems (AMC, AIME, USAMO levels). Still challenging.
- **HumanEval** (Chen et al. 2021, OpenAI): 164 hand-written Python problems with unit tests. The original code benchmark.
- **MBPP** (Austin et al. 2021): 1k basic Python problems.
- **HumanEval+ / MBPP+** (Liu et al. 2023): augmented test cases revealed many "passing" solutions were wrong; harder.
- **SWE-bench / Verified / Live**: real GitHub issues; the modern coding-agent benchmark (see Topic 6).
- **ARC-AGI** (Chollet 2019; ARC-AGI-2 2024): visual-grid reasoning; resistant to scale; designed to measure "true" reasoning.
- **BIG-Bench Hard (BBH)** (Suzgun et al. 2022): 23 hardest BIG-Bench subtasks; reasoning-focused.
- **AGIEval** (Zhong et al. 2023): standardized human exams (SAT, LSAT, GRE).
- **What an interviewer wants**: that you know each benchmark's *purpose* and *limit*, not all the numbers.

**Common follow-ups.**
- "Which of these is the most reliable signal in 2026?" → No single one. GPQA + SWE-bench Verified + MMLU-Pro + LiveCodeBench + Arena Elo gives a more honest picture than any single number.
- "What's LiveCodeBench?" → Coding bench using freshly-collected LeetCode-style problems with date filters; resistant to contamination.

**Common mistakes.**
- Quoting MMLU as the universal score; the field has moved on.
- Treating GSM8K as a strong reasoning measure; it's largely saturated and frequently contaminated.

**References.**
- [Hendrycks et al. — MMLU](https://arxiv.org/abs/2009.03300) — MMLU.
- [Wang et al. — MMLU-Pro](https://arxiv.org/abs/2406.01574) — MMLU-Pro.
- [Rein et al. — GPQA](https://arxiv.org/abs/2311.12022) — GPQA.
- [Cobbe et al. — GSM8K](https://arxiv.org/abs/2110.14168) — GSM8K.
- [Hendrycks et al. — MATH](https://arxiv.org/abs/2103.03874) — MATH.
- [Chen et al. — HumanEval](https://arxiv.org/abs/2107.03374) — HumanEval.
- [Jimenez et al. — SWE-bench](https://arxiv.org/abs/2310.06770) — SWE-bench.
- [Chollet — ARC](https://arxiv.org/abs/1911.01547) — ARC.

---

### Q: What's the difference between offline and online evaluation? How do you bridge them?

**Category:** concept
**Difficulty:** mid
**Tags:** [offline-eval, online-eval, shadow-traffic, ab-testing]

**Short answer.** **Offline**: evaluate on a fixed dataset, in a sandbox; fast, cheap, repeatable. **Online**: measure in production — actual users, real traffic, the business metric. Offline can mislead (proxy metric doesn't track the business metric); online is the truth but slow and risky. Bridge with **shadow traffic** (route real queries to the new model, log outputs, don't show users), **gated rollout** (small % of users), and **A/B testing** with statistical power calculations.

**Expansion / why this is the answer.**
- **Offline strengths**:
  - Reproducible, version-controlled.
  - Fast iteration on model changes.
  - Cheap (no user exposure).
- **Offline limits**:
  - Eval set may not match real traffic distribution.
  - Metric is a proxy; the business metric might not move.
- **Online strengths**:
  - Ground truth: did users like it? Did they convert? Did support tickets drop?
- **Online costs**:
  - Risk: bad model exposed to users.
  - Slow: needs traffic and time.
  - Statistical: small effects need many samples.
- **Bridging techniques**:
  - **Shadow traffic**: run the new model in parallel; compare outputs; never expose to users. Cheap online signal, no risk.
  - **Gated rollout**: 1% → 10% → 50% → 100% over weeks. Monitor key metrics at each step.
  - **A/B test**: randomly assign users to old/new; measure delta. Need sample size from power analysis.
  - **Holdout group**: always keep a small % on the old model as a long-running baseline.
- **Bridging signal quality**:
  - Build offline metrics that correlate with online ones. If your offline metric doesn't predict online wins, find a better metric.

**Common follow-ups.**
- "What's sample-size for an A/B test?" → Depends on baseline rate and the effect to detect. Order-of-magnitude formula: `n ≈ 16 · p(1−p) / Δ²` per arm for a binary metric.
- "What's CUPED?" → Variance-reduction technique using pre-experiment data to reduce sample size needed.

**Common mistakes.**
- Optimizing offline forever without an online proof.
- Reporting an online win without checking the guardrails (latency, cost, error rate).

**References.**
- [Kohavi, Tang, Xu — *Trustworthy Online Controlled Experiments*](https://experimentguide.com/) — canonical text.
- [Deng et al. — "CUPED"](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf) — variance reduction.

---

### Q: How would you build an LLM-as-judge eval that's calibrated to humans?

**Category:** concept
**Difficulty:** senior
**Tags:** [llm-judge-calibration, eval-design, gold-data]

**Short answer.** (1) Collect a **gold set** of `(input, output, human label)` for the task. (2) Run the LLM judge against the gold set; compute **agreement rate** with humans. (3) **Iterate the judge prompt** until agreement is high enough for the use (often >85%). (4) **Spot-check periodically** in production; if agreement drifts, retune. The judge is only usable when its agreement with humans is competitive with inter-human agreement on the same task.

**Expansion / why this is the answer.**
- **Step 1: gold set**:
  - 100–500 examples is typical; more for rare/edge cases.
  - Multiple human raters per example; report inter-rater agreement (Cohen's κ or Krippendorff's α).
  - If humans disagree a lot, your task is subjective and the judge will be at least as bad.
- **Step 2: agreement metrics**:
  - Pairwise agreement: % matching judgments.
  - Cohen's κ: agreement above chance.
  - Spearman / Kendall correlation for rated scales.
- **Step 3: prompt engineering**:
  - **Rubric explicitness**: judge's prompt should describe each criterion and how to weight them.
  - **Examples**: include a few worked examples in the prompt.
  - **Chain-of-thought**: have the judge reason before scoring.
  - **Bias mitigation**: randomize order (pairwise), include calibration anchors.
- **Step 4: drift monitoring**:
  - Sample N judge decisions per week; have humans re-grade.
  - Track agreement over time; retune when it dips.
- **Limits**:
  - Some tasks have low human agreement (subjective); the judge can't beat that.
  - Domain expertise the judge model lacks → low ceiling; bring in domain LLMs or humans.

**Common follow-ups.**
- "What if judge-human agreement is only 70%?" → Either task is too subjective (need humans), or the judge needs a stronger model or better prompt.
- "Can you fine-tune the judge?" → Yes — train on `(input, output, human label)` pairs. Process-reward models (Lightman et al.) do this for math step-grading.

**Common mistakes.**
- Skipping the calibration step.
- Using a single human grader to define "ground truth."

**References.**
- [Zheng et al. — MT-Bench](https://arxiv.org/abs/2306.05685) — judge calibration.
- [Lightman et al. — "Let's Verify Step by Step"](https://arxiv.org/abs/2305.20050) — process reward models.

---

### Q: How do you eval an LLM's safety / refusal behavior?

**Category:** concept
**Difficulty:** mid
**Tags:** [safety-eval, red-team, refusal]

**Short answer.** Combination of: (a) **harmful-prompt suites** (HarmBench, AdvBench) — measure refusal rate on disallowed categories; (b) **over-refusal sets** (XSTest, OR-Bench) — measure benign queries that wrongly get refused; (c) **jailbreak / red-team** suites measuring resistance to adversarial prompts; (d) **calibrated severity** for high-stakes domains (bio, weapons). Frontier labs publish "system cards" describing methodology. Crucially, the **two-sided** axis — refuse harmful AND don't refuse benign — must be measured.

**Expansion / why this is the answer.**
- **The two-sided axis**:
  - **Refusal of harmful**: the model should refuse "how do I make a bomb?"
  - **Over-refusal of benign**: the model should not refuse "what's the chemistry behind exothermic reactions?"
  - Optimizing one alone makes the other worse.
- **Benchmarks**:
  - **HarmBench** (Mazeika et al. 2024): standardized harmful-prompt + automated attack suite.
  - **AdvBench** (Zou et al. 2023): adversarial behavior dataset; widely cited.
  - **XSTest** (Röttger et al. 2023): "exaggerated safety" tests; benign queries phrased similarly to harmful ones.
  - **OR-Bench** (Cui et al. 2024): over-refusal benchmark.
  - **JailbreakBench** (Chao et al. 2024): standardized attack methodology.
- **Frontier-lab practice**:
  - System cards (OpenAI, Anthropic) document categories, refusal rates, dual-use evals.
  - Red-team teams (in-house + external) attempt jailbreaks pre-launch.
  - Cat-3 risk categories (bio, chemical, weapons, CSAM) often have category-specific evals with very low tolerated false-negative rates.
- **Measurement design**:
  - Auto-grading: a classifier or LLM judge predicts "did it refuse appropriately?"
  - Human spot-checks on the disagreements.

**Common follow-ups.**
- "What's adversarial suffix attack?" → Zou et al. 2023 — gradient-optimized suffix appended to a harmful prompt elicits the harmful response on many models.
- "What's a 'capability eval' for dangerous content?" → Categorized prompts attempting to elicit *correct* harmful info, not just any harmful info; harder for bio/chem.

**Common mistakes.**
- Reporting refusal rate without over-refusal rate.
- Static benchmarks; jailbreaks evolve.

**References.**
- [Mazeika et al. — "HarmBench"](https://arxiv.org/abs/2402.04249) — HarmBench.
- [Röttger et al. — "XSTest"](https://arxiv.org/abs/2308.01263) — over-refusal.
- [Zou et al. — "Universal and Transferable Adversarial Attacks"](https://arxiv.org/abs/2307.15043) — adversarial suffix.

---

### Q: Why are model performance numbers often hard to reproduce? Walk me through the reproducibility tax.

**Category:** concept
**Difficulty:** senior
**Tags:** [reproducibility, eval-methodology, prompting]

**Short answer.** Same model + same benchmark can yield different scores because of: **prompt template** differences, **few-shot exemplar** choices, **decoding parameters** (temperature, top-p), **scoring method** (exact match vs. parsed vs. LLM-judge), **chain-of-thought yes/no**, **system prompt** variations, **model version drift** (closed APIs change), and **eval framework** differences (HELM vs. lm-eval-harness vs. internal). Always pin every dial; report the framework.

**Expansion / why this is the answer.**
- **Sources of variance**:
  1. Prompt template / formatting (`Q: ... A:` vs. `Question: ... Answer:`).
  2. Few-shot count and selection.
  3. Decoding: temperature, top-p, max-tokens.
  4. Parsing: "the answer is 7" vs. "7" — exact-match vs. extractor.
  5. CoT: appending "Let's think step by step" changes scores 5–20pp.
  6. System prompt variations.
  7. Model-version drift on hosted APIs (`gpt-4` → `gpt-4-0613` → ...).
  8. Eval framework — HELM, lm-eval-harness, OpenAI Evals, internal forks all differ.
- **Real example**: lm-eval-harness gets different MMLU scores than the published HELM scores for the same model, because the prompt formats and answer extraction differ.
- **Best practice**:
  - Always cite framework + version + prompt template + decoding params.
  - Don't take "X% on MMLU" claims at face value; check the methodology.
- **2026 norm**: model cards / tech reports now disclose framework + prompt; community-trusted leaderboards (Arena, Open LLM Leaderboard, EvalPlus) standardize.

**Common follow-ups.**
- "Why don't all benchmarks publish their official prompts?" → Some do; many don't, leaving room for divergence.
- "What's the Open LLM Leaderboard?" → HF's standardized eval on a fixed set of benchmarks with a fixed harness; widely used for open-model comparisons.

**Common mistakes.**
- Comparing scores across papers without checking framework.
- Treating one prompt's score as "the" score.

**References.**
- [Liang et al. — "Holistic Evaluation of Language Models" (HELM)](https://arxiv.org/abs/2211.09110) — HELM methodology.
- [Gao et al. — lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — community-standard harness.

---

### Q: What's the difference between intrinsic and extrinsic evaluation? Examples.

**Category:** concept
**Difficulty:** intro
**Tags:** [intrinsic, extrinsic, eval-types]

**Short answer.** **Intrinsic**: evaluate a property of the model in isolation (perplexity, embedding similarity, classification accuracy on a probe). **Extrinsic**: evaluate the model on the downstream task you care about (does it close more tickets? improve search recall? raise revenue?). Intrinsic is cheap and fast; extrinsic is expensive but actually answers the business question. Both have roles: intrinsic for tight iteration, extrinsic for proving impact.

**Expansion / why this is the answer.**
- **Intrinsic examples**:
  - Perplexity on held-out text (LM quality).
  - MTEB score (embedding quality).
  - MMLU/MATH/HumanEval (capability).
- **Extrinsic examples**:
  - Customer support: deflection rate, CSAT.
  - Coding assistant: PRs merged, time-to-fix.
  - Search: click-through rate, satisfaction.
- **The classical NLP framing** (Jurafsky & Martin): intrinsic metrics evaluate a component; extrinsic metrics evaluate the component's contribution to a larger system.
- **Why this matters**: intrinsic metrics can improve while extrinsic stagnates (or vice versa). Don't optimize only intrinsic.
- **Decision discipline**:
  - Iterate on intrinsic.
  - Validate with extrinsic before launching.
  - Maintain both axes long-term.

**Common follow-ups.**
- "Is perplexity a good intrinsic metric for an instruction-tuned model?" → Less useful — it measures the prediction distribution; extrinsic capability matters more after SFT/RLHF.
- "Can MMLU correlate with extrinsic metrics?" → Loosely; capability benchmarks correlate with broad capability, but specific extrinsic outcomes depend on the application.

**Common mistakes.**
- Reporting intrinsic and assuming extrinsic follows.
- Reporting extrinsic without an intrinsic signal — slower iteration.

**References.**
- [Jurafsky & Martin — *Speech and Language Processing*, §B](https://web.stanford.edu/~jurafsky/slp3/) — intrinsic vs extrinsic.

---

### Q: How does Chatbot Arena work, and why do people trust it?

**Category:** concept
**Difficulty:** mid
**Tags:** [chatbot-arena, elo, pairwise, lmsys]

**Short answer.** Chatbot Arena (LMSYS) is a crowdsourced pairwise comparison platform: users submit a prompt, get back two anonymous model responses, pick the better one. Pairwise votes are aggregated via the Bradley-Terry / Elo model into a public leaderboard. People trust it because it (a) uses real user prompts, not curated benchmarks; (b) is resistant to overfitting because models can't game prompts they don't see; (c) measures *user preference*, not a proxy capability; (d) has a very large sample size (millions of votes).

**Expansion / why this is the answer.**
- **Mechanism**:
  1. User enters a prompt.
  2. Two anonymized model responses shown.
  3. User picks (A wins / B wins / tie / both bad).
  4. Vote aggregated; identities revealed.
- **Ranking math**:
  - Bradley-Terry model: `P(A beats B) = σ(r_A − r_B)`.
  - From pairwise outcomes, fit per-model ratings `r_i`.
  - Equivalent to a chess-style Elo system with logistic scoring.
- **Why people trust it**:
  - **Real users, real prompts**: distribution matches actual use.
  - **Contamination-resistant**: prompts are user-supplied; models can't memorize.
  - **Sample size**: hundreds of thousands to millions of votes per model.
  - **Confidence intervals**: error bars published per model.
- **Caveats**:
  - **User-skew bias**: heavy ChatGPT/Claude users tend to favor those models' styles.
  - **English-heavy**: less reliable for non-English performance.
  - **Length bias**: long answers tend to win — adjusted scoring (Arena Hard, Arena Hard-Auto) controls for this.
  - **No ground truth**: pure preference, not correctness.
- **Variants**:
  - **Arena Hard** (Li et al. 2024): a curated subset; emphasizes harder prompts.
  - **Style Control**: adjustments for length / markdown / refusal style.

**Common follow-ups.**
- "How do they prevent gaming?" → Pairwise + anonymity; rate limiting; bot detection.
- "Is the ranking transitive?" → Approximately under Bradley-Terry; not strictly transitive in practice (rock-paper-scissors triples can exist).

**Common mistakes.**
- Treating Arena Elo as an absolute capability measure; it's a preference measure.
- Comparing Arena ranks across very different model classes (vision vs. text) — different leaderboards.

**References.**
- [Zheng et al. — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"](https://arxiv.org/abs/2306.05685) — Arena methodology.
- [LMSYS Chatbot Arena](https://lmarena.ai/) — primary.

---

### Q: How do you evaluate creative tasks (writing, code-style)?

**Category:** concept
**Difficulty:** senior
**Tags:** [creative-eval, subjective, llm-as-judge]

**Short answer.** No single metric works. The toolkit: (a) **pairwise human preference** on a rubric (clearer, more interesting, more on-topic); (b) **LLM-as-judge pairwise** calibrated to the human set; (c) **specific quality dimensions** scored separately (coherence, factuality, style match); (d) **task-specific eval** (does the joke land? does the code compile?). Combine; report multiple dimensions; never reduce to one number for creative tasks.

**Expansion / why this is the answer.**
- **Why a single metric fails**:
  - Multiple competing dimensions: creativity vs. accuracy, conciseness vs. depth.
  - Subjective: different readers prefer different things.
  - No "ground truth" for an open creative prompt.
- **Methods**:
  - **Pairwise human**: most reliable for relative ranking. Slow, expensive. Calibrate inter-rater agreement.
  - **Single-grade rubric**: faster; rater drift risk.
  - **LLM-as-judge**: cheap; calibrate against the human set.
  - **Reference-based**: when "good answers" exist, compare against reference (Bleu / ROUGE for translation/summarization, with known limits).
  - **Task-specific**: a joke is funny if it lands with the audience; code style passes a linter / matches the team's conventions.
- **For code-style**:
  - Linter / formatter compliance (deterministic).
  - PR-acceptance rate.
  - Code-review scores (LLM-judge or human).
- **For creative writing**:
  - Multi-dimensional: coherence, voice, originality, factuality.
  - Avoid average-into-one-number; report each dim.
- **Failure modes**:
  - LLM judges favor verbose / hedging output for "creative" tasks — adversarial to creative writing's compactness.
  - Single rater drift.

**Common follow-ups.**
- "How would you eval poetry?" → Probably mostly human; LLM judges aren't reliable on aesthetic judgments.
- "How do you avoid LLM judges hallucinating quality?" → Reference-based grading where possible; tight rubrics; human calibration.

**Common mistakes.**
- Reducing creative quality to one score.
- Trusting LLM-as-judge on a task with no human calibration.

**References.**
- [Chiang et al. — "Can Large Language Models Be an Alternative to Human Evaluations?"](https://arxiv.org/abs/2305.01937) — LLM-as-judge for subjective tasks.
- [Zheng et al. — MT-Bench](https://arxiv.org/abs/2306.05685) — judge calibration.

---

### Q: How do you evaluate an agent's tool-use accuracy?

**Category:** concept
**Difficulty:** mid
**Tags:** [agent-eval, tool-use-accuracy]

**Short answer.** Measure at multiple granularities: **(1) tool-selection accuracy** (right tool given the query?), **(2) argument accuracy** (correct args / types), **(3) trajectory accuracy** (correct sequence of calls), **(4) end-to-end task success**. Use a labeled eval set with ground-truth `(prompt, expected tool calls)` traces. Trace-level metrics distinguish "right tool, wrong args" from "wrong tool" — the failure modes have different fixes.

**Expansion / why this is the answer.**
- **Eval set construction**:
  - 100–500 examples per use case.
  - Each: `(prompt, expected_tool_call_sequence, expected_outcome)`.
  - Include negative examples (no tool call should be made).
- **Metrics**:
  - **Tool-selection accuracy**: `correct_tool / total_calls`. Confusion matrix tells you which tools the model mixes up.
  - **Argument accuracy**: structured comparison of expected vs. predicted args. Exact-match for IDs, semantic for free-text.
  - **Trajectory F1**: treat the expected call-sequence as a set or ordered list; measure F1.
  - **End-to-end success**: did the task succeed? Most important; depends on tool-call accuracy + downstream tool execution + response generation.
  - **No-call-when-not-needed precision**: % of "no tool call" prompts where the agent correctly didn't call a tool.
- **Tooling**:
  - **τ-bench**: standardized multi-turn agent eval with structured tools.
  - **Tool-use evals in OpenAI/Anthropic**: internal evals on tool-call format compliance.
- **Trace inspection**:
  - Log every tool call + args + outcome.
  - Failed-trajectory analysis: where in the sequence did the agent go wrong?

**Common follow-ups.**
- "How do you handle multiple valid trajectories?" → Score against the *set* of valid trajectories; or use end-to-end success as the headline metric and tool-call accuracy as a diagnostic.
- "What's tool-call latency in the budget?" → Each LLM call adds 1–5s; tool execution varies; budget per-task.

**Common mistakes.**
- Only measuring end-to-end success; can't diagnose where it broke.
- Measuring on synthetic prompts that don't match real distribution.

**References.**
- [Yao et al. — "τ-bench"](https://arxiv.org/abs/2406.12045) — tool-use eval.
- [BFCL — Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — function-calling benchmark.

---

### Q: What's the difference between capability eval and user-experience eval?

**Category:** concept
**Difficulty:** mid
**Tags:** [capability-eval, ux-eval, evaluation-types]

**Short answer.** **Capability eval**: does the model *can* do X? Tested on curated benchmarks (MMLU, HumanEval, GPQA). Holds the prompt format and difficulty constant. **UX eval**: does the model *help users* in a real product context? Tested on real or realistic prompts, with metrics that reflect end-user value (deflection, CSAT, task completion, latency). A model can be a capability beast and a UX disaster (e.g. correct but verbose / unfriendly / slow).

**Expansion / why this is the answer.**
- **Capability eval**:
  - Static benchmarks; controlled conditions.
  - "Could the model, in principle, do this?"
  - Examples: MMLU, MATH, GSM8K, HumanEval, SWE-bench.
  - Limits: synthetic prompts; benchmark contamination; doesn't reflect deployment.
- **UX eval**:
  - Real or realistic prompts.
  - "Does this product help its users?"
  - Examples: A/B test deflection rate, CSAT, task completion, time-to-resolve, repeat-question rate.
  - Includes latency, cost, refusal-rate, conversational tone.
- **Gaps**:
  - A capability-strong model may UX-fail because of:
    - Verbose responses (users prefer concise).
    - Over-refusal (overly cautious tone).
    - Bad formatting / no markdown.
    - High latency.
- **In a hiring loop**: an interviewer asking about a product is likely asking about UX eval; an interviewer asking about a model release is likely asking about capability eval. Distinguish.
- **Modern eval frameworks** often combine both — capability suite + user-trajectory simulation.

**Common follow-ups.**
- "What's an example of a capability-strong, UX-weak model?" → Some open-weight models that score well on MMLU but are verbose / unhelpful in real chat — without instruction tuning + preference shaping.
- "How do you bridge them?" → Hold-out user-trajectory eval set; sample real prompts and evaluate.

**Common mistakes.**
- Citing capability eval as proof of product readiness.
- Skipping capability eval and only running UX (misses fundamental quality regressions).

**References.**
- [Bowman & Dahl — "What Will it Take to Fix Benchmarking in NLU?"](https://arxiv.org/abs/2104.02145).
- [Liang et al. — HELM](https://arxiv.org/abs/2211.09110) — holistic eval framework.

---

### Q: How would you evaluate an LLM for medical or legal accuracy?

**Category:** concept
**Difficulty:** senior
**Tags:** [domain-specific-eval, medical, legal, expert-eval]

**Short answer.** Domain-specific eval needs **(1) expert-labeled ground truth** (medical: board-certified physician annotations; legal: licensed attorney annotations), **(2) domain-specific benchmarks** (MedQA, USMLE-style sets, LegalBench), **(3) explicit safety criteria** (no false certainty on medical advice; flag when to escalate), and **(4) reference-based grading** by domain experts rather than LLM judges. Closed-API capability scores are not enough; you need to verify in your specific use case with experts.

**Expansion / why this is the answer.**
- **Why generic eval fails**:
  - MMLU includes medical questions but isn't a medical accuracy test.
  - Domain-specific edge cases (drug interactions, jurisdiction-specific law) are rare in general benchmarks.
- **Domain benchmarks**:
  - **Medical**: MedQA (USMLE), MedMCQA, PubMedQA, MultiMedQA.
  - **Legal**: LegalBench (Guha et al. 2023), CaseHOLD, contract review datasets.
  - **Financial**: FinanceBench.
- **Expert annotation**:
  - Sample model outputs on real / realistic prompts.
  - Have domain experts grade for: factual accuracy, safety (e.g. "did it correctly say 'see a doctor'"), citation accuracy, completeness.
  - Inter-rater agreement metric.
- **Safety criteria** (medical example):
  - Model should never give a treatment recommendation that requires diagnosis.
  - Should consistently recommend professional consultation for serious symptoms.
  - Should not hallucinate drug dosages or interactions.
- **LLM-as-judge limits**:
  - On domain-specific accuracy, the judge needs to be at least as competent as the student. For specialized domains, that may mean a different LLM, or no LLM judge at all.
- **Regulatory**:
  - In medical: FDA software-as-a-medical-device implications.
  - In legal: unauthorized practice of law boundaries.

**Common follow-ups.**
- "Can you use Med-PaLM as a judge?" → Med-PaLM 2 has demonstrated USMLE-passing performance; using it as a judge is possible but doesn't substitute for human review on safety-sensitive outputs.
- "How often do you re-evaluate?" → Monthly to quarterly; every model swap; whenever the use case expands.

**Common mistakes.**
- Trusting general LLM-as-judge for medical / legal accuracy.
- Treating MMLU score as a domain-specific competency proxy.

**References.**
- [Singhal et al. — "Med-PaLM"](https://arxiv.org/abs/2212.13138) — medical LLM eval.
- [Guha et al. — "LegalBench"](https://arxiv.org/abs/2308.11462) — legal eval benchmark.
- [Jin et al. — "MedQA"](https://arxiv.org/abs/2009.13081) — medical QA benchmark.

---

### Q: What is statistical power in A/B testing, and how do you choose sample size?

**Category:** concept
**Difficulty:** mid
**Tags:** [ab-testing, power, sample-size, statistics]

**Short answer.** Statistical power = `P(detect effect | effect exists)`. Typical target: 80%. Sample size depends on (a) baseline rate, (b) minimum detectable effect, (c) variance, (d) α (significance level, typically 0.05). Rule-of-thumb formula for binary outcome: `n ≈ 16 · p(1-p) / Δ²` per arm for 80% power, 5% significance, two-tailed. Under-powered tests miss real effects ("the metric didn't move" can mean "we didn't have enough samples").

**Expansion / why this is the answer.**
- **The four knobs**:
  - **α**: false-positive rate (typically 0.05).
  - **β**: false-negative rate (1 - power; typically 0.20).
  - **Effect size**: the minimum effect you care about (MDE — minimum detectable effect).
  - **Variance**: how noisy the metric is.
- **Sample-size formulas** (per arm):
  - **Binary outcome** (e.g. conversion rate): `n ≈ 16 · p(1-p) / Δ²` for power 80%, α 0.05.
  - **Continuous outcome**: `n ≈ 16 · σ² / Δ²` where σ is per-user std dev.
- **What if you're under-powered**:
  - Can't conclude no-effect; can only conclude "didn't detect."
  - Either: collect more data; or accept higher false-negative risk.
- **Variance reduction techniques**:
  - **CUPED** (Deng et al. 2013): use pre-experiment data to reduce variance — same power at smaller n.
  - **Stratification**: split by user segments; reduces between-segment variance.
- **Common gotchas**:
  - **Peeking**: looking at the metric daily; inflates false-positive rate. Use sequential testing or set a fixed end date.
  - **Multiple comparisons**: many metrics → Bonferroni or FDR correction.
  - **Novelty effect**: short-term lift on UI changes that fades.
- **Practical example**:
  - Baseline conversion 5%; want to detect a 5% relative lift (Δ_abs = 0.25 pp).
  - `n ≈ 16 · 0.05 · 0.95 / (0.0025)² ≈ 121,600` per arm.
  - Two arms: 243k users total. Need a lot of traffic.

**Common follow-ups.**
- "What's CUPED?" → Variance-reduction technique using a pre-experiment covariate; the same A/B at smaller n.
- "How do you handle multiple metrics?" → Pre-register the primary metric; treat others as secondary; correct for multiple testing.

**Common mistakes.**
- Concluding "no effect" from an under-powered test.
- Peeking at results during the test.

**References.**
- [Kohavi, Tang, Xu — *Trustworthy Online Controlled Experiments*](https://experimentguide.com/) — canonical reference.
- [Deng et al. — "CUPED"](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf).

---
