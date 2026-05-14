# ML & DL Fundamentals — questions

The fundamentals layer. Every entry conforms to the schema in [CONTRIBUTING.md](../../CONTRIBUTING.md#the-qa-entry-schema).

---

### Q: Explain the bias–variance tradeoff. What does each end of the spectrum look like in practice?

**Category:** concept
**Difficulty:** intro
**Tags:** [bias-variance, generalization, regularization]

**Short answer.** Bias is error from a model too simple to capture the signal (underfitting); variance is error from a model too sensitive to noise in the training set (overfitting). The classical decomposition of expected squared error is `bias² + variance + irreducible noise`, and reducing one usually increases the other for a fixed-capacity learner — so the tradeoff is the lever you tune (via regularization, model size, training data) to minimize total error.

**Expansion / why this is the answer.**
- The decomposition (Geman, Bienenstock & Doursat 1992; covered in Bishop §3.2): `E[(y − f̂(x))²] = (E[f̂(x)] − f(x))² + E[(f̂(x) − E[f̂(x)])²] + σ²`.
- High bias in practice: linear regression on a non-linear signal; predictions are systematically off in the same direction across training sets.
- High variance in practice: deep tree with no depth limit on small data; predictions swing wildly if you re-train on a slightly different sample.
- The textbook curves (training error monotone-decreasing, validation error U-shaped) are the visual.
- **Modern caveat.** With over-parameterized models and modern regularization (deep nets, LLMs), the classical U-curve is replaced by the **double descent** phenomenon (Belkin et al. 2019, Nakkiran et al. 2020): generalization error decreases, increases at the interpolation threshold, then decreases again as capacity grows further. Mention this if asked about deep nets specifically.

**Common follow-ups.**
- "How would you diagnose whether your model is bias-limited or variance-limited?" → Compare training-loss vs. val-loss: gap small + both high = bias; gap large = variance.
- "How do you reduce variance?" → More data, regularization, ensembling, smaller capacity.
- "Where does double descent fit?" → It complicates "more parameters = more variance" intuition for over-parameterized regimes.

**Common mistakes.**
- Saying "complex models always have high variance" — true classically, broken by double descent.
- Conflating bias-variance with under/overfitting without naming the connection.
- Forgetting the irreducible-noise term entirely.

**References.**
- [Bishop, *Pattern Recognition and Machine Learning*, §3.2](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — the canonical derivation.
- [Belkin, Hsu, Ma, Mandal — "Reconciling modern machine-learning practice and the classical bias–variance trade-off"](https://www.pnas.org/doi/10.1073/pnas.1903070116) — the double-descent paper.
- [Nakkiran et al. — "Deep Double Descent"](https://arxiv.org/abs/1912.02292) — deep-net version.

---

### Q: When would you use L1 versus L2 regularization?

**Category:** concept
**Difficulty:** intro
**Tags:** [regularization, sparsity]

**Short answer.** L1 (`Σ|w|`) drives weights exactly to zero — use it when you want sparsity or feature selection. L2 (`Σw²`) shrinks weights smoothly toward zero without zeroing them — use it when you want stability and don't need sparsity. ElasticNet combines both.

**Expansion / why this is the answer.**
- Geometry intuition (Hastie/Tibshirani/Friedman, ESL §3.4): the L1 constraint region is a diamond with corners on the axes; the optimal point typically hits a corner → sparsity. The L2 region is a ball; the optimal point typically hits a non-axis point → small but non-zero weights.
- L1 is **non-differentiable at zero**, so optimization uses sub-gradient or proximal methods (ISTA/FISTA, coordinate descent).
- L2 has a closed-form Bayesian interpretation as a Gaussian prior on weights; L1 as a Laplace prior.
- In deep learning, **L2 is the dominant choice** (often called weight decay; technically equivalent to L2 only for SGD, and not equivalent for Adam — hence AdamW which decouples weight decay from the gradient update).
- For feature selection in classical ML on tabular data with many irrelevant features, **L1 (Lasso) is the standard**.

**Common follow-ups.**
- "What's the difference between L2 and weight decay in Adam?" → They diverge; AdamW (Loshchilov & Hutter 2019) decouples them, which is why AdamW is the default in modern LLM training.
- "Why does L1 produce sparsity but L2 doesn't?" → The geometry argument plus the subgradient at zero containing zero.
- "What's ElasticNet?" → `α·L1 + (1−α)·L2`; balances sparsity with stability.

**Common mistakes.**
- Saying L2 produces sparsity (it shrinks, it does not zero).
- Forgetting the AdamW distinction in DL contexts.
- Confusing the **regularization** L1/L2 with the **norm** Lp generally (the names overload).

**References.**
- [Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, §3.4](https://hastie.su.domains/ElemStatLearn/) — the canonical treatment.
- [Loshchilov & Hutter — "Decoupled Weight Decay Regularization" (AdamW)](https://arxiv.org/abs/1711.05101) — why L2 ≠ weight decay in Adam.
- [Tibshirani — "Regression Shrinkage and Selection via the Lasso"](https://www.jstor.org/stable/2346178) — the original Lasso paper.

---

### Q: When would you choose cross-entropy loss over MSE for a classification problem?

**Category:** derivation
**Difficulty:** intro
**Tags:** [loss-functions, cross-entropy, MSE]

**Short answer.** Almost always — cross-entropy is what you derive from maximum likelihood for categorical targets with a softmax output, and its gradient behaves well (large when the prediction is confidently wrong). MSE on a softmax output produces tiny gradients at confident-wrong predictions, slowing learning dramatically.

**Expansion / why this is the answer.**
- MLE derivation: for one-hot label `y` and softmax output `p`, the per-example log-likelihood is `log p_y`; minimizing negative log-likelihood gives `−Σ y_i log p_i`, which is cross-entropy.
- Gradient with softmax + cross-entropy: `∂L/∂z_i = p_i − y_i` (clean linear-in-error gradient; this is why the combo is the default).
- Gradient with softmax + MSE: includes a `p_i(1−p_i)` factor that vanishes when the model is confident-wrong (`p_correct ≈ 0`), so learning stalls.
- For binary classification, the equivalent is **binary cross-entropy** with sigmoid output.
- For **regression**, MSE is appropriate (it's MLE under Gaussian noise assumptions); MAE is appropriate under Laplacian noise / outlier robustness.

**Common follow-ups.**
- "Derive ∂(cross-entropy)/∂z for softmax." → Show the `p − y` result.
- "When would you ever use MSE for classification?" → Some calibration / regression-trick setups (e.g. label smoothing as a form of regression), but rarely as the headline loss.
- "What's focal loss and when do you use it?" → CE with an `(1−p)^γ` modulation; designed for severe class imbalance (Lin et al., 2017, RetinaNet paper).

**Common mistakes.**
- "MSE works fine for classification" — technically true, but learning stalls; this is the question.
- Forgetting label smoothing as an alternative interpretation.
- Treating cross-entropy as somehow arbitrary rather than derived from MLE.

**References.**
- [Bishop, *Pattern Recognition and Machine Learning*, §4.3.4](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — softmax + cross-entropy MLE derivation.
- [Goodfellow, Bengio, Courville — *Deep Learning*, §6.2](https://www.deeplearningbook.org/) — why cross-entropy is the default loss for classification.
- [Lin et al. — "Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002) — the class-imbalance variant.

---

### Q: Walk through SGD → Momentum → Adam → AdamW. Why did each step come about?

**Category:** concept
**Difficulty:** mid
**Tags:** [optimization, adam, adamw]

**Short answer.** SGD is "step in the direction of the gradient." Momentum adds an EMA of past gradients, smoothing the path through ravines. RMSProp/Adam additionally scale per-parameter by an EMA of squared gradients, giving each parameter its own effective learning rate. AdamW fixes a subtle bug where standard Adam's "weight decay" wasn't actually weight decay because the regularizer was being scaled by the same per-parameter denominator. AdamW is the default for modern LLM training.

**Expansion / why this is the answer.**
- **SGD**: `θ ← θ − η · g`. Sensitive to learning-rate choice; bounces around in narrow ravines.
- **SGD with Momentum** (Polyak / Nesterov): `v ← β v + g; θ ← θ − η v`. Smooths direction over time.
- **RMSProp** (Hinton, Coursera lectures): per-parameter scaling by EMA of squared gradients — `s ← β s + (1−β) g²; θ ← θ − η g / (√s + ε)`. Adaptive learning rate per parameter.
- **Adam** (Kingma & Ba 2014): RMSProp + momentum, with bias correction for the EMA cold start.
- **AdamW** (Loshchilov & Hutter 2019): in standard Adam, the L2 regularizer `λθ` was added to the gradient before the `/√s` rescaling, so parameters with small gradients ended up with effectively *more* regularization. AdamW applies `−η λ θ` *outside* the adaptive scaling, decoupling weight decay from the gradient update. This noticeably improves generalization, particularly for transformers, and is the modern default.
- For LLM training specifically, **AdamW + cosine LR schedule + warmup** is canonical (used in GPT-3, LLaMA, Mixtral, etc.).
- **Lion** (Chen et al. 2023) is an alternative — smaller memory footprint, competitive results on some benchmarks; not the default yet.

**Common follow-ups.**
- "Why does AdamW use more memory than SGD?" → It stores first and second moments per parameter (`m`, `v`) — roughly 2× model size in optimizer state.
- "What's bias correction in Adam?" → The EMA starts at 0; dividing by `1 − β^t` corrects the initial underestimate.
- "Why is the cosine schedule popular?" → Smooth decay to a low final LR; empirically improves final loss vs. linear decay (Loshchilov & Hutter 2017, SGDR).

**Common mistakes.**
- Saying "Adam is always better than SGD." On some vision tasks SGD with Nesterov momentum + careful schedule outperforms Adam — the Adam-is-default claim is for transformers/LLMs.
- Confusing weight decay with L2 in Adam.
- Forgetting bias correction.

**References.**
- [Kingma & Ba — "Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980) — Adam.
- [Loshchilov & Hutter — "Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101) — AdamW.
- [Loshchilov & Hutter — "SGDR: Stochastic Gradient Descent with Warm Restarts"](https://arxiv.org/abs/1608.03983) — cosine schedule.
- [Chen et al. — "Symbolic Discovery of Optimization Algorithms" (Lion)](https://arxiv.org/abs/2302.06675) — Lion.

---

### Q: What is the vanishing-gradient problem, and how do modern architectures avoid it?

**Category:** concept
**Difficulty:** intro
**Tags:** [vanishing-gradient, residuals, activations, initialization]

**Short answer.** In a deep network, gradients chained through many layers can shrink toward zero (vanish) or blow up (explode), making the early layers untrainable. Modern fixes: ReLU-family activations (so derivatives are 0 or 1, not sigmoid's ≤0.25), careful initialization (He/Kaiming for ReLU), residual connections (so gradient flows through identity shortcuts), and normalization layers (Batch / Layer / RMSNorm) that keep activations in a healthy range.

**Expansion / why this is the answer.**
- Mathematical source: the chain rule multiplies many Jacobians; if each has spectral norm ≪ 1, the product shrinks exponentially in depth.
- **Sigmoid** has max derivative 0.25 at zero, so 10-deep sigmoid stack already kills gradients.
- **ReLU** (`max(0, x)`) has derivative 0 or 1 — no compression. Modern variants: **GELU** (Hendrycks & Gimpel 2016), **SiLU/Swish**, **SwiGLU** (Shazeer 2020, used in PaLM and LLaMA).
- **He/Kaiming initialization** (He et al. 2015): scale weights by `√(2/fan_in)` for ReLU to preserve activation variance.
- **Residual connections** (He et al. 2015, ResNet): `y = F(x) + x` provides a gradient highway; in a transformer block, every sublayer is a residual.
- **Normalization**: LayerNorm (Ba et al. 2016) is the modern transformer default. RMSNorm (Zhang & Sennrich 2019) drops the mean-centering, used in LLaMA and many recent models.
- The exploding-gradient cousin is typically addressed by gradient clipping.

**Common follow-ups.**
- "Why pre-norm instead of post-norm in modern LLMs?" → Pre-norm (Xiong et al. 2020) gives a cleaner residual path; trains more stably at depth.
- "Why RMSNorm over LayerNorm?" → RMSNorm drops the mean-subtraction step; saves a small amount of compute with negligible quality cost; LLaMA uses it.

**Common mistakes.**
- Citing only one fix; modern nets use *all four* (activation, init, residuals, norm) — they're complementary.
- Saying "ReLU solved vanishing gradients" — necessary but not sufficient (still need residuals at depth ≥ ~30).

**References.**
- [He et al. — "Deep Residual Learning for Image Recognition" (ResNet)](https://arxiv.org/abs/1512.03385) — residual connections.
- [He et al. — "Delving Deep into Rectifiers" (He init)](https://arxiv.org/abs/1502.01852) — Kaiming initialization.
- [Ba, Kiros, Hinton — "Layer Normalization"](https://arxiv.org/abs/1607.06450) — LayerNorm.
- [Zhang & Sennrich — "Root Mean Square Layer Normalization"](https://arxiv.org/abs/1910.07467) — RMSNorm.
- [Xiong et al. — "On Layer Normalization in the Transformer Architecture"](https://arxiv.org/abs/2002.04745) — pre-norm vs. post-norm analysis.

---

### Q: BatchNorm vs. LayerNorm vs. RMSNorm — what does each normalize over, and where is each used?

**Category:** concept
**Difficulty:** mid
**Tags:** [normalization, batchnorm, layernorm, rmsnorm]

**Short answer.** BatchNorm normalizes across the batch dimension for each feature, requires a large enough batch, and has train/eval mode divergence (uses running stats at eval). LayerNorm normalizes across the feature dimension for each sample independently; batch-size-agnostic; the default in transformers. RMSNorm is LayerNorm minus the mean-subtraction step — slightly cheaper, used in LLaMA / Mistral / many modern LLMs. GroupNorm splits channels into groups and normalizes within each.

**Expansion / why this is the answer.**
- **BatchNorm** (Ioffe & Szegedy 2015): for a `(B, C, ...)` tensor, compute `μ, σ` across `B` for each channel; normalize; then learnable `γ, β`. Strong for CNNs at high batch sizes. **Failure modes**: small batches, sequence models (token positions don't correspond cleanly), and the eval/train divergence (it uses running averages at inference, which is a source of train–test gaps).
- **LayerNorm** (Ba et al. 2016): for a `(B, T, D)` tensor, compute `μ, σ` across `D` for each token; normalize; learnable `γ, β`. Batch-independent; works on a batch of 1. **The default in transformers** for this reason.
- **RMSNorm** (Zhang & Sennrich 2019): drop the mean-subtraction; `y = x / √(mean(x²) + ε) * γ`. Slightly cheaper; comparable quality; **used in LLaMA**.
- **GroupNorm** (Wu & He 2018): channel groups; useful when batch size is small (e.g., detection, segmentation).

**Common follow-ups.**
- "Why does LayerNorm replace BatchNorm in transformers?" → BatchNorm doesn't make sense per-token because each position has different statistics; batch-size sensitivity makes it brittle.
- "What's the placement question — pre-norm vs. post-norm?" → Modern LLMs are pre-norm: `x + Attn(Norm(x))`; gives more stable training at depth.

**Common mistakes.**
- Confusing "normalize across batch" with "normalize across the batch dimension." LayerNorm does *not* touch the batch axis — each sample is normalized independently.
- Forgetting that BatchNorm has different behavior at train vs. eval.
- Saying RMSNorm is "the same as LayerNorm" — it skips mean-centering, which is the entire difference.

**References.**
- [Ioffe & Szegedy — "Batch Normalization"](https://arxiv.org/abs/1502.03167) — BatchNorm.
- [Ba, Kiros, Hinton — "Layer Normalization"](https://arxiv.org/abs/1607.06450) — LayerNorm.
- [Zhang & Sennrich — "Root Mean Square Layer Normalization"](https://arxiv.org/abs/1910.07467) — RMSNorm.
- [Wu & He — "Group Normalization"](https://arxiv.org/abs/1803.08494) — GroupNorm.

---

### Q: When would you pick GBMs (XGBoost/LightGBM) over a neural network?

**Category:** concept
**Difficulty:** mid
**Tags:** [gbm, xgboost, tabular]

**Short answer.** On tabular data with mixed feature types and limited training samples (say <1M rows), GBMs are the empirical default — they handle missing values, non-linear feature interactions, and feature scales out of the box, and they beat or match neural nets on most tabular benchmarks (Grinsztajn et al. 2022). Pick a neural net when you have unstructured inputs (text, images, audio), very large data, or you need representation learning across modalities.

**Expansion / why this is the answer.**
- GBMs (XGBoost, LightGBM, CatBoost) handle: mixed categorical/numeric features, missing values natively (XGBoost / LightGBM both), monotonic constraints, scale invariance.
- The Grinsztajn et al. (2022) study formally benchmarks: GBMs beat tuned deep nets on a broad tabular benchmark, even after extensive tuning of the deep nets.
- Where neural nets win: text / images / audio (representation learning beats hand features), >10M samples with rich interactions, multi-modal data, transfer from pretrained features.
- Practical interview answer: "I'd start with a GBM baseline on tabular; bring out a deep model only if I have a specific reason (representation learning, scale, transfer, multimodality)."

**Common follow-ups.**
- "What are TabNet / FT-Transformer?" → Deep models targeted at tabular; closed the gap somewhat but typically still don't beat well-tuned GBMs on average.
- "Why does XGBoost outperform a random forest most of the time?" → GBMs are sequential / additive, fitting residuals, vs. RF's parallel bagging; the bias-reduction step generally yields stronger results when tuned.

**Common mistakes.**
- "Deep learning is always better with enough data" — false on tabular even at moderate scale.
- Conflating GBMs with random forests.

**References.**
- [Grinsztajn, Oyallon, Varoquaux — "Why do tree-based models still outperform deep learning on tabular data?"](https://arxiv.org/abs/2207.08815) — the canonical 2022 benchmark.
- [Chen & Guestrin — "XGBoost: A Scalable Tree Boosting System"](https://arxiv.org/abs/1603.02754) — XGBoost paper.
- [Ke et al. — "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) — LightGBM.

---

### Q: When is accuracy the wrong metric? Walk me through precision, recall, F1, ROC-AUC, PR-AUC.

**Category:** concept
**Difficulty:** intro
**Tags:** [metrics, class-imbalance, roc, pr-auc]

**Short answer.** Accuracy is misleading under class imbalance — predicting always-negative on a 1%-positive dataset is 99% accurate and useless. Precision = TP/(TP+FP), recall = TP/(TP+FN); F1 is their harmonic mean. ROC-AUC measures separability across thresholds (TPR vs. FPR) but is optimistic under heavy imbalance because FPR's denominator (TN) dominates. **PR-AUC** (precision vs. recall) is the right summary metric for imbalanced binary classification.

**Expansion / why this is the answer.**
- Set up: TP, FP, TN, FN; the four corners.
- **Precision**: "of the items I called positive, what fraction were truly positive?" — costly when false positives are expensive (spam, ads served to wrong people).
- **Recall** (sensitivity): "of the truly positive items, what fraction did I catch?" — costly when false negatives are expensive (cancer screening, fraud).
- **F1** = `2·P·R/(P+R)`: balances them; useful when you want a single number and the classes are imbalanced.
- **ROC-AUC**: integral of TPR over FPR as you sweep the threshold. Threshold-independent. **Pitfall**: under severe class imbalance (say 0.1% positive), FPR is dominated by the huge negative pool, so even a poor classifier can get a high ROC-AUC.
- **PR-AUC** (a.k.a. Average Precision): integral of precision over recall. Robust to imbalance — recommended for imbalanced problems (Saito & Rehmsmeier 2015).
- **Calibration**: separate axis — a model can have great ROC-AUC and terrible calibration. Use Brier score / ECE.

**Common follow-ups.**
- "What's the F-beta score?" → `(1 + β²) · P·R / (β²·P + R)`; β > 1 weights recall, β < 1 weights precision.
- "When would you choose PR-AUC over F1?" → PR-AUC is threshold-independent; F1 is at a specific threshold. PR-AUC for model selection, F1 at deployment threshold.
- "Why is accuracy fine for balanced multiclass tasks like MNIST?" → No imbalance.

**Common mistakes.**
- Reporting ROC-AUC alone on a 0.5%-positive dataset.
- Conflating F1 (a point on the PR curve) with PR-AUC (the full curve).
- Forgetting calibration entirely.

**References.**
- [Saito & Rehmsmeier — "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) — the canonical PR-AUC argument.
- [Davis & Goadrich — "The Relationship Between Precision-Recall and ROC Curves"](https://dl.acm.org/doi/10.1145/1143844.1143874) — formal connection.

---

### Q: What is data leakage and how do you prevent it?

**Category:** concept
**Difficulty:** mid
**Tags:** [leakage, evaluation, pipelines]

**Short answer.** Data leakage is when information that wouldn't be available at prediction time leaks into training — through features (e.g., post-outcome features), through target encoding done before the split, through duplicates spanning train/val, or through time-ordered data split randomly. The cure is to enforce strict pipeline hygiene: split first, fit transforms on train only, time-respect when temporal, dedup across splits.

**Expansion / why this is the answer.**
- Common sources:
  - **Target leakage**: a feature that is computed using the target (e.g., "average order value of customers who churned this month" as a feature for churn prediction).
  - **Train/test contamination**: scaling, imputing means, target encoding, or selecting top-k features *before* the split → val data leaks into training.
  - **Temporal leakage**: random shuffling of a time-series; the val set's future is part of the train.
  - **Group leakage**: same patient in both train and val (medical), same user (recsys), same image in different lighting (CV).
  - **Embedding / pretraining contamination**: pretraining corpus contains the eval set — the modern LLM-era leakage problem (see Topic 7).
- The cure:
  - Split first; fit everything on train only; apply to val.
  - Use group-aware splits (`GroupKFold` in scikit-learn) when group leakage is possible.
  - Use **time-based splits** for temporal data; never random.
  - Decontaminate eval sets against pretraining corpora (n-gram check at minimum).

**Common follow-ups.**
- "Show me a real-world example of leakage." → Kaggle competitions have a long history of these (e.g. "patient ID was correlated with label by accident"); FAANG interviewers love to test for this.
- "How would you detect leakage in a model that performs suspiciously well on val?" → Permutation importance + temporal-split sanity check + checking whether removing one feature collapses performance.

**Common mistakes.**
- Scaling/imputing before the split.
- Saying "I used cross-validation, so leakage is impossible" — false; you still have to fit transforms inside the fold.
- Forgetting group leakage entirely.

**References.**
- [Kaufman, Rosset, Perlich — "Leakage in Data Mining: Formulation, Detection, and Avoidance"](https://dl.acm.org/doi/10.1145/2382577.2382579) — the canonical taxonomy.
- [scikit-learn — Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) — primary docs on GroupKFold, TimeSeriesSplit.

---

### Q: When would you handle class imbalance with class weights vs. oversampling vs. undersampling vs. focal loss?

**Category:** concept
**Difficulty:** mid
**Tags:** [class-imbalance, focal-loss, oversampling]

**Short answer.** Light imbalance (90/10): often nothing — just pick the right metric (PR-AUC). Moderate imbalance: class weights in the loss or SMOTE-style oversampling. Severe (>1000:1): undersampling the majority class + threshold tuning, often combined with focal loss for the remaining classifier. Always evaluate on the original distribution, not the rebalanced one.

**Expansion / why this is the answer.**
- **Class weights** in the loss: `w_pos · L_pos + w_neg · L_neg`, with weights inversely proportional to class frequency. Cheap, works well at moderate imbalance.
- **Oversampling minority**: random duplication (simple, can overfit), SMOTE (Chawla et al. 2002 — synthesize via interpolation; works on tabular, dubious on text/images).
- **Undersampling majority**: throws away data but reduces compute; pairs well with ensembling (train many classifiers each on a different undersampled batch).
- **Focal loss** (Lin et al. 2017): `−(1 − p_t)^γ · log p_t`. Down-weights easy examples; was developed for dense object detection where the negative class is dominant.
- **Threshold tuning**: train however; at deployment, pick the threshold that hits your operational PR point.
- **Anomaly-detection framing**: at >10⁴:1 imbalance, consider one-class or anomaly-detection methods rather than supervised classification.

**Common follow-ups.**
- "SMOTE on text?" → Usually not; text isn't a Euclidean space where linear interpolation makes sense.
- "Why focal loss over class weights?" → Class weights weight by *class*; focal loss weights by *example difficulty*. The combination is common.
- "When can you ignore imbalance entirely?" → When the loss + the metric handle it (cross-entropy + PR-AUC) and the imbalance isn't extreme.

**Common mistakes.**
- Evaluating on the rebalanced data and reporting that as "the metric."
- Oversampling before the train/val split (textbook leakage).
- Treating focal loss as a free lunch; it has its own hyperparameter `γ`.

**References.**
- [Chawla et al. — "SMOTE: Synthetic Minority Over-sampling Technique"](https://www.jair.org/index.php/jair/article/view/10302) — SMOTE.
- [Lin et al. — "Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002) — focal loss.
- [scikit-learn — class imbalance handling](https://scikit-learn.org/stable/modules/svm.html#unbalanced-problems) — practical recipes.

---

### Q: Walk me through k-fold cross-validation. When is it the wrong choice?

**Category:** concept
**Difficulty:** intro
**Tags:** [cv, evaluation, time-series]

**Short answer.** Split the data into k folds; for each fold, train on the other k−1 and evaluate on the held-out fold; average the k evaluation scores. It's the wrong choice when the data has temporal structure (use TimeSeriesSplit instead), when groups span observations (use GroupKFold), or when you have so much data that a single train/val/test split has tight enough confidence intervals.

**Expansion / why this is the answer.**
- Standard k-fold: k=5 or k=10 typical; stratify for classification (StratifiedKFold) to preserve class balance per fold.
- **Why not always use it?** Cost (k× training time), and it's not actually variance-reducing in the modal "I have a million examples" deep-learning setup — a single held-out 5–10% is statistically tight enough.
- **Temporal data**: each fold's training set must precede its val set. `TimeSeriesSplit` produces expanding-window folds.
- **Grouped data**: same patient/user in both train and val biases the metric. `GroupKFold` keeps groups in one fold.
- **Repeated k-fold / nested k-fold**: nested for unbiased hyperparameter selection + evaluation (Cawley & Talbot 2010); rarely used in modern DL because compute dominates.
- LOO (leave-one-out) is a degenerate `k = N`; high variance estimator; rarely useful in practice.

**Common follow-ups.**
- "What's nested CV and why might you use it?" → Outer loop for model evaluation, inner loop for hyperparameter tuning. Avoids the "you tuned on the test set" trap.
- "Why is CV cost prohibitive in deep learning?" → Each fold needs a full retrain; the practical alternative is a fixed train/val/test split with bootstrap CIs on the test metric.

**Common mistakes.**
- Random k-fold on time-series.
- Tuning hyperparameters on a CV score, then reporting that same score as your "test" performance.

**References.**
- [Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, §7.10](https://hastie.su.domains/ElemStatLearn/) — CV theory.
- [Cawley & Talbot — "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation"](https://www.jmlr.org/papers/v11/cawley10a.html) — nested CV.
- [scikit-learn — cross-validation iterators](https://scikit-learn.org/stable/modules/cross_validation.html) — GroupKFold, TimeSeriesSplit.

---

### Q: What is the curse of dimensionality, and how do you mitigate it?

**Category:** concept
**Difficulty:** intro
**Tags:** [high-dim, distance-metrics, embeddings]

**Short answer.** In high dimensions, data becomes sparse and distance metrics lose discriminative power — every pair of points ends up at roughly the same Euclidean distance, so methods that rely on distance (kNN, k-means, density estimation) degrade. Mitigations: dimensionality reduction (PCA, t-SNE, UMAP for visualization; learned embeddings for downstream tasks), feature selection (L1, mutual information), or using methods less sensitive to dimension (tree ensembles).

**Expansion / why this is the answer.**
- The math: for points uniform on `[0,1]^d`, the ratio `(max - min) / mean` of pairwise distances → 0 as `d → ∞` (Beyer et al. 1999). Concretely, all points are "equally far."
- Density estimation needs exponentially more samples in `d` to maintain the same coverage of the input space.
- **What dimensionality reduction actually does**: project to a lower-dimensional subspace where distance is meaningful again, or learn representations (embeddings) that put semantically similar points near each other.
- Modern LLMs implicitly handle high dimension by **learning low-dimensional manifolds** — the embedding space is high-dimensional (768, 1536, ...), but the data lives on a small manifold.
- The "blessing of dimensionality" qualification (Donoho 2000): in some structured high-dim regimes (sparse signals, low intrinsic dimension), high dim is fine.

**Common follow-ups.**
- "Why does kNN fail in high dimensions and not GBMs?" → Tree splits use one dimension at a time; the curse is about distance metrics that aggregate across all dimensions.
- "Difference between PCA and t-SNE?" → PCA: linear, preserves global variance, deterministic. t-SNE: non-linear, preserves local neighborhoods, stochastic, only for visualization.

**Common mistakes.**
- Saying "embeddings make the curse go away" — they help by learning where the manifold is, but the embedding space itself is high-dim.
- Confusing PCA (variance maximization, linear) with autoencoders (general non-linear).

**References.**
- [Beyer et al. — "When Is 'Nearest Neighbor' Meaningful?"](https://link.springer.com/chapter/10.1007/3-540-49257-7_15) — the canonical proof.
- [Donoho — "High-Dimensional Data Analysis: The Curses and Blessings of Dimensionality"](https://www.semanticscholar.org/paper/High-Dimensional-Data-Analysis%3A-The-Curses-and-of-Donoho/eb0bf3ad9ec4be0cd34541b7e0fe6fb6e7c34e1c) — the canonical "qualification" essay.

---

### Q: When does PCA stop being useful, and what should you use instead?

**Category:** concept
**Difficulty:** mid
**Tags:** [pca, dimensionality-reduction, embeddings]

**Short answer.** PCA captures linear directions of maximum variance — it stops being useful when the structure you want is non-linear (e.g., a curved manifold), when the dimensions you care about aren't the high-variance ones (e.g., class boundaries lie along low-variance directions), or when the data is categorical/non-numeric. Alternatives: kernel PCA / autoencoders for non-linear; LDA when supervised; UMAP / t-SNE for visualization; learned embeddings (Word2Vec / SBERT / OpenAI / Cohere) for text.

**Expansion / why this is the answer.**
- PCA finds orthogonal axes maximizing variance via SVD on centered data. Optimal **linear** reconstruction.
- **Where it fails**:
  - The "Swiss roll" / non-linear manifold case: the high-variance direction is one that wraps around the manifold; PCA flattens it incorrectly.
  - Supervised settings: the class boundary may be perpendicular to the highest-variance axis (LDA is the supervised cousin).
  - Categorical / sparse text data: PCA on bag-of-words is a weak version of LSA; modern alternative is a learned dense embedding.
- **Alternatives**:
  - **Kernel PCA / Isomap / LLE**: non-linear manifold learning, niche outside of demos today.
  - **Autoencoders**: learned non-linear reduction; popular as a representation learner.
  - **t-SNE / UMAP**: for **visualization only** — they preserve local neighborhoods but distort global structure; do not feed t-SNE outputs into a downstream classifier.
  - **Pretrained embeddings** (Word2Vec, SBERT, OpenAI, Cohere): the practical "dim reduction" for text in 2026.

**Common follow-ups.**
- "What does t-SNE optimize?" → KL divergence between joint distributions defined by similarities in input and embedding space (van der Maaten & Hinton 2008).
- "Why is UMAP often preferred to t-SNE?" → Faster, theoretically better at global structure, deterministic-ish given a seed.

**Common mistakes.**
- Using t-SNE coordinates as features for a downstream model.
- Forgetting that PCA requires centering (and ideally scaling).
- Treating PCA as causal — it's coordinate change, not feature selection.

**References.**
- [Bishop, *Pattern Recognition and Machine Learning*, §12](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — PCA derivation.
- [van der Maaten & Hinton — "Visualizing Data using t-SNE"](https://www.jmlr.org/papers/v9/vandermaaten08a.html) — t-SNE.
- [McInnes, Healy, Melville — "UMAP: Uniform Manifold Approximation and Projection"](https://arxiv.org/abs/1802.03426) — UMAP.

---

### Q: Difference between generative and discriminative models — with examples.

**Category:** concept
**Difficulty:** intro
**Tags:** [generative, discriminative, bayes]

**Short answer.** A discriminative model learns `p(y|x)` directly — given input, predict label. A generative model learns the full joint `p(x, y)` (or `p(x)` alone) — from which `p(y|x)` can be derived via Bayes' rule. Discriminative: logistic regression, SVM, most classifiers. Generative: naive Bayes, GMMs, VAEs, diffusion models, autoregressive LMs.

**Expansion / why this is the answer.**
- Practical tradeoff (Ng & Jordan 2001, "On Discriminative vs. Generative Classifiers"): discriminative models tend to have **lower asymptotic error** when data is abundant (they don't waste capacity modeling `p(x)`), while generative models **converge faster from fewer samples** and let you generate / explain.
- **Modern context**: LLMs are autoregressive generative models — they learn `p(x_t | x_<t)`. ChatGPT, Claude, Gemini are all generative under the hood.
- Generative models also enable:
  - **Anomaly detection** (low `p(x)` → likely anomaly).
  - **Conditional generation** (text-to-image, etc.).
  - **Bayesian-style explanations** via the joint.
- Caveat: "generative" in 2026 usually means the deep generative family (autoregressive LMs, diffusion, VAEs, GANs), not the old GMM/naive-Bayes view.

**Common follow-ups.**
- "Is a softmax classifier discriminative or generative?" → Discriminative — it directly maps `x → p(y|x)`.
- "Is GPT discriminative or generative?" → Generative: it learns `p(token | context)` and you sample tokens.
- "Can a generative model do classification?" → Yes via Bayes: pick the class `y` maximizing `p(x, y)`. Often worse than a tuned discriminative classifier at scale, but used in low-shot settings.

**Common mistakes.**
- Saying GANs / diffusion are discriminative because they have a discriminator — the discriminator is just a training-time component; the generator is the generative model.
- Treating "generative" as a property of the output (text-out!) rather than the modeled distribution.

**References.**
- [Ng & Jordan — "On Discriminative vs. Generative Classifiers"](https://papers.nips.cc/paper_files/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html) — the canonical tradeoff paper.
- [Goodfellow, Bengio, Courville — *Deep Learning*, §5.1.3](https://www.deeplearningbook.org/) — generative vs. discriminative formal definitions.

---

### Q: What is the difference between supervised, unsupervised, self-supervised, and reinforcement learning — with one modern example of each?

**Category:** concept
**Difficulty:** intro
**Tags:** [paradigms, self-supervised, rlhf]

**Short answer.** Supervised: human-labeled `(x, y)`. Unsupervised: no labels — discover structure (clustering, density estimation). Self-supervised: labels generated from the data itself (next-token prediction, masked LM, contrastive pairs); the dominant pretraining paradigm. Reinforcement learning: learn a policy from environment-provided rewards. Modern examples: ResNet for image classification (supervised); k-means on customer segmentation (unsupervised); GPT pretraining (self-supervised); RLHF for LLM alignment (RL).

**Expansion / why this is the answer.**
- **Supervised**: classical ML; needs labels. Cost scales with labels.
- **Unsupervised**: no labels; tricky to evaluate.
- **Self-supervised**: the LLM revolution's substrate — predict the next token / fill the blank / contrast positive vs. negative pairs (CLIP, SimCLR). Generates "labels" from the data itself.
- **Reinforcement learning**: agent ↔ environment, reward signal, policy optimization. Classical: Atari, Go. Modern: RLHF on top of LLMs (Christiano et al. 2017; Ouyang et al. 2022 InstructGPT) — though most "RLHF" in 2026 is being replaced or supplemented by DPO/GRPO.
- The boundary blurs: self-supervised + supervised fine-tuning + RLHF is the standard LLM pipeline.

**Common follow-ups.**
- "Is BERT supervised or self-supervised?" → Self-supervised (masked LM); the downstream task (e.g. sentiment classification) is supervised.
- "Is contrastive learning supervised or self-supervised?" → Self-supervised when pairs are generated automatically (augmentations of the same image); supervised when pairs come from a labeled relationship.

**Common mistakes.**
- Calling InstructGPT-style fine-tuning "supervised learning" only — it has an SFT step that is supervised, but the headline alignment step is RL (or DPO).
- Conflating self-supervised with semi-supervised (the latter uses a small labeled set + a large unlabeled one).

**References.**
- [Goodfellow, Bengio, Courville — *Deep Learning*, §5.1](https://www.deeplearningbook.org/) — paradigm definitions.
- [Ouyang et al. — "Training language models to follow instructions with human feedback" (InstructGPT)](https://arxiv.org/abs/2203.02155) — RLHF.
- [Chen et al. — "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)](https://arxiv.org/abs/2002.05709) — self-supervised contrastive learning.

---

### Q: Derive the gradient of softmax + cross-entropy with respect to the pre-softmax logits.

**Category:** derivation
**Difficulty:** mid
**Tags:** [softmax, cross-entropy, gradients, derivation]

**Short answer.** `∂L/∂z_i = p_i − y_i`, where `p = softmax(z)` and `y` is the one-hot target. The clean "predicted minus actual" form is the whole reason softmax + cross-entropy is the standard pairing — the messy `p_i(1 − p_i)` derivative from softmax cancels against the `1/p_i` from cross-entropy's `log`.

**Expansion / why this is the answer.**
- Setup: logits `z ∈ ℝ^K`; softmax `p_i = exp(z_i) / Σ_j exp(z_j)`; loss `L = −Σ_k y_k log p_k` (one-hot `y`).
- Softmax Jacobian: `∂p_i / ∂z_j = p_i(δ_{ij} − p_j)`.
- Chain rule:
  `∂L/∂z_j = Σ_i (∂L/∂p_i)(∂p_i/∂z_j) = Σ_i (−y_i/p_i) · p_i(δ_{ij} − p_j) = Σ_i −y_i(δ_{ij} − p_j) = −y_j + p_j · Σ_i y_i = p_j − y_j` (since `Σ y_i = 1`).
- The same identity is what makes `softmax + NLL` numerically convenient — gradient computation collapses to a subtraction.
- This is the canonical interview derivation; you should be able to do it cold on a whiteboard.

**Common follow-ups.**
- "What about non-one-hot targets (label smoothing)?" → Same derivation; `y` becomes the smoothed distribution; result is still `p − y`.
- "Why doesn't this simplification happen for MSE + softmax?" → MSE's `∂L/∂p` is `2(p − y)`, not `−y/p`, so the softmax-Jacobian factor `p_i(1 − p_i)` doesn't cancel — and that's why MSE-on-softmax has the dead-gradient problem.

**Common mistakes.**
- Forgetting that `Σ y_i = 1` lets you collapse the final step.
- Getting the sign of `δ_{ij}` wrong (off-by-sign is common under pressure).

**References.**
- [Bishop, *Pattern Recognition and Machine Learning*, §4.3.4](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — softmax derivation.
- [Goodfellow, Bengio, Courville — *Deep Learning*, §6.2.2.3](https://www.deeplearningbook.org/) — cross-entropy gradient.

---

### Q: When does bagging beat boosting, and when does the reverse hold?

**Category:** concept
**Difficulty:** mid
**Tags:** [bagging, boosting, ensembles, random-forest, gbm]

**Short answer.** Bagging (Random Forest) trains independent learners in parallel on bootstrap samples and averages — best when base learners are high-variance and you want stability + parallelism. Boosting (GBM/XGBoost/LightGBM/CatBoost) trains learners sequentially, each correcting the previous's residuals — typically wins on accuracy when well-tuned, at the cost of being sensitive to noisy labels and harder to parallelize.

**Expansion / why this is the answer.**
- **Bagging** (Breiman 1996; Random Forest 2001):
  - Bootstrap-sample the data; train a high-variance model (deep tree) on each.
  - Aggregate by averaging (regression) or voting (classification).
  - Variance reduction via averaging; bias unchanged from the base learner.
  - Embarrassingly parallel.
  - Robust to noisy labels (outliers averaged out).
- **Boosting** (Friedman 2001 Gradient Boosting; XGBoost, LightGBM, CatBoost):
  - Train sequentially; each weak learner fits the previous's residuals (or gradient).
  - Reduces both bias and variance; typically lower bias than bagging.
  - Sensitive to label noise (model amplifies it through residuals).
  - Sequential — harder to parallelize across boosting iterations (but parallel within a tree).
- **When bagging wins**:
  - Very noisy labels.
  - You want predictable runtime / parallelism.
  - Quick baseline with minimal tuning.
- **When boosting wins**:
  - Most Kaggle / production tabular problems — boosted-tree variants top leaderboards.
  - Cleaner labels; willing to tune.
- **Modern picks**: LightGBM for speed at scale; CatBoost for categorical-heavy tabular; XGBoost for general workhorse.
- **Deep tabular** (TabNet, FT-Transformer): closed some gaps but typically still lose to boosted trees per Grinsztajn 2022.

**Common follow-ups.**
- "Why is Random Forest 'embarrassingly parallel'?" → Each tree trains on its own bootstrap independently; no inter-tree dependency.
- "What's gradient boosting's gradient, exactly?" → The negative gradient of the loss with respect to the current prediction, computed sample-by-sample.

**Common mistakes.**
- Calling Random Forest "boosting" or vice versa.
- Forgetting that boosting overfits noisy labels.

**References.**
- [Breiman — "Random Forests"](https://link.springer.com/article/10.1023/A:1010933404324) — RF.
- [Friedman — "Greedy Function Approximation: A Gradient Boosting Machine"](https://www.jstor.org/stable/2699986) — gradient boosting.
- [Chen & Guestrin — "XGBoost"](https://arxiv.org/abs/1603.02754) — XGBoost.

---

### Q: Walk through the EM algorithm using GMM as the example.

**Category:** derivation
**Difficulty:** senior
**Tags:** [em, gmm, latent-variables, mle]

**Short answer.** EM (Expectation-Maximization) iteratively maximizes a likelihood with latent variables: **E-step** computes the posterior over latents given current parameters; **M-step** updates parameters to maximize the expected complete-data log-likelihood under that posterior. For GMM, E-step computes "responsibilities" `γ_ik` (probability that point `i` came from cluster `k`); M-step updates mixing weights, means, and covariances as responsibility-weighted statistics.

**Expansion / why this is the answer.**
- **The problem**: data `x_1..x_N`; model `p(x) = Σ_k π_k N(x | μ_k, Σ_k)`. Marginal log-likelihood is non-convex; direct MLE is hard because of the sum-inside-log.
- **EM trick**: introduce latent assignment `z_i ∈ {1..K}` (which Gaussian generated `x_i`); compute the lower bound (ELBO) on `log p(x)`; alternate.
- **E-step** (responsibilities): `γ_ik = π_k N(x_i | μ_k, Σ_k) / Σ_j π_j N(x_i | μ_j, Σ_j)`.
- **M-step** (closed-form for GMM):
  - `N_k = Σ_i γ_ik`
  - `π_k = N_k / N`
  - `μ_k = (1/N_k) Σ_i γ_ik x_i`
  - `Σ_k = (1/N_k) Σ_i γ_ik (x_i − μ_k)(x_i − μ_k)ᵀ`
- **Properties**:
  - Each step never decreases the data log-likelihood (Dempster, Laird, Rubin 1977).
  - Converges to a local maximum (not global).
  - Sensitive to initialization (k-means warmup is standard).
- **Connection to k-means**: k-means is a hard-assignment limit of GMM-EM (let covariances → identity-scaled, responsibilities → one-hot).
- **Modern uses**: still important for mixture models, HMMs, topic models (LDA via variational EM). The general framework underlies the variational autoencoder.

**Common follow-ups.**
- "Why does EM monotonically increase log-likelihood?" → Jensen's-inequality bound becomes tight after the E-step; M-step increases the bound.
- "When does EM fail?" → Singular covariance (one Gaussian collapses to one point with zero covariance — degenerate likelihood); poor init.

**Common mistakes.**
- Forgetting EM converges to a local optimum (not global).
- Mixing the assignment step with the parameter update (think of them as separate).

**References.**
- [Bishop, *Pattern Recognition and Machine Learning*, §9](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — EM and GMM canonical reference.
- [Dempster, Laird, Rubin — "Maximum Likelihood from Incomplete Data via the EM Algorithm"](https://www.jstor.org/stable/2984875) — the original 1977 paper.

---

### Q: Walk through backpropagation step by step for a 2-layer MLP.

**Category:** derivation
**Difficulty:** mid
**Tags:** [backprop, chain-rule, mlp]

**Short answer.** Forward: compute activations through layers. Backward: starting from the loss, recursively apply the chain rule to propagate gradients back to each parameter. For a 2-layer MLP `y = W₂ · σ(W₁ x + b₁) + b₂`, gradients are: `∂L/∂W₂` from `∂L/∂y × hᵀ`; `∂L/∂h` from `W₂ᵀ ∂L/∂y`; through σ' to get `∂L/∂(W₁x + b₁)`; then to `∂L/∂W₁` and `∂L/∂b₁`.

**Expansion / why this is the answer.**
- Setup:
  - `z₁ = W₁ x + b₁` (pre-activation)
  - `h = σ(z₁)` (hidden)
  - `z₂ = W₂ h + b₂` (output logits)
  - `L = Loss(z₂, y_target)`
- Forward pass: compute `z₁, h, z₂` and `L` in order.
- Backward pass (denote `δᵢ = ∂L/∂zᵢ`):
  - `δ₂ = ∂L/∂z₂` (e.g. `softmax(z₂) − y` for softmax+CE).
  - `∂L/∂W₂ = δ₂ hᵀ`
  - `∂L/∂b₂ = δ₂`
  - `∂L/∂h = W₂ᵀ δ₂`
  - `δ₁ = (W₂ᵀ δ₂) ⊙ σ'(z₁)` (Hadamard product with activation derivative).
  - `∂L/∂W₁ = δ₁ xᵀ`
  - `∂L/∂b₁ = δ₁`
- **Why it works**: each layer's `δ` is the layer's "error signal"; gradients with respect to parameters follow from the chain rule and the outer product with the layer's input.
- **Autograd**: deep-learning frameworks (PyTorch, JAX) build a computation graph at forward time and walk it backward; you never write these formulas by hand in practice — but you should understand them for interviews.

**Common follow-ups.**
- "What's the cost in memory of backprop?" → Must store all forward activations (or recompute via gradient checkpointing).
- "Why σ' multiplied in `δ₁`?" → Chain through the activation: `∂h/∂z₁ = σ'(z₁)`, applied elementwise (Hadamard).
- "How does this generalize to a deep net?" → Same recursion; `δ_l = (W_{l+1}ᵀ δ_{l+1}) ⊙ σ'_l`.

**Common mistakes.**
- Forgetting the Hadamard with `σ'` for the activation derivative.
- Transposing `W` in the wrong direction.
- Treating gradients as scalars when they're tensors with specific shapes.

**References.**
- [Goodfellow, Bengio, Courville — *Deep Learning*, §6.5](https://www.deeplearningbook.org/) — backpropagation canonical.
- [Rumelhart, Hinton, Williams — "Learning representations by back-propagating errors"](https://www.nature.com/articles/323533a0) — the 1986 paper.

---

### Q: What is dropout? Why does it work, and where do you not use it?

**Category:** concept
**Difficulty:** intro
**Tags:** [dropout, regularization, deep-learning]

**Short answer.** Dropout (Srivastava et al. 2014) randomly zeros out a fraction `p` of activations during training, forcing each unit to be useful without relying on any particular neighbor — an implicit ensemble. At inference, no dropout is applied, but activations are scaled by `(1-p)` to preserve expected magnitudes (or "inverted dropout" applies the `/(1-p)` at train time so inference is unchanged). Strong on overparameterized models with small data. Modern LLMs largely don't use dropout in the FFN/attention layers because they're trained on enough data that regularization is unnecessary and dropout interacts badly with mixed precision.

**Expansion / why this is the answer.**
- The mechanism: per training step, sample a binary mask; zero out activations; rescale the rest.
- **Ensemble interpretation**: dropout is approximate model averaging over an exponential number of sub-networks.
- **Hyperparameter**: dropout probability `p ∈ [0, 0.5]`. Higher = more regularization.
- **Variants**:
  - Standard dropout: per-element.
  - DropConnect (Wan et al. 2013): drop weights, not activations.
  - Spatial dropout (CNNs): drop entire feature maps.
- **Where it's used**:
  - Smaller models (BERT-base, classical CNNs).
  - LoRA layers (typical: dropout 0.05–0.1 on the adapter input).
  - During SFT of LLMs (small dropout, e.g. 0.1).
- **Where it's not used (in 2026 frontier LLM pretraining)**:
  - Decoder-only pretraining; the data is so abundant that overfitting isn't the binding constraint.
  - With Mixed Precision, dropout requires careful masking math; the benefit doesn't justify the complexity.
- **DropPath / Stochastic Depth** (Huang et al. 2016): drop entire layers at training time; a regularizer used in some vision transformers.

**Common follow-ups.**
- "Why scale by `(1-p)` at inference?" → To preserve expected activation magnitudes. Inverted dropout scales at train time instead.
- "What's MC Dropout?" → Keep dropout on at inference and average many forward passes; gives an approximate Bayesian uncertainty estimate (Gal & Ghahramani 2016).

**Common mistakes.**
- Forgetting the scaling factor entirely → magnitudes drift at inference.
- Using dropout on layers that already have strong regularization (e.g. heavy weight decay + LayerNorm).

**References.**
- [Srivastava et al. — "Dropout"](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) — the canonical paper.
- [Gal & Ghahramani — "Dropout as a Bayesian Approximation"](https://arxiv.org/abs/1506.02142) — MC dropout.

---

### Q: Difference between batch GD, stochastic GD, and mini-batch GD. Which do modern systems use?

**Category:** concept
**Difficulty:** intro
**Tags:** [sgd, batch-size, optimization]

**Short answer.** **Batch GD**: gradient over the entire dataset per step — slow, accurate, smooth descent. **Stochastic GD (SGD)**: gradient from one sample — fast updates, noisy descent. **Mini-batch GD**: gradient from a batch of `B` samples — the universal modern compromise. LLMs use mini-batch with very large effective batch sizes (millions of tokens), often via gradient accumulation across micro-batches.

**Expansion / why this is the answer.**
- **Batch GD**: theoretically smooth; impractical at scale (one update per epoch).
- **SGD** (one sample): fast updates per step; noisy gradients can help escape saddle points.
- **Mini-batch** (typical 32–8192 samples): trade-off; the standard everywhere.
- **LLM training scales**:
  - Effective batch = micro-batch × gradient-accumulation-steps × DP-degree.
  - GPT-3 trained at batch size ~3.2M tokens.
  - LLaMA 3 reportedly trained at 16M tokens/batch.
  - Limits: too-large batch hits diminishing returns (Smith et al. 2017, "Don't Decay the Learning Rate, Increase the Batch Size").
- **Practical considerations**:
  - **Gradient accumulation**: simulate larger batch when GPU memory limits the micro-batch.
  - **LR scaling**: larger batch typically needs higher LR (linear scaling rule, Goyal et al. 2017, but it breaks down beyond a regime).
  - **Critical batch size** (McCandlish et al. 2018): the size beyond which gains saturate; depends on the gradient noise.

**Common follow-ups.**
- "Why do we need bigger batches?" → Throughput (utilization), stability, sample efficiency tradeoffs.
- "What's the LR-batch relationship?" → Linear scaling rule (LR ∝ batch) within a window; breaks down at very large batch.

**Common mistakes.**
- Conflating "batch" (full dataset) with "mini-batch" (subset).
- Forgetting that DDP / FSDP multiply effective batch.

**References.**
- [Smith et al. — "Don't Decay the Learning Rate, Increase the Batch Size"](https://arxiv.org/abs/1711.00489).
- [McCandlish et al. — "An Empirical Model of Large-Batch Training"](https://arxiv.org/abs/1812.06162) — critical batch size.
- [Goyal et al. — "Accurate, Large Minibatch SGD"](https://arxiv.org/abs/1706.02677) — linear LR scaling rule.

---

### Q: When is early stopping the right regularizer, and when is it not?

**Category:** concept
**Difficulty:** intro
**Tags:** [early-stopping, regularization, generalization]

**Short answer.** Early stopping halts training when validation loss stops improving — the most common and cheapest regularizer in classical ML. Good when training-vs-validation curves diverge clearly (overfitting). Bad when (a) you're under the double-descent threshold (modern overparameterized regime where more training helps), (b) you have a very small validation set with noisy signals, or (c) the val loss has multiple local minima during training.

**Expansion / why this is the answer.**
- The simple rule: monitor val loss; stop after N epochs without improvement; revert to the best-val checkpoint.
- **Why it works**: classically, after some point, gradient updates fit training noise rather than signal. Early stop = effective capacity reduction.
- **Modern overparameterized DL**: the picture is different. Models often keep improving past the apparent overfitting point (Nakkiran et al. 2020 double descent). LLMs are trained well past the "naive overfitting" boundary because they're overparameterized for the dataset.
- **When still used**:
  - Classical ML (GBMs, SVMs, smaller DL).
  - Fine-tuning regime where overfitting is fast.
  - Resource constraints (can't afford full training).
- **Variants**:
  - **Patience-based**: stop after N epochs without val improvement.
  - **Loss-threshold-based**: stop when val loss < threshold.
- **Failure modes**:
  - Tiny val set → noisy signal → false-positive stops.
  - Cyclic learning rates: val loss oscillates; check after each cycle.

**Common follow-ups.**
- "How do you do early stopping in distributed training?" → All ranks track val loss; stop signaled across all ranks.
- "Doesn't this require a held-out val set?" → Yes; the cost is one extra forward pass periodically.

**Common mistakes.**
- Early stopping on training loss (defeats the purpose).
- Tiny patience → false stops.

**References.**
- [Goodfellow, Bengio, Courville — *Deep Learning*, §7.8](https://www.deeplearningbook.org/) — early stopping.
- [Nakkiran et al. — "Deep Double Descent"](https://arxiv.org/abs/1912.02292) — overparameterized regime breaks classical intuition.

---

### Q: Compare Random Forest and Gradient Boosting on a tabular problem.

**Category:** concept
**Difficulty:** mid
**Tags:** [random-forest, gbm, comparison, tabular]

**Short answer.** Both are tree ensembles. **RF**: independent trees trained on bootstrap samples; aggregate by voting; reduces variance. **GBM**: sequential trees, each fitting residuals; reduces both bias and variance. RF is faster to train (parallel) and more robust to noisy labels. GBM (XGBoost / LightGBM / CatBoost) is typically more accurate on most tabular benchmarks when tuned. Default first pick: GBM. RF as a baseline or under label-noise constraints.

**Expansion / why this is the answer.**
- **Training time**:
  - RF: parallel; embarrassingly so.
  - GBM: sequential across boosting rounds; parallel within each tree.
- **Hyperparameter sensitivity**:
  - RF: relatively robust; `n_estimators`, `max_depth` cover most of the variance.
  - GBM: many knobs (learning rate, depth, regularization, subsampling); needs tuning.
- **Performance**:
  - On most tabular benchmarks: GBM > RF when both are tuned.
  - On noisy data: RF often competitive; GBM can overfit noise.
- **Modern picks**:
  - LightGBM: fastest GBM, leaf-wise growth.
  - XGBoost: most-tested; broad feature set.
  - CatBoost: best for categorical features (ordered boosting).
  - RF (scikit-learn): the no-tuning baseline.
- **Both handle**:
  - Missing values (with different mechanisms).
  - Mixed numeric / categorical (with encoding).
  - No need to scale features.
- **Neither needs**: scaling, PCA, neural-net-style preprocessing.

**Common follow-ups.**
- "Why is GBM more accurate?" → Sequential residual-fitting reduces bias; the model class is more expressive.
- "When would you stick with RF?" → Quick baseline, noisy labels, parallelism-critical.
- "Why is the leaf-wise growth in LightGBM faster than XGBoost's level-wise?" → Less wasted computation; can be more accurate but more prone to overfit on small data.

**Common mistakes.**
- Saying RF is a kind of boosting (it's bagging).
- Picking RF for accuracy on clean data.

**References.**
- [Breiman — "Random Forests"](https://link.springer.com/article/10.1023/A:1010933404324) — RF.
- [Chen & Guestrin — "XGBoost"](https://arxiv.org/abs/1603.02754) — XGBoost.
- [Ke et al. — "LightGBM"](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) — LightGBM.
- [Prokhorenkova et al. — "CatBoost"](https://arxiv.org/abs/1706.09516) — CatBoost.

---

### Q: Explain SVMs and the kernel trick.

**Category:** concept
**Difficulty:** mid
**Tags:** [svm, kernel-trick, max-margin]

**Short answer.** SVM finds the hyperplane that maximizes the margin between two classes — robust to outliers far from the boundary. The kernel trick replaces dot products `xᵢ · xⱼ` with a kernel `K(xᵢ, xⱼ)` that implicitly maps inputs to a higher-dimensional space, letting SVM learn non-linear boundaries without ever computing the high-dimensional feature vectors. Common kernels: RBF, polynomial, linear. Mostly displaced by deep learning at scale but still strong on small-to-medium tabular with non-linear structure.

**Expansion / why this is the answer.**
- **Hard-margin SVM** (separable case): minimize `||w||²/2` subject to `y_i(w · x_i + b) ≥ 1`. Dual involves `Σ α_i y_i y_j x_i · x_j`.
- **Soft-margin** (non-separable): add slack `ξ_i` with hinge loss; trade margin width for misclassification.
- **Kernel trick**: replace `x_i · x_j` in the dual with `K(x_i, x_j)`. The implicit feature map `φ(x)` satisfies `K(x_i, x_j) = φ(x_i) · φ(x_j)` but you never compute `φ` explicitly.
- **Common kernels**:
  - **Linear**: `K(x, y) = x · y`. Equivalent to no kernel.
  - **Polynomial**: `(x · y + c)^d`.
  - **RBF / Gaussian**: `exp(-γ ||x − y||²)`. Implicit feature space is infinite-dimensional.
  - **Sigmoid**: `tanh(α x · y + c)`.
- **Support vectors**: the training points with non-zero `α_i` — the ones near the boundary. Prediction depends only on these.
- **Complexity**: training is `O(N²)` to `O(N³)`; prohibitive past ~100k samples. Use linear-SVM solvers (LIBLINEAR) for large-scale linear case.
- **Modern context**: SVMs are rarely the right tool for big tabular (boosted trees win) or text/image (deep nets win). They linger in small-data, classical-ML, or kernel-method research.

**Common follow-ups.**
- "What does the RBF kernel's `γ` control?" → Width of the Gaussian; smaller `γ` = wider influence, smoother boundary.
- "Why is the kernel trick called a 'trick'?" → You compute K without computing the feature map; sometimes the feature space is infinite-dimensional (RBF).

**Common mistakes.**
- Treating SVM as a probabilistic classifier; it's geometrically motivated (margin), Platt scaling adds probabilities post-hoc.
- Forgetting SVMs don't scale to >100k samples in their classical form.

**References.**
- [Cortes & Vapnik — "Support-Vector Networks"](https://link.springer.com/article/10.1007/BF00994018) — original SVM paper.
- [Bishop, *Pattern Recognition and Machine Learning*, §7](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — kernel methods chapter.
- [Schölkopf & Smola — *Learning with Kernels*](https://mitpress.mit.edu/9780262536578/learning-with-kernels/) — kernel-methods textbook.

---

### Q: Logistic regression vs. linear regression — when to use each, and why are they "linear"?

**Category:** concept
**Difficulty:** intro
**Tags:** [logistic-regression, linear-regression, glm]

**Short answer.** **Linear regression**: predicts a continuous target; minimizes squared error; assumes Gaussian noise. **Logistic regression**: predicts probability of a binary class; applies sigmoid to a linear combination; minimizes cross-entropy. Both are "linear models" because the prediction is a linear function of the inputs (before any output transformation). Logistic regression is the workhorse binary classifier; linear regression for continuous targets when relationships are roughly linear.

**Expansion / why this is the answer.**
- **Linear regression**: `y = w · x + b`. MLE under Gaussian noise = least squares. Closed-form solution via the normal equations; or solve with gradient descent at scale.
- **Logistic regression**: `p = σ(w · x + b)`. MLE under Bernoulli noise = cross-entropy. No closed form; solve with gradient descent / Newton's method (IRLS).
- Both belong to the **GLM family** (Generalized Linear Models), differing only by the link function (identity vs. logit) and the response distribution (Gaussian vs. Bernoulli).
- **Why "linear"?** The decision boundary or regression surface is *linear in the inputs* — the sigmoid in logistic regression makes the output non-linear, but the boundary `w · x + b = 0` is still a hyperplane.
- **Extending to non-linear**:
  - Feature engineering (polynomial features, interactions).
  - Kernelized variants (kernel logistic regression).
  - Or, more practically, switch to GBM / DL.
- **When linear regression is wrong**:
  - Heteroscedastic noise (variance varies with `x`).
  - Heavy-tailed errors (use Huber loss or quantile regression).
  - Target is bounded or non-Gaussian (use a GLM with the right link).

**Common follow-ups.**
- "What's the gradient of logistic regression?" → `∂L/∂w = Xᵀ(p − y) / N` (clean, same shape as the softmax gradient).
- "Why softmax for multi-class instead of one-vs-rest?" → Softmax gives a calibrated joint distribution; OvR can have inconsistencies (probabilities not summing to 1).

**Common mistakes.**
- Using linear regression for binary classification (gives unbounded predictions).
- Treating logistic regression's output as "more confident" — it's a probability, not a margin.

**References.**
- [Bishop, *Pattern Recognition and Machine Learning*, §4.3](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/).
- [Hastie, Tibshirani, Friedman — *Elements of Statistical Learning*, §4.4](https://hastie.su.domains/ElemStatLearn/) — logistic regression.

---

### Q: Interpret ROC-AUC concretely — what does AUC = 0.8 actually mean?

**Category:** concept
**Difficulty:** intro
**Tags:** [auc, roc, metrics]

**Short answer.** ROC-AUC is the probability that a random positive scores higher than a random negative under the model. AUC = 0.8 means: pick a random positive and a random negative; 80% of the time, the model scores the positive higher. AUC = 0.5 is random; AUC = 1.0 is perfect; AUC < 0.5 means the model is anti-predictive (flip the sign).

**Expansion / why this is the answer.**
- ROC: plot TPR vs. FPR as the threshold sweeps from 0 to 1.
- AUC = area under that curve, equivalent to the rank statistic above (Mann-Whitney U statistic / 2).
- Threshold-independent — useful for comparing models without picking an operating point.
- Insensitive to class imbalance in a misleading way: under heavy imbalance, FPR's huge denominator hides false positives. Use PR-AUC instead.

**Common follow-ups.**
- "What's the relationship between ROC-AUC and the rank statistic?" → They're identical up to normalization.
- "Why does it not depend on threshold?" → It integrates over all thresholds.

**Common mistakes.**
- Reporting AUC on a 0.1%-positive dataset and treating it as the truth.
- Conflating ROC-AUC with PR-AUC.

**References.**
- [Fawcett — "An introduction to ROC analysis"](https://people.inf.elte.hu/kiss/13dwhdm/roc.pdf).
- [Saito & Rehmsmeier — PR vs ROC on imbalanced data](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432).

---

### Q: What is the difference between MLE and MAP?

**Category:** derivation
**Difficulty:** mid
**Tags:** [mle, map, bayesian, frequentist]

**Short answer.** **MLE** (Maximum Likelihood): pick parameters maximizing `p(data | θ)`. Frequentist. **MAP** (Maximum A Posteriori): pick parameters maximizing `p(θ | data) ∝ p(data | θ) · p(θ)` — same as MLE + a prior. With a Gaussian prior, MAP ≡ MLE + L2 regularization; with a Laplace prior, MAP ≡ MLE + L1.

**Expansion / why this is the answer.**
- MLE: `θ_MLE = argmax_θ p(data | θ)`. Equivalent to minimizing negative log-likelihood (NLL).
- MAP: `θ_MAP = argmax_θ p(data | θ) p(θ)`. NLL + prior penalty.
- The prior on `θ` translates directly to a regularizer:
  - Gaussian `N(0, σ²)` → L2 penalty `λ ||θ||²` with `λ = 1/(2σ²)`.
  - Laplace `Lap(0, b)` → L1 penalty.
- Modern DL: implicit MAP via weight decay (L2 prior). LLMs are MLE on next-token prediction during pretraining.
- Full Bayes integrates over the posterior — much more expensive; rarely done in DL.

**Common follow-ups.**
- "When does MLE = MAP?" → Uniform prior on `θ`. Limit of MAP as the prior becomes uninformative.
- "Why doesn't MLE work great on small data?" → Overfits; needs regularization (which is implicit MAP).

**Common mistakes.**
- Treating MAP as "the Bayesian answer" — full Bayes integrates over the posterior; MAP is a single-point estimate.

**References.**
- [Murphy — *Probabilistic Machine Learning*, §4.5](https://probml.github.io/pml-book/book1.html) — MLE/MAP.
- [Bishop — *Pattern Recognition*, §1.2.5](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — Bayesian estimation.

---

### Q: Explain naive Bayes. Why is it called "naive"?

**Category:** concept
**Difficulty:** intro
**Tags:** [naive-bayes, generative, conditional-independence]

**Short answer.** Naive Bayes assumes features are conditionally independent given the class: `p(x | y) = Π p(x_i | y)`. Then by Bayes' rule, `p(y | x) ∝ p(y) · Π p(x_i | y)`. "Naive" because the conditional-independence assumption is rarely true; nonetheless it often works well, especially for high-dim sparse data (text classification with bag-of-words).

**Expansion / why this is the answer.**
- Training: estimate `p(y)` and `p(x_i | y)` from data (Bernoulli, multinomial, or Gaussian variants).
- Prediction: pick class maximizing the joint.
- Famous text-classification baseline; surprisingly strong despite the wrong assumption (the rank ordering of classes is often correct even when probabilities are miscalibrated).
- Modern context: rarely used at scale; replaced by logistic regression / GBMs / DL. Still useful as a quick baseline.

**Common follow-ups.**
- "Why does it work despite the assumption being wrong?" → Domingos & Pazzani 1997: the conditional-independence assumption is most damaging to *calibrated* probabilities but barely affects the argmax decision.
- "Multinomial vs Bernoulli naive Bayes?" → Multinomial: feature is a count (word counts). Bernoulli: feature is 0/1 (word presence).

**Common mistakes.**
- Trusting NB's predicted probabilities as calibrated (they typically aren't).

**References.**
- [Domingos & Pazzani — "On the Optimality of the Simple Bayesian Classifier"](https://link.springer.com/article/10.1023/A:1007413511361).

---

### Q: Walk through entropy, cross-entropy, KL divergence, and mutual information.

**Category:** derivation
**Difficulty:** mid
**Tags:** [information-theory, entropy, kl, mutual-information]

**Short answer.** **Entropy** `H(p) = −Σ p(x) log p(x)`: average uncertainty in `p`. **Cross-entropy** `H(p, q) = −Σ p(x) log q(x)`: average code length using `q` to encode samples from `p`. **KL divergence** `KL(p || q) = Σ p(x) log(p(x)/q(x)) = H(p, q) − H(p)`: extra cost of using `q` instead of `p`. **Mutual information** `I(X; Y) = H(X) − H(X | Y) = KL(p(x,y) || p(x)p(y))`: information shared between `X` and `Y`.

**Expansion / why this is the answer.**
- All are foundational to ML loss design.
- Cross-entropy loss `= H(p, q)` minimized when `q = p`.
- Minimizing cross-entropy ⇔ minimizing KL ⇔ MLE for the model distribution.
- KL is asymmetric: `KL(p || q) ≠ KL(q || p)`. Reverse KL (used in variational inference) gives mode-seeking behavior; forward KL is mean-seeking.
- Mutual information is symmetric: `I(X; Y) = I(Y; X)`.
- Conditional entropy: `H(X | Y) = H(X, Y) − H(Y)`.

**Common follow-ups.**
- "Why is reverse KL used in VAEs?" → `KL(q || p)`: avoids overestimating probability mass where `p` is small.
- "What's variational free energy?" → ELBO = `E_q[log p(x | z)] - KL(q(z|x) || p(z))`.

**Common mistakes.**
- Forgetting KL's asymmetry.
- Saying "cross-entropy loss measures distance" — it doesn't satisfy the triangle inequality.

**References.**
- [Cover & Thomas — *Elements of Information Theory*](https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959) — canonical.
- [Murphy — *Probabilistic Machine Learning*, §6](https://probml.github.io/pml-book/book1.html) — info theory for ML.

---

### Q: What's the difference between feature engineering, feature selection, and feature extraction?

**Category:** concept
**Difficulty:** intro
**Tags:** [features, pipeline, preprocessing]

**Short answer.** **Feature engineering**: hand-design new features from raw data (e.g. extract `day_of_week` from timestamp). **Feature selection**: pick a subset of existing features (e.g. drop low-variance, L1-induced sparsity, mutual-information ranking). **Feature extraction**: learn a lower-dim representation (PCA, autoencoders, embeddings).

**Expansion / why this is the answer.**
- Feature engineering is the human-in-the-loop step. Still dominant in tabular ML; deep learning has subsumed it in vision / language by learning features end-to-end.
- Feature selection types: filter (univariate stats), wrapper (model-based, e.g. RFE), embedded (L1).
- Feature extraction types: linear (PCA, LDA), non-linear (kernel PCA, autoencoders, learned embeddings).

**Common follow-ups.**
- "When is feature engineering most valuable?" → Tabular ML, small data, when domain expertise outpaces what a generic model can learn.
- "Why don't LLMs need feature engineering?" → Tokens are the features; the model learns representations.

**Common mistakes.**
- Conflating selection (pick from existing) with extraction (compute new).

**References.**
- [scikit-learn — feature selection guide](https://scikit-learn.org/stable/modules/feature_selection.html).

---

### Q: When does Bayesian optimization beat random search for hyperparameter tuning?

**Category:** concept
**Difficulty:** mid
**Tags:** [hyperparameter-tuning, bayesian-optimization, search]

**Short answer.** Bayesian optimization (with a Gaussian-process or tree-Parzen surrogate) wins when each evaluation is *expensive* and the search space is *modest* (≤ 10–20 dims). It uses past evaluations to model the loss landscape and propose the next trial more informatively. For very large or non-smooth search spaces, or when evaluations are cheap, random search wins (Bergstra & Bengio 2012). For massively parallel runs, Hyperband / ASHA dominate.

**Expansion / why this is the answer.**
- **Random search**: simple, embarrassingly parallel, surprisingly strong baseline.
- **Grid search**: dense in low dims; combinatorially expensive in high dims.
- **Bayesian optimization** (BO): fit a surrogate to seen `(hyperparams, val_loss)` points; the acquisition function (expected improvement, UCB) suggests the next point. Sequential by design.
- **Hyperband / ASHA**: early-stopping unpromising trials. Good when training is iterative.
- **PBT** (Population-Based Training): evolves a population during training.
- **Modern picks**: Optuna (TPE-based BO + pruning), Ray Tune (ASHA + BO + PBT).

**Common follow-ups.**
- "Why doesn't BO work in 100+ dim search spaces?" → GP scales `O(N³)`; high-dim posterior modeling breaks down; random search becomes competitive.
- "What's TPE?" → Tree-structured Parzen Estimator — Bayesian-optimization variant used in Optuna.

**Common mistakes.**
- Defaulting to grid search in high-dim spaces.
- Using BO with very fast evaluations (parallelism + random wins).

**References.**
- [Bergstra & Bengio — "Random Search for Hyper-Parameter Optimization"](https://www.jmlr.org/papers/v13/bergstra12a.html).
- [Li et al. — "Hyperband"](https://arxiv.org/abs/1603.06560).
- [Optuna docs](https://optuna.org/).

---

### Q: What is SHAP, and how does it improve over feature importance?

**Category:** concept
**Difficulty:** mid
**Tags:** [shap, interpretability, feature-attribution]

**Short answer.** SHAP (SHapley Additive exPlanations, Lundberg & Lee 2017) assigns each feature an attribution for a *specific prediction* based on Shapley values from cooperative game theory. Unique among feature-attribution methods in satisfying local accuracy, missingness, and consistency. Better than gain / permutation importance because it gives *per-instance* attributions (instead of just global), is consistent across model changes, and has theoretical grounding.

**Expansion / why this is the answer.**
- **Global feature importance** (gain in GBMs, permutation importance): aggregate signal across the dataset. Hides per-instance variation.
- **SHAP**: for prediction `f(x)`, compute `φ_i = Σ_{S} (|S|! (n-|S|-1)!/n!) [f(S ∪ {i}) − f(S)]` over all subsets — the average marginal contribution of feature `i`.
- Computationally heavy in general (`O(2^n)`); TreeSHAP exploits tree structure for polynomial-time exact SHAP on tree ensembles.
- Sum of SHAP values for a prediction = `f(x) − E[f]` (local accuracy).
- **Visualizations**: waterfall plot per prediction, beeswarm plot for summary.

**Common follow-ups.**
- "Why is SHAP slow for non-tree models?" → Combinatorial in number of features; sampling approximations help (KernelSHAP).
- "What's the difference between SHAP and LIME?" → LIME fits a local linear model around the prediction; SHAP uses Shapley values; SHAP has stronger theoretical guarantees and consistency.

**Common mistakes.**
- Trusting SHAP for causality — it's attribution within the model, not causal effect on the world.
- Using SHAP from the wrong baseline (the "average" matters).

**References.**
- [Lundberg & Lee — "A Unified Approach to Interpreting Model Predictions" (SHAP)](https://arxiv.org/abs/1705.07874).
- [Lundberg et al. — "TreeSHAP"](https://arxiv.org/abs/1802.03888).

---

### Q: What is the bootstrap, and what is it used for?

**Category:** concept
**Difficulty:** intro
**Tags:** [bootstrap, resampling, confidence-intervals]

**Short answer.** The bootstrap resamples the dataset with replacement to estimate the sampling distribution of a statistic — typically a confidence interval or standard error. Effron (1979). Useful when you don't have a clean analytic CI: medians, complex metrics like AUC, model performance numbers, etc.

**Expansion / why this is the answer.**
- Procedure: draw `B` resamples (e.g. `B = 1000`) of size `N` with replacement; compute the statistic on each; the empirical distribution of those `B` statistics approximates the sampling distribution.
- Confidence interval: percentile method (2.5 and 97.5 quantiles for 95% CI).
- Standard error: std of the bootstrap distribution.
- Uses in ML:
  - CI on metric (AUC, accuracy, RMSE).
  - Bagging (bootstrap aggregation; Random Forest uses bootstrap samples).
  - Out-of-bag (OOB) estimate of generalization error.
- Limits: assumes the empirical distribution approximates the true distribution; fails for heavy-tailed or boundary-near data.

**Common follow-ups.**
- "How many bootstrap samples?" → 1000+ for stable CI; 10k for tight tails.
- "What's the OOB estimate in Random Forest?" → For each tree, evaluate on the samples not in its bootstrap; average across trees.

**Common mistakes.**
- Bootstrapping without replacement (defeats the purpose).
- Reporting a bootstrap CI that's too narrow because `B` was too small.

**References.**
- [Efron & Tibshirani — *An Introduction to the Bootstrap*](https://www.routledge.com/An-Introduction-to-the-Bootstrap/Efron-Tibshirani/p/book/9780412042317).

---

### Q: Multi-armed bandits — Thompson sampling vs UCB.

**Category:** concept
**Difficulty:** mid
**Tags:** [bandits, thompson, ucb, exploration]

**Short answer.** Multi-armed bandit: trade off exploring arms with uncertain reward vs. exploiting the current best. **UCB** (Upper Confidence Bound): pick the arm with highest `mean + c·√(log(t)/n_arm)` — deterministic, optimistic. **Thompson sampling**: sample a parameter from each arm's posterior; pick the arm with the best sample — stochastic, Bayesian. Both achieve `O(log T)` regret; Thompson is empirically often simpler and more robust.

**Expansion / why this is the answer.**
- Standard bandit: `K` arms; each pull gives a stochastic reward; maximize cumulative reward over `T` pulls.
- Regret: `T · μ* − Σ_t μ_{a_t}` (gap to playing the optimal arm).
- **UCB1**: `arm_t = argmax_a (μ̂_a + √(2 log t / n_a))`.
- **Thompson sampling**: with conjugate prior (e.g. Beta-Bernoulli for binary rewards), sample `θ_a ∼ Beta(α_a, β_a)`, pick `argmax θ_a`, update with observed reward.
- **Contextual bandits**: arm choice conditions on a context vector (used in recsys, ads); linear (LinUCB), neural-network variants.
- **Use cases in ML**: A/B testing with adaptive allocation, recsys cold-start, prompt selection.

**Common follow-ups.**
- "When does Thompson beat UCB in practice?" → Often comparable; Thompson has lower-variance regret and is easier to extend (LinTS, NeuralTS).
- "Why is Thompson Bayesian and UCB frequentist?" → TS samples from a posterior; UCB constructs a frequentist upper confidence bound.

**Common mistakes.**
- Treating bandits as full RL — they have no state transitions.
- Forgetting the exploration term decays over time.

**References.**
- [Auer, Cesa-Bianchi, Fischer — "Finite-time Analysis of the Multiarmed Bandit Problem" (UCB1)](https://link.springer.com/article/10.1023/A:1013689704352).
- [Russo et al. — "A Tutorial on Thompson Sampling"](https://arxiv.org/abs/1707.02038).

---

### Q: Concept drift vs data drift vs label drift — definitions and detection.

**Category:** concept
**Difficulty:** mid
**Tags:** [drift, monitoring, production-ml]

**Short answer.** **Data drift** (covariate shift): `p(X)` changes; same `p(Y|X)`. Inputs shift but the relationship is stable. **Label drift** (prior shift): `p(Y)` changes; e.g., positive rate rises. **Concept drift**: `p(Y|X)` changes; the underlying relationship itself shifts. All three degrade model performance; you detect with population stability index (PSI), KS test, or comparing per-segment metrics over time.

**Expansion / why this is the answer.**
- **Data drift** examples: new user segments arrive; seasonality changes; product mix shifts.
- **Label drift** examples: pandemic causes fraud-rate spike; election cycle shifts conversion rates.
- **Concept drift** examples: user preferences shift; a new product category emerges that the model never saw.
- **Detection**:
  - Population Stability Index (PSI): per-feature binning, compare distributions; threshold (e.g. 0.2 = major shift).
  - KS test for continuous, chi-square for categorical.
  - Statistical-process control on input distributions.
  - Performance monitoring: compare per-period metrics.
- **Response**:
  - Data drift: usually retrain on recent data.
  - Label drift: recalibrate (Platt) or retrain.
  - Concept drift: retrain; possibly redesign features.

**Common follow-ups.**
- "When can data drift not cause performance degradation?" → If the new region is in-distribution and the model generalizes; rare but possible.
- "What's covariate shift?" → Synonym for data drift.

**Common mistakes.**
- Conflating these — they require different responses.
- Only monitoring outputs (you miss input shifts before they cause errors).

**References.**
- [Quionero-Candela et al. — *Dataset Shift in Machine Learning*](https://mitpress.mit.edu/9780262170055/dataset-shift-in-machine-learning/) — canonical reference.
- [Lu et al. — "Learning under Concept Drift: A Review"](https://arxiv.org/abs/1907.10202).

---

### Q: What is online learning, and when does it beat batch learning?

**Category:** concept
**Difficulty:** intro
**Tags:** [online-learning, streaming, ftrl]

**Short answer.** Online learning updates the model one example (or mini-batch) at a time as data arrives, rather than retraining from scratch. Beats batch learning when (a) data arrives in a stream, (b) the underlying distribution drifts and recency matters, (c) compute can't fit a full retrain. Classic algorithms: FTRL (used by Google for ad CTR), online SGD, passive-aggressive.

**Expansion / why this is the answer.**
- Online learning vs. incremental learning vs. continual learning:
  - **Online**: one example at a time; can't revisit data.
  - **Incremental**: update on new data; may revisit old.
  - **Continual**: update across tasks over time; mitigate forgetting.
- **FTRL-Proximal** (McMahan et al. 2013): per-coordinate adaptive LR; widely used for ad CTR.
- **Online SGD**: simplest form.
- **Passive-Aggressive**: update only when current example is misclassified.
- **Modern context**:
  - Production ML systems do hourly-to-daily batch retrains rather than true online.
  - Pure online learning is in niches (real-time recsys, ad CTR).
  - LLMs: not online-learned in the typical sense; pretrain + periodic fine-tune.

**Common follow-ups.**
- "Why isn't pure online learning more common?" → Stability concerns; one bad example can drift the model; harder to debug.
- "What's the regret of online learning?" → A standard analysis bound; OGD achieves `O(√T)` regret.

**Common mistakes.**
- Confusing online learning (data) with online inference (deployment).

**References.**
- [McMahan et al. — "Ad Click Prediction: a View from the Trenches"](https://research.google/pubs/pub41159/) — FTRL.
- [Shalev-Shwartz — "Online Learning and Online Convex Optimization"](https://arxiv.org/abs/1909.05207) — survey.

---

### Q: Demographic parity vs equalized odds vs calibration — which fairness metric, when?

**Category:** concept
**Difficulty:** senior
**Tags:** [fairness, equalized-odds, demographic-parity]

**Short answer.** **Demographic parity**: equal positive-prediction rate across groups. **Equalized odds**: equal TPR and FPR across groups. **Calibration**: predicted probabilities mean the same thing across groups. They're mutually incompatible (Chouldechova 2017; Kleinberg et al. 2017) when base rates differ. The right choice depends on the use case — there's no universally "right" fairness metric.

**Expansion / why this is the answer.**
- **Demographic parity**: `P(ŷ=1 | A=0) = P(ŷ=1 | A=1)`. Strong condition; ignores label.
- **Equalized odds**: `P(ŷ=1 | y, A=0) = P(ŷ=1 | y, A=1)` for `y ∈ {0,1}` — equal TPR and FPR.
- **Calibration**: `P(y=1 | ŷ=p, A) = p` for all groups `A`.
- **Impossibility**: if base rates `P(y=1 | A)` differ between groups, you can't satisfy all three simultaneously (Chouldechova 2017).
- **Context determines choice**:
  - Lending / hiring: equalized odds often preferred (don't underserve qualified people in any group).
  - Criminal-risk prediction (COMPAS): calibrated by group, but FPR differs by race — fundamental tension.
  - Marketing: demographic parity may be relevant for inclusion goals.
- **Mitigation**:
  - Re-weighting training data.
  - Post-hoc threshold adjustment per group (where legally permitted).
  - In-processing constraints (regularization toward fairness).
- **Real world**:
  - "Disparate treatment" vs. "disparate impact" — different legal frames.
  - Many uses must satisfy domain-specific regulations (ECOA, GDPR, EU AI Act).

**Common follow-ups.**
- "What's the COMPAS controversy?" → ProPublica showed COMPAS had different FPR by race; Northpointe responded that it's calibrated by race. Both correct — the impossibility theorem.
- "Is there a 'fair' classifier?" → No general one; the answer depends on what notion of fairness is operative for the domain.

**Common mistakes.**
- Treating fairness as a single technical metric.
- Forgetting the impossibility result.

**References.**
- [Chouldechova — "Fair Prediction with Disparate Impact"](https://arxiv.org/abs/1610.07524).
- [Kleinberg, Mullainathan, Raghavan — "Inherent Trade-offs in the Fair Determination of Risk Scores"](https://arxiv.org/abs/1609.05807).

---

### Q: What's differential privacy, and how would you train an LLM with it?

**Category:** concept
**Difficulty:** senior
**Tags:** [differential-privacy, dp-sgd, privacy]

**Short answer.** Differential Privacy (DP) bounds how much one training example can change the model's output distribution — formalized as the `(ε, δ)`-DP guarantee. **DP-SGD** (Abadi et al. 2016): per-example gradient clipping + Gaussian noise injection. Trades off privacy budget against utility. For LLMs at frontier scale, end-to-end DP training is rarely done (utility costs are high); selective DP fine-tuning is more common.

**Expansion / why this is the answer.**
- **Definition**: a mechanism `M` is `(ε, δ)`-DP if for adjacent datasets `D, D'` (differing in one record), `P(M(D) ∈ S) ≤ e^ε · P(M(D') ∈ S) + δ`.
- **DP-SGD**:
  - Compute per-example gradients.
  - Clip each gradient to norm ≤ `C`.
  - Sum and add Gaussian noise `N(0, σ² C² I)`.
  - Step.
- **Privacy accounting**: each step consumes privacy budget; total `ε` grows with steps (composition theorems — moments accountant for tight bounds).
- **Practical for LLMs**:
  - Full pretraining with DP is expensive (utility-budget tradeoff is hard).
  - DP-SFT (fine-tune with DP) is more common, especially for medical / legal data.
  - **Private prediction** (DP at inference time): output noisy predictions; rare for LLMs.
- **Membership inference attacks** quantify how much a model leaks about its training data; DP mitigates this.
- **Modern context**: Apple's on-device privacy, federated-learning + DP combinations.

**Common follow-ups.**
- "What's ε in practice?" → `ε ≤ 1` strong; `ε ≤ 10` weak. Domain-dependent.
- "Can you DP-tune a public model?" → The pretraining isn't DP, so the model's parametric memory leaks. DP-fine-tune adds privacy *for the fine-tune data*, not for pretraining.

**Common mistakes.**
- Treating DP as binary; it's a continuous budget.
- Forgetting that DP bounds privacy *of the algorithm*, not the deployment.

**References.**
- [Abadi et al. — "Deep Learning with Differential Privacy"](https://arxiv.org/abs/1607.00133) — DP-SGD.
- [Dwork & Roth — *The Algorithmic Foundations of Differential Privacy*](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) — canonical text.

---

### Q: What is active learning, and when does it help?

**Category:** concept
**Difficulty:** mid
**Tags:** [active-learning, sample-efficiency, labeling]

**Short answer.** Active learning: the model selects the most informative unlabeled examples for a human to label. Beats random sampling when (a) labeling is expensive but data is abundant, and (b) the model's uncertainty is well-calibrated. Common acquisition criteria: highest model uncertainty, query-by-committee disagreement, expected model change. Gains plateau after a few thousand labels; rarely transformative for modern LLM pretraining (the abundance of unlabeled text dominates).

**Expansion / why this is the answer.**
- The classical setup: model trained on small `L`; large unlabeled pool `U`; pick examples from `U` for the human to label.
- **Acquisition functions**:
  - **Uncertainty sampling**: query the example the model is most uncertain on (highest entropy).
  - **Query-by-committee**: train an ensemble; query where they disagree.
  - **Expected model change**: query where adding the label would change the model most.
  - **BALD**: Bayesian Active Learning by Disagreement; based on mutual information.
- **Modern context**:
  - Annotation tools (Snorkel, Label Studio) implement active-learning loops.
  - For LLM fine-tuning: active learning can pick the most informative SFT data.
  - For LLM eval-set construction: active labeling of the hardest examples.

**Common follow-ups.**
- "Why does active learning sometimes fail?" → Uncertain predictions are sometimes uninformative (noisy regions); the model selects "garbage."
- "BALD vs. uncertainty sampling?" → BALD captures model-vs-data uncertainty separation; better than naive entropy.

**Common mistakes.**
- Trusting model uncertainty before the model is calibrated.

**References.**
- [Settles — "Active Learning Literature Survey"](https://burrsettles.com/pub/settles.activelearning.pdf).

---

### Q: Compare semi-supervised learning approaches.

**Category:** concept
**Difficulty:** mid
**Tags:** [semi-supervised, pseudo-labels, fixmatch]

**Short answer.** Semi-supervised learning uses both labeled and unlabeled data. Common methods: **pseudo-labeling** (use model's confident predictions as labels), **consistency regularization** (encourage same prediction on augmented variants of unlabeled data), **co-training** (two views of data train two models that supervise each other), **graph-based** (label propagation on a graph). Modern: **FixMatch** (Sohn et al. 2020) combines pseudo-labels + consistency for strong vision SSL. SSL also fundamentally underlies LLM self-supervised pretraining.

**Expansion / why this is the answer.**
- **Pseudo-labeling** (Lee 2013): train on labeled, predict on unlabeled, keep predictions with high confidence as new labels, retrain.
- **Consistency regularization** (Π-model, Mean Teacher, MixMatch, FixMatch): augment unlabeled data; loss = MSE/KL between predictions on different augmentations.
- **MixMatch / FixMatch**: combine these; strong-augmentation pseudo-labels supervise weak-augmentation predictions.
- **Graph-based** (label propagation): build a similarity graph; propagate labels from labeled to unlabeled along edges.
- **Self-supervised pretraining + fine-tune**: arguably the dominant modern SSL — pretrain on huge unlabeled data, fine-tune on small labeled. LLM pretraining is the headline example.

**Common follow-ups.**
- "When does pseudo-labeling fail?" → Confirmation bias: model amplifies its own errors. Mitigate with confidence thresholding and ensembles.
- "Is LLM pretraining 'semi-supervised'?" → Technically self-supervised; sometimes lumped in.

**Common mistakes.**
- Treating semi-supervised and self-supervised as identical.

**References.**
- [Sohn et al. — "FixMatch"](https://arxiv.org/abs/2001.07685).
- [Berthelot et al. — "MixMatch"](https://arxiv.org/abs/1905.02249).
- [Lee — "Pseudo-Label"](https://www.researchgate.net/publication/280581078_Pseudo-Label_The_Simple_and_Efficient_Semi-Supervised_Learning_Method_for_Deep_Neural_Networks).

---

### Q: Confidence interval vs prediction interval — what's the difference?

**Category:** concept
**Difficulty:** intro
**Tags:** [intervals, uncertainty, regression]

**Short answer.** **Confidence interval (CI)**: range likely to contain a *population parameter* (e.g., the true mean). Reflects sampling uncertainty. **Prediction interval (PI)**: range likely to contain a *new individual observation*. Reflects sampling + irreducible noise. PIs are always wider than CIs because they include the per-observation noise term.

**Expansion / why this is the answer.**
- For a regression mean prediction at `x*`:
  - 95% CI for `E[Y | x*]`: `ŷ ± 1.96 · SE(ŷ)`.
  - 95% PI for `Y_new(x*)`: `ŷ ± 1.96 · √(SE(ŷ)² + σ²)`, where `σ²` is residual variance.
- The `σ²` term is irreducible noise; CI ignores it.
- For ML: bootstrap can give a CI; quantile regression gives a PI.

**Common follow-ups.**
- "How would you build a PI from a neural net?" → Quantile regression heads; conformal prediction.
- "Conformal prediction?" → Distribution-free PIs with finite-sample coverage guarantees (Vovk et al.; modern: Romano et al. 2019 conformal quantile regression).

**Common mistakes.**
- Reporting a 95% CI as if it covers individual predictions.

**References.**
- [Angelopoulos & Bates — "A Gentle Introduction to Conformal Prediction"](https://arxiv.org/abs/2107.07511).

---

### Q: When does ensembling help in deep learning, and what are the modern alternatives?

**Category:** concept
**Difficulty:** mid
**Tags:** [ensembling, deep-ensembles, snapshot, dropout]

**Short answer.** Ensembling reduces variance and improves calibration. **Deep ensembles**: train N independent models with different seeds; average predictions. Strong, reliably improves AUC / log-loss / calibration. **Snapshot ensembles**: collect checkpoints at cyclical LR minima from one training run — much cheaper but weaker. **MC Dropout** at inference: cheap Bayesian-ish ensemble. **Self-distillation**: distill the ensemble into one model for cheap inference. At scale (LLMs), full deep ensembles are prohibitive; alternatives dominate.

**Expansion / why this is the answer.**
- **Why ensembles help**: independent errors average out; variance reduction.
- **Deep ensembles** (Lakshminarayanan et al. 2017): the strongest baseline for prediction-uncertainty and out-of-distribution detection.
- **Snapshot ensembles** (Huang et al. 2017): cyclic LR; collect checkpoints at the bottom of each cycle.
- **Stochastic Weight Averaging (SWA)** (Izmailov et al. 2018): average weights along the SGD trajectory; surprisingly improves generalization without ensemble cost at inference.
- **MC Dropout** (Gal & Ghahramani 2016): keep dropout on at inference; average many forward passes.
- **For LLMs**:
  - Full ensembling is too expensive.
  - Self-consistency at decoding (sample multiple chains-of-thought, majority-vote) is the analog.
  - Multi-temperature sampling + aggregation.

**Common follow-ups.**
- "When does ensembling not help?" → Already well-calibrated, low-variance models. Or when ensembled models are too similar (correlated errors).
- "What's the trade-off vs. just training one bigger model?" → A bigger model is often comparable; ensembling is a way to get the benefit when compute can't be reallocated to scaling.

**Common mistakes.**
- Averaging ensemble *probabilities* but reporting *predictions* — argmax over averaged probs is the right way.

**References.**
- [Lakshminarayanan et al. — "Deep Ensembles"](https://arxiv.org/abs/1612.01474).
- [Huang et al. — "Snapshot Ensembles"](https://arxiv.org/abs/1704.00109).
- [Izmailov et al. — "SWA"](https://arxiv.org/abs/1803.05407).

---

### Q: What's the difference between transfer learning, fine-tuning, and domain adaptation?

**Category:** concept
**Difficulty:** intro
**Tags:** [transfer-learning, fine-tuning, domain-adaptation]

**Short answer.** **Transfer learning**: take a pretrained model and apply it (possibly with fine-tuning) to a new task or domain. The umbrella term. **Fine-tuning**: a specific transfer-learning method — update some/all parameters on the new task. **Domain adaptation**: a sub-case of transfer learning where the *task* stays the same but the *input distribution* changes (English → German sentiment; web text → medical text).

**Expansion / why this is the answer.**
- **Transfer learning** is the broad concept; many methods underneath.
- **Methods**:
  - Linear probing: freeze backbone, train a small head.
  - Fine-tuning: update some/all weights.
  - PEFT (LoRA): low-rank update.
  - Prompting / in-context: no weight changes.
- **Domain adaptation** specifically: task = same; data distribution = different.
  - **Unsupervised DA**: labeled source, unlabeled target.
  - **Adversarial DA** (DANN, Ganin 2016): train a feature representation indistinguishable across domains.
- **Modern LLM use**: pretrain on web → SFT on chat → DPO on preferences = a chain of transfer steps. Each is "transfer" in the broad sense.

**Common follow-ups.**
- "What's negative transfer?" → When transferring hurts performance; the source and target are too different.
- "When is linear probing enough vs. full fine-tune?" → Linear probing for diagnostic / quick baseline; fine-tune when the task needs new behaviors.

**Common mistakes.**
- Treating fine-tuning as the only transfer-learning method.

**References.**
- [Pan & Yang — "A Survey on Transfer Learning"](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf).
- [Ganin et al. — "DANN"](https://arxiv.org/abs/1409.7495) — adversarial DA.

---

### Q: What's the difference between zero-shot and few-shot learning vs. zero-shot and few-shot evaluation?

**Category:** concept
**Difficulty:** intro
**Tags:** [few-shot, zero-shot, evaluation]

**Short answer.** "Zero-/few-shot **learning**" originally meant training a model to generalize to unseen classes with zero or few examples (Lampert et al. 2009; metric-learning prototypes). "Zero-/few-shot **evaluation**" (the 2020+ LLM sense) means prompting an LLM with zero or `k` in-context examples and measuring performance. Different things; the LLM-era usage has dominated, but the older meaning still appears in vision papers.

**Expansion / why this is the answer.**
- **Classical few-shot learning** (vision):
  - Train on `N` base classes; at test time, see a few examples of new classes; classify new examples in those classes.
  - Methods: prototypical networks, MAML (meta-learning).
- **LLM zero/few-shot evaluation** (Brown et al. 2020):
  - Take a pretrained LLM (no task-specific training).
  - Zero-shot: just the task instruction.
  - Few-shot: instruction + `k` `(input, output)` examples in the prompt.
- **In-context learning**: the LLM does few-shot via in-context examples — no weight updates.

**Common follow-ups.**
- "Why is the LLM zero-shot still surprising?" → Because the model can perform tasks it wasn't explicitly trained on, just from the prompt.
- "Connection to prompt engineering?" → Prompt engineering is mostly few-shot / zero-shot with instruction tuning.

**Common mistakes.**
- Confusing the two senses; some papers ambiguous.

**References.**
- [Brown et al. — GPT-3](https://arxiv.org/abs/2005.14165) — LLM few-shot framework.
- [Lampert et al. — "Learning to Detect Unseen Object Classes by Between-Class Attribute Transfer"](https://www.di.ens.fr/~lampert/papers/2009-cvpr-lampert.pdf) — classical.

---

### Q: What is meta-learning / "learning to learn"?

**Category:** concept
**Difficulty:** senior
**Tags:** [meta-learning, maml, few-shot]

**Short answer.** Meta-learning trains a model on a *distribution of tasks* so that it can quickly adapt to a new task with few examples. The headline algorithm is **MAML** (Finn et al. 2017): the meta-objective is "after one gradient step on a new task, perform well." Other approaches: prototypical networks (Snell et al. 2017), Reptile, ANIL. Modern LLM in-context learning is sometimes framed as implicit meta-learning emerging from pretraining.

**Expansion / why this is the answer.**
- The setup: tasks `T_1, T_2, ...` sampled from a distribution. Each task has its own `(D_train, D_test)`.
- **MAML**: meta-train so that `θ - α∇_θ L(θ, D_train_τ)` performs well on `D_test_τ` after one gradient step.
- **Prototypical networks**: learn a feature embedding such that "prototype" (mean) of each class in the support set classifies the query set.
- **Modern relevance**:
  - Vision few-shot benchmarks (miniImageNet, Omniglot).
  - LLM in-context learning is *not* explicit meta-learning, but the behavior has parallels; some research formalizes the connection (von Oswald et al. 2023).

**Common follow-ups.**
- "Why isn't MAML widely used now?" → LLMs and pretrained foundation models have largely replaced explicit meta-learning by providing strong base features.

**Common mistakes.**
- Calling all transfer learning "meta-learning" — meta-learning trains *for* fast adaptation; transfer learning just *uses* a pretrained model.

**References.**
- [Finn et al. — "MAML"](https://arxiv.org/abs/1703.03400).
- [Snell et al. — "Prototypical Networks"](https://arxiv.org/abs/1703.05175).

---

### Q: What is curriculum / self-paced learning, and when does it help in DL?

**Category:** concept
**Difficulty:** mid
**Tags:** [curriculum-learning, self-paced]

**Short answer.** Curriculum learning trains on easy examples first, gradually adding harder ones. Self-paced learning lets the model itself decide example difficulty as it trains. For deep learning on classical supervised tasks, gains are modest and inconsistent. For specific settings — reasoning fine-tuning (math), RL-from-easy-to-hard, long-context training — curriculum can substantially help.

**Expansion / why this is the answer.**
- See also T3's curriculum-learning entry for the LLM-specific context.
- Empirical: works for some tasks (sequence learning, RL with sparse rewards) more than others (image classification).
- **Self-paced** (Kumar et al. 2010): model loss determines what's "easy"; weight examples by current model performance.
- **Modern LLM use**:
  - Math fine-tuning: easy problems first; harder problems later (DeepSeek-Math).
  - Long-context training: start with shorter contexts; extend.
- The Bengio et al. 2009 paper made the case; empirical support is mixed.

**Common follow-ups.**
- "Why doesn't it help much in image classification?" → ImageNet samples are reasonably i.i.d.; curriculum effects average out.
- "What's a good signal for 'easy'?" → Current loss, gradient norm, human-curated difficulty.

**Common mistakes.**
- Treating curriculum as universally helpful.

**References.**
- [Bengio et al. — "Curriculum Learning"](https://dl.acm.org/doi/10.1145/1553374.1553380).
- [Kumar et al. — "Self-Paced Learning"](https://papers.nips.cc/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html).

---

### Q: What is the universal approximation theorem, and does it actually matter in practice?

**Category:** concept
**Difficulty:** mid
**Tags:** [universal-approximation, theory, expressiveness]

**Short answer.** A 2-layer feedforward network with enough hidden units and a non-linear activation can approximate any continuous function arbitrarily well (Cybenko 1989; Hornik 1991). In practice, **it matters less than people think**: the theorem says nothing about (a) how many units are needed, (b) whether you can find the weights with gradient descent, (c) generalization. Depth, optimization landscape, and inductive biases matter more than expressiveness.

**Expansion / why this is the answer.**
- The theorem is an existence proof; it doesn't tell you how to construct or train.
- Modern deep learning's empirical success comes from:
  - **Depth** vs. width: deep networks express certain functions exponentially more compactly than shallow.
  - **Optimization**: SGD finds good minima even though the loss landscape is non-convex (lottery-ticket, mode connectivity).
  - **Inductive bias**: convolutions (translation equivariance), attention (relational), positional encodings (sequence).
- **What it does explain**: you don't need exotic architectures to achieve approximation; standard NNs are enough.
- **What it doesn't explain**: why deep networks generalize so well despite huge parameter counts.

**Common follow-ups.**
- "Why depth, then?" → Some functions need exponentially many shallow units but few deep ones (Telgarsky 2016).
- "Does this apply to transformers?" → Transformers are also universal approximators with some caveats (Yun et al. 2020).

**Common mistakes.**
- Citing universal approximation to argue "deep nets work because they're universal" — every nontrivial model class is universal.

**References.**
- [Cybenko — "Approximation by Superpositions of a Sigmoidal Function"](https://link.springer.com/article/10.1007/BF02551274).
- [Telgarsky — "Benefits of depth in neural networks"](https://arxiv.org/abs/1602.04485).

---

### Q: What is the lottery-ticket hypothesis?

**Category:** concept
**Difficulty:** senior
**Tags:** [lottery-ticket, pruning, sparsity]

**Short answer.** Frankle & Carbin (2019): inside a randomly-initialized dense network, there exists a sparse subnetwork ("winning ticket") that — *if trained in isolation from its original initialization* — matches the dense network's performance. Demonstrated by iterative magnitude pruning + re-rewinding. Suggests over-parameterization helps find good subnetworks at init, not just for expressiveness. Mixed replication at very large scale.

**Expansion / why this is the answer.**
- Procedure:
  1. Train a dense network.
  2. Prune the smallest-magnitude weights (e.g. 20%).
  3. **Rewind** the remaining weights to their original initialization.
  4. Retrain on the same data.
  5. Iterate.
- Winning ticket: the pruned subnetwork at the original init, retrained, matches the dense model.
- Implications:
  - Networks are over-parameterized; only a sparse subnetwork "matters."
  - Suggests dense training helps because it gives many candidate winning tickets in parallel.
- Caveats:
  - At very large scale (ImageNet ResNet, LLM), the "rewind to original init" version breaks; later work uses "rewind to early checkpoint."
  - Not consistently a path to faster training (the pruning process itself is expensive).

**Common follow-ups.**
- "Is this related to neural-network pruning?" → Same family; pruning typically prunes a trained network; lottery-ticket goes further by re-rewinding.
- "Practical takeaway?" → For deploying smaller models, pruning + fine-tune. Lottery-ticket is more a scientific finding than a recipe.

**Common mistakes.**
- Conflating lottery-ticket with standard pruning.

**References.**
- [Frankle & Carbin — "The Lottery Ticket Hypothesis"](https://arxiv.org/abs/1803.03635).

---

### Q: What's Goodhart's Law in ML, and where have you seen it?

**Category:** concept
**Difficulty:** mid
**Tags:** [goodhart, metrics, reward-hacking]

**Short answer.** "When a measure becomes a target, it ceases to be a good measure." In ML: optimizing for a proxy metric (test accuracy, reward model score, click-through rate) drifts the model toward gaming the metric rather than improving the underlying capability. Examples: reward hacking in RLHF, benchmark contamination, RLAIF length bias, recsys engagement-bait. Mitigation: monitor multiple metrics, hold out gold-set checks, periodically audit.

**Expansion / why this is the answer.**
- **Original framing**: economic theory; Goodhart 1975.
- **ML examples**:
  - Reward hacking in RL: policy maxes the reward model, gets verbose/sycophantic.
  - Benchmark over-fitting: model trained explicitly on benchmark questions or paraphrases.
  - Engagement metrics: recsys learns outrage drives clicks.
  - Code agents that delete tests to make them "pass."
- **Mitigations**:
  - Multi-metric optimization with guardrails.
  - Held-out gold sets sampled less often.
  - Adversarial probing.
  - Process supervision (not just outcome).
- **Strong version**: any sufficiently powerful optimizer will Goodhart any proxy metric. Defense in depth, not a single fix.

**Common follow-ups.**
- "What's an example from recsys?" → Optimizing CTR ⇒ clickbait. Solution: balance CTR with dwell, retention, satisfaction surveys.
- "Reward over-optimization curve?" → Gao, Schulman, Hilton (2022) — proxy reward rises while gold reward plateaus then drops.

**Common mistakes.**
- Treating Goodhart as a "theoretical concern" — it's a deployment-time reality.

**References.**
- [Gao, Schulman, Hilton — "Reward Model Overoptimization"](https://arxiv.org/abs/2210.10760).
- [Manheim & Garrabrant — "Categorizing Variants of Goodhart's Law"](https://arxiv.org/abs/1803.04585).

---

### Q: What is mode collapse, and where does it appear in ML?

**Category:** concept
**Difficulty:** mid
**Tags:** [mode-collapse, gans, sampling]

**Short answer.** Mode collapse: the model output distribution covers only a fraction of the target distribution. Most famously in GANs (the generator finds one mode that fools the discriminator and stops there). Also appears in: LLM generation (model repeats safe phrasing), RL policies (deterministic policy ignores high-reward exploration), and DPO ("decrease likelihood of both chosen and rejected" pathology).

**Expansion / why this is the answer.**
- **GAN mode collapse**: generator emits a narrow distribution; discriminator can't distinguish; equilibrium reached without diversity.
  - Mitigations: unrolled GANs, Wasserstein GAN, mode-regularizers, conditional GANs.
- **LLM mode collapse**: post-RLHF models can converge on safe / generic phrasings ("As an AI language model..."); reduces output diversity.
  - Mitigations: lower KL penalty β, temperature/top-p at inference, DPO with SFT auxiliary loss.
- **RL mode collapse**: deterministic policy is trapped in a local optimum; doesn't explore.
  - Mitigations: entropy regularization, exploration bonuses, ε-greedy.
- **DPO pathology** (Rafailov 2023): the loss can be minimized by lowering log-prob of both chosen and rejected; both fall.
  - Mitigation: SFT auxiliary loss.

**Common follow-ups.**
- "How do you detect mode collapse in LLMs?" → Output-diversity metrics (distinct-n, semantic-diversity); user reports of "model is repetitive."
- "Why is it called 'mode' collapse?" → Probability theory: a mode is a peak of the distribution; collapse = many modes lost.

**Common mistakes.**
- Confusing mode collapse with overfitting.

**References.**
- [Arjovsky et al. — "Wasserstein GAN"](https://arxiv.org/abs/1701.07875).
- [Rafailov et al. — "DPO"](https://arxiv.org/abs/2305.18290).

---

### Q: What is the bias-variance decomposition for cross-entropy / general losses?

**Category:** derivation
**Difficulty:** senior
**Tags:** [bias-variance, decomposition, theory]

**Short answer.** The bias-variance decomposition is cleanest for squared error: `E[(y - ŷ)²] = bias² + variance + irreducible noise`. For cross-entropy and general losses, an analogous decomposition exists (Heskes 1998) but is less clean — bias and variance interact non-additively for non-quadratic losses. Modern practice: report bias-variance qualitatively for classification, or use proper scoring rules (Brier score) for cleaner decompositions.

**Expansion / why this is the answer.**
- **MSE decomposition**: `E[(y − ŷ)²] = (E[ŷ] − f)² + E[(ŷ − E[ŷ])²] + σ²`.
- **For 0-1 loss**: similar shape but doesn't decompose additively; Domingos 2000 has a version.
- **For cross-entropy / log-loss**: Heskes (1998) — bias and variance defined via KL divergences from mean prediction.
- **Brier score**: `(p̂ − y)²` (binary), proper scoring rule, decomposes cleanly.
- **In practice**: people use the MSE-style decomposition as a *heuristic* for cross-entropy, knowing the math isn't strictly clean.

**Common follow-ups.**
- "Why does the decomposition not hold for arbitrary loss?" → Cross-entropy isn't quadratic in `ŷ`; the variance term doesn't split cleanly.
- "How do you measure variance empirically?" → Train `N` models with different seeds; measure spread of predictions on a held-out point.

**Common mistakes.**
- Citing "the bias-variance decomposition" for classification without acknowledging the math is loose.

**References.**
- [Heskes — "Bias-Variance Decompositions for Likelihood-Based Estimators"](https://www.researchgate.net/publication/220280066_Bias-Variance_Decompositions_for_Likelihood-Based_Estimators).
- [Domingos — "A Unified Bias-Variance Decomposition"](https://homes.cs.washington.edu/~pedrod/papers/mlc00a.pdf).

---

### Q: What is the no-free-lunch theorem, and what does it actually mean for ML?

**Category:** concept
**Difficulty:** mid
**Tags:** [no-free-lunch, theory, inductive-bias]

**Short answer.** Wolpert (1996): averaged over *all* possible target functions, every algorithm has equal expected performance. There's no universally best learner. In practice this means: **inductive bias matters** — algorithms that perform well do so because they're well-matched to the distribution of real-world problems, not because they're "universally good."

**Expansion / why this is the answer.**
- The theorem covers an unrealistic uniform distribution over all possible functions (most are random / unstructured).
- Real-world data has structure (smoothness, locality, compositionality); algorithms with matching inductive bias win.
- Examples of inductive bias:
  - CNNs: translation equivariance, spatial locality.
  - RNNs / Transformers: sequential / relational structure.
  - GBMs: piecewise-constant axis-aligned decision boundaries.
- The theorem is often cited to argue "no algorithm is universally best" — true but trivially, because real data isn't uniform.

**Common follow-ups.**
- "So is the theorem useful?" → As a philosophical reminder; not as a practical guide.
- "What's the inductive bias of a transformer?" → Permutation-equivariant attention + positional encoding = strong on sequence/relational data.

**Common mistakes.**
- Citing NFL to argue any specific algorithm is hopeless — the theorem applies to uniform over all functions, not to your specific problem.

**References.**
- [Wolpert — "The Lack of A Priori Distinctions Between Learning Algorithms"](https://www.semanticscholar.org/paper/The-Lack-of-A-Priori-Distinctions-Between-Learning-Wolpert/9a01015d3f12d50f9358b1090ed4d0acf7d020fc).

---

### Q: What's the difference between L1 / L2 / Huber loss for regression?

**Category:** concept
**Difficulty:** intro
**Tags:** [regression, loss, outliers, huber]

**Short answer.** **L2 (MSE)**: `(y − ŷ)²`; smooth gradient; sensitive to outliers (squared penalty amplifies them). **L1 (MAE)**: `|y − ŷ|`; robust to outliers; gradient non-smooth at zero (subgradient methods). **Huber**: quadratic for small errors, linear for large — robust *and* smooth. Modern regression: use Huber when data has outliers; MSE otherwise.

**Expansion / why this is the answer.**
- **L2 / MSE**: MLE under Gaussian noise; smooth; the workhorse.
- **L1 / MAE**: MLE under Laplace noise; gives the median, not the mean; robust.
- **Huber**: `L_δ(r) = ½r² if |r| ≤ δ else δ(|r| − ½δ)`. Smooth, robust.
- **Quantile loss**: pinball loss `max(τ(y − ŷ), (τ − 1)(y − ŷ))`. Gives the τ-th quantile; useful for prediction intervals.
- **Why outliers matter**: L2's squared penalty makes a single 10σ outlier dominate the sum.

**Common follow-ups.**
- "Why does L1 give the median?" → Minimizer of `Σ|y − ŷ|` is the median; minimizer of `Σ(y − ŷ)²` is the mean.
- "How is Huber different from clipping?" → Clipping zeroes large errors; Huber linearly penalizes them.

**Common mistakes.**
- Using MSE on data with heavy-tailed outliers; the model gets pulled toward them.

**References.**
- [Huber — "Robust Estimation of a Location Parameter"](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full).

---

### Q: How does a decision tree decide splits, and how does information gain compare to Gini?

**Category:** concept
**Difficulty:** intro
**Tags:** [decision-tree, information-gain, gini]

**Short answer.** A decision tree picks the split that maximally reduces impurity. **Information gain** uses entropy: `IG = H(parent) − Σ (n_child/n) · H(child)`. **Gini**: `Gini(p) = 1 − Σ p_i²`. Both are concave in `p`; in practice they yield nearly identical trees. Gini is faster (no log); CART uses Gini, ID3/C4.5 use information gain.

**Expansion / why this is the answer.**
- For each candidate feature × split point, compute the impurity reduction.
- Pick the split that maximizes the reduction; recurse on children.
- Stop when: pure node, max depth, min samples, no improvement.
- Information gain and Gini give very similar splits empirically.
- **Continuous features**: sort and consider thresholds between adjacent values.
- **Regression**: minimize variance instead of entropy/Gini.

**Common follow-ups.**
- "Why does gradient boosting not need info gain?" → GBMs fit residuals; the per-tree split criterion is the loss gradient.
- "What's pruning?" → Cost-complexity pruning; train deep tree, then prune back using a validation criterion.

**Common mistakes.**
- Saying info gain "is" entropy — it's a *reduction* in entropy.

**References.**
- [Breiman et al. — *Classification and Regression Trees*](https://www.routledge.com/Classification-and-Regression-Trees/Breiman-Friedman-Stone-Olshen/p/book/9780412048418) — CART.

---

### Q: What is the kernel trick mathematically?

**Category:** derivation
**Difficulty:** senior
**Tags:** [kernel-trick, svm, derivation]

**Short answer.** A kernel `K(x, y)` is a function such that `K(x, y) = ⟨φ(x), φ(y)⟩` for some feature map `φ`. Mercer's theorem says a symmetric positive-definite `K` always corresponds to some `φ` (possibly in an infinite-dimensional Hilbert space). The "trick" is that in any algorithm where inputs appear only via dot products, you can replace `⟨x, y⟩` with `K(x, y)` and never compute `φ` explicitly.

**Expansion / why this is the answer.**
- **The math**: linear algorithms (SVM, ridge regression, PCA) can often be re-expressed in terms of dot products.
  - SVM dual: `max Σ α_i − ½ Σ α_i α_j y_i y_j ⟨x_i, x_j⟩` subject to constraints. Replace `⟨x_i, x_j⟩` with `K(x_i, x_j)`.
- **Mercer's theorem** (informally): a continuous symmetric positive-definite kernel has an eigendecomposition giving the feature map.
- **Common kernels**:
  - Linear: `K(x, y) = x · y`.
  - Polynomial: `(x · y + c)^d`.
  - RBF: `exp(-γ ||x − y||²)`. Infinite-dimensional feature space.
- **Why "trick"**: you never compute `φ`; only `K`. For RBF, `φ` is infinite-dim — explicit computation impossible.
- **In modern DL**: largely supplanted by learned representations (deep nets compute features directly). Kernel methods linger in Gaussian processes, kernel-density-estimation, small-data tabular niches.

**Common follow-ups.**
- "What's a kernel that's NOT positive-definite?" → Sigmoid kernel `tanh(αx·y + c)`; conditionally positive but commonly used despite the technicality.
- "Connection to attention?" → The attention score `softmax(QKᵀ)V` can be viewed as a kernel-trick variant where the kernel is learned.

**Common mistakes.**
- Treating any function `K(x, y)` as a kernel; not all are valid (must be PSD).

**References.**
- [Schölkopf & Smola — *Learning with Kernels*](https://mitpress.mit.edu/9780262536578/learning-with-kernels/).

---

### Q: What is convex optimization, and why does it matter in ML?

**Category:** concept
**Difficulty:** mid
**Tags:** [convex-optimization, theory, optimization]

**Short answer.** Convex optimization: minimize a convex function over a convex constraint set. The defining property: every local minimum is global. ML algorithms that are convex (linear / logistic regression, SVM, Lasso, kernel ridge) have unique optima reachable by gradient descent. Modern deep learning is **non-convex**; we work with empirical convergence (SGD finds good-enough minima) rather than theoretical guarantees.

**Expansion / why this is the answer.**
- **Convex function**: `f(tx + (1-t)y) ≤ t f(x) + (1-t) f(y)`. Geometrically: any line segment between two points on the graph lies above the graph.
- **Convex ML problems**:
  - Linear regression with MSE.
  - Logistic regression (loss is convex in weights).
  - SVM (convex in the dual).
  - Lasso, ridge.
- **Non-convex ML problems**: any neural network.
- **Why convexity matters**:
  - Gradient descent converges to the global minimum.
  - Strong theoretical guarantees on convergence rate.
  - Convex problems are "solved" in principle.
- **For neural networks**: we have no convergence guarantees but empirically SGD works. Recent work (lottery-ticket, mode connectivity) explores why.

**Common follow-ups.**
- "Why is deep learning's loss landscape easier than expected?" → High dimensionality + over-parameterization means many local minima are near-global (Choromanska et al. 2015).
- "What's a convex optimization library?" → CVXPY for prototyping convex problems.

**Common mistakes.**
- Calling neural-network optimization "convex" because the loss is convex in *each layer* — it's non-convex in the joint parameters.

**References.**
- [Boyd & Vandenberghe — *Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/) — canonical text.

---

### Q: What's the difference between batch normalization at training time and inference time?

**Category:** concept
**Difficulty:** mid
**Tags:** [batch-norm, train-eval]

**Short answer.** At **training**, BN normalizes using the current batch's statistics (mean, std). At **inference**, BN uses running averages of batch statistics accumulated during training. This is a source of train-eval gaps — a model evaluated on a single sample uses a *different* normalization than at training, and these stats can drift if the deployment data distribution differs from training.

**Expansion / why this is the answer.**
- **Train**: per-batch `μ, σ` from `B` samples. Backprop through normalization.
- **Eval**: pre-computed running averages (typically exponential moving average across training batches).
- Why this matters:
  - **Batch-size sensitivity**: BN is poorly behaved at very small batch sizes (noisy `μ, σ`).
  - **Train-eval gap**: running statistics may differ from deployment-time true statistics if data drifts.
  - **Multi-GPU**: synchronized BN (SyncBN) averages stats across GPUs to avoid per-rank batch being too small.
- **Why LayerNorm replaces BN in transformers**:
  - Per-sample normalization; no batch dependence.
  - No train-eval gap.

**Common follow-ups.**
- "What's GroupNorm's solution to BN's small-batch issue?" → Normalize within groups of channels, batch-independent.
- "What if you train with BN, deploy with batch=1?" → Running stats kick in; no issue *if* the running stats are accurate for the deployment distribution.

**Common mistakes.**
- Forgetting to set the model to `eval()` mode at inference; BN behaves wrong.

**References.**
- [Ioffe & Szegedy — "Batch Normalization"](https://arxiv.org/abs/1502.03167).

---

### Q: What is the relationship between PCA and SVD?

**Category:** derivation
**Difficulty:** mid
**Tags:** [pca, svd, linear-algebra]

**Short answer.** PCA on a centered data matrix `X` is mathematically equivalent to taking the right singular vectors of `X`: if `X = UΣVᵀ`, then `V` are the principal directions and `Σ²/n` are the principal variances. So PCA *is* SVD (on centered data) — same computation, different framing.

**Expansion / why this is the answer.**
- Setup: data `X ∈ ℝ^{N×D}`, centered (column means subtracted).
- **PCA**: eigendecompose the covariance matrix `Σ_X = (1/N) XᵀX`. Eigenvectors = principal directions; eigenvalues = variances.
- **SVD**: `X = UΣVᵀ`. Then `XᵀX = V Σ² Vᵀ`. So `V` = principal directions; `Σ²/N` = principal variances.
- **Numerically**: SVD on `X` is more stable than eigendecomposing `XᵀX` (which squares the conditioning).
- **Whitening**: project to PCA basis and divide by sqrt of eigenvalue; resulting features have identity covariance.

**Common follow-ups.**
- "Why is SVD numerically preferred?" → Avoids forming `XᵀX`, which doubles the conditioning number.
- "What's the rank of PCA's representation at `k` components?" → `min(k, rank(X))`.

**Common mistakes.**
- Forgetting to center before PCA (or before SVD-for-PCA).
- Confusing left singular vectors (sample basis) with right (feature basis).

**References.**
- [Trefethen & Bau — *Numerical Linear Algebra*, Lecture 4](https://people.maths.ox.ac.uk/trefethen/text.html) — SVD canonical.

---

### Q: What is a Bayesian neural network?

**Category:** concept
**Difficulty:** senior
**Tags:** [bayesian-nn, uncertainty, vi]

**Short answer.** A Bayesian neural network treats weights as random variables with a prior `p(θ)`, and infers a posterior `p(θ | data)`. Predictions integrate over the posterior: `p(y | x) = ∫ p(y | x, θ) p(θ | D) dθ`. Useful for uncertainty quantification. Methods: variational inference (Blundell et al. 2015, "Bayes by Backprop"), Monte Carlo dropout (Gal & Ghahramani 2016), Laplace approximation. Computationally expensive — deep ensembles are often a strong empirical baseline.

**Expansion / why this is the answer.**
- **Why**: standard NNs give point predictions; Bayesian NNs give predictive distributions and *epistemic* uncertainty (how confident is the model in its parameters).
- **Methods**:
  - **Variational inference**: approximate posterior with a tractable family (e.g. Gaussian); minimize ELBO.
  - **MC Dropout**: approximate inference via dropout at test time.
  - **Laplace approximation**: Gaussian centered at MAP estimate; covariance from inverse Hessian.
  - **HMC / SGLD**: MCMC sampling from the posterior; slow but exact.
- **Practical alternatives**:
  - **Deep ensembles**: train N models with different seeds; ensemble. Empirically strong; expensive.
  - **Heteroscedastic regression**: output variance head; predicts aleatoric uncertainty (data noise), not epistemic.
- **When to use**:
  - Safety-critical applications (medical, autonomous driving).
  - Active learning (need calibrated uncertainty to select examples).
  - Bayesian optimization (need posterior over surrogate).

**Common follow-ups.**
- "Aleatoric vs epistemic uncertainty?" → Aleatoric: irreducible data noise. Epistemic: model uncertainty (decreases with more data).
- "Why don't LLMs use Bayesian methods?" → Computationally infeasible at scale; ensembles or temperature-based proxies used instead.

**Common mistakes.**
- Conflating MC Dropout with full Bayesian inference (it's an approximation).

**References.**
- [Blundell et al. — "Bayes by Backprop"](https://arxiv.org/abs/1505.05424).
- [Gal & Ghahramani — "MC Dropout"](https://arxiv.org/abs/1506.02142).

---

### Q: When does data augmentation help, and what are some common pitfalls?

**Category:** concept
**Difficulty:** intro
**Tags:** [data-augmentation, vision, training]

**Short answer.** Augmentation expands the effective training set by applying label-preserving transformations (image flips, color jitter, mixup). Helps especially in small-data and image-domain settings; benefits diminish at very large scale (the data itself is abundant). Pitfalls: augmentations that change the label (e.g. horizontal flip on text or asymmetric objects), augmentation distribution mismatch (overly aggressive cropping reduces signal), and forgetting to disable augmentation at validation/test.

**Expansion / why this is the answer.**
- **Image augmentation**: flip, crop, rotate, color jitter, cutout, mixup, CutMix.
- **Text augmentation**: synonym replacement, back-translation, paraphrasing — generally less effective than image.
- **Mixup** (Zhang et al. 2018): linearly interpolate two images and their labels; regularizer.
- **CutMix** (Yun et al. 2019): replace a region with a patch from another image; mix labels proportionally.
- **AutoAugment / RandAugment**: learned / random augmentation policies.
- **When augmentation hurts**: when the augmentation breaks the label semantics (digits, asymmetric objects), or when very aggressive augmentations dominate the true distribution.

**Common follow-ups.**
- "Is augmentation useful at LLM-pretraining scale?" → Limited; the data itself is huge. Text augmentation is more useful for fine-tuning small models on narrow data.
- "What's test-time augmentation?" → Apply augmentations at inference, average predictions; cheap ensembling for classification.

**Common mistakes.**
- Forgetting to disable augmentation at eval.
- Augmenting in a way that changes the label.

**References.**
- [Zhang et al. — "mixup"](https://arxiv.org/abs/1710.09412).
- [Yun et al. — "CutMix"](https://arxiv.org/abs/1905.04899).

---

### Q: What is contrastive learning, and how does SimCLR work?

**Category:** concept
**Difficulty:** mid
**Tags:** [contrastive-learning, simclr, self-supervised]

**Short answer.** Contrastive learning trains representations by pulling together "positive pairs" and pushing apart "negative pairs." **SimCLR** (Chen et al. 2020): generate two augmented views of the same image (positive pair); contrast against other images in the batch (negative pairs); use the NT-Xent loss (a temperature-scaled softmax over similarities). Strong self-supervised baseline for vision; CLIP extends it to image-text.

**Expansion / why this is the answer.**
- **Goal**: learn an encoder `f(x)` such that semantically similar inputs have similar embeddings.
- **SimCLR**:
  - Take an image `x`; apply two random augmentations to get `x_1, x_2`.
  - Encode → projection head → compute embeddings.
  - Loss: `−log exp(sim(z_1, z_2)/τ) / Σ_k exp(sim(z_1, z_k)/τ)` over all `z_k` in the batch.
  - Negative pairs: all other batch elements.
- **Key ingredients**:
  - Strong augmentation (random crop + color distortion + Gaussian blur).
  - Large batch size (more negatives = better learning).
  - Projection head (linear projection during training; discarded for downstream).
- **MoCo** (He et al. 2019): same idea, but uses a momentum-updated encoder and a memory bank of negatives — works with smaller batches.
- **CLIP** (Radford et al. 2021): contrastive learning across image-text pairs; learned joint embedding space.

**Common follow-ups.**
- "Why is the projection head discarded for downstream?" → Empirically, the pre-projection features are more general.
- "Why is large batch helpful?" → More negatives improve the contrastive signal.

**Common mistakes.**
- Forgetting that augmentations define what's "semantically similar" — augmentation design is crucial.

**References.**
- [Chen et al. — "SimCLR"](https://arxiv.org/abs/2002.05709).
- [He et al. — "MoCo"](https://arxiv.org/abs/1911.05722).
- [Radford et al. — "CLIP"](https://arxiv.org/abs/2103.00020).

---

### Q: What is the receptive field in CNNs, and why does it matter?

**Category:** concept
**Difficulty:** mid
**Tags:** [cnn, receptive-field, architecture]

**Short answer.** A neuron's receptive field is the region of the input image that affects its activation. It grows with depth, kernel size, and stride. For a model to recognize large objects, its deeper layers' receptive field must cover relevant spatial extent. The effective receptive field (Luo et al. 2016) is Gaussian-shaped — neurons are most influenced by the center of their nominal RF, with diminishing influence at the edges.

**Expansion / why this is the answer.**
- **Theoretical RF**: depth-recursive — kernel size + stride compound across layers.
- **Effective RF** (Luo et al. 2016): the actual sensitivity, weighted by gradient flow; much smaller than theoretical, Gaussian-shaped.
- **Stride and pooling** grow RF faster than convolutions alone.
- **Dilated convolutions** (Yu & Koltun 2015): increase RF exponentially without adding parameters.
- **For transformers**: attention's "RF" is the whole context (causal mask aside) — every token can attend to every other. This is part of why transformers beat CNNs on long-context tasks.

**Common follow-ups.**
- "What's the receptive field of attention?" → Full sequence (within the causal mask); contrasts with the limited RF of conv.
- "Why does ResNet have huge theoretical RF but limited effective RF?" → Skip connections route info; gradient magnitude diminishes far from center.

**Common mistakes.**
- Reporting theoretical RF and assuming the neuron "sees" the whole region equally.

**References.**
- [Luo et al. — "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks"](https://arxiv.org/abs/1701.04128).
- [Yu & Koltun — "Multi-Scale Context Aggregation by Dilated Convolutions"](https://arxiv.org/abs/1511.07122).

---
