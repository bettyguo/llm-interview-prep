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
