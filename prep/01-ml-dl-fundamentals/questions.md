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
