# ML Algorithm Thread Analysis

Detailed forward/backward thread breakdown for every major ML algorithm family.

## Table of Contents

1. [Supervised — Linear Models](#linear-models)
2. [Supervised — Tree-Based](#tree-based)
3. [Supervised — Distance-Based](#distance-based)
4. [Supervised — Probabilistic](#probabilistic)
5. [Supervised — Margin-Based](#margin-based)
6. [Supervised — Deep Learning](#deep-learning)
7. [Unsupervised — Clustering](#clustering)
8. [Unsupervised — Dimensionality Reduction](#dimensionality-reduction)
9. [Ensemble Methods](#ensemble-methods)

---

## Linear Models

### Linear Regression

```
Forward:  X → Wx + b → ŷ (continuous value)
Backward: (ŷ - y)² → gradient → update W,b   OR   W = (XᵀX)⁻¹Xᵀy (closed-form)
```

- **Shape:** input (n_samples, n_features) → output (n_samples, 1)
- **Operation:** weighted sum of features
- **Info loss:** assumes linear relationship — curves are invisible
- **Learning:** gradient descent OR direct solve (Normal Equation)
- **Swap test:** add polynomial features → captures curves; add regularization → prevents overfitting

### Logistic Regression

```
Forward:  X → Wx + b → sigmoid(z) → probability [0,1]
Backward: cross-entropy loss → gradient → update W,b
```

- **Shape:** input (n_samples, n_features) → output (n_samples, n_classes) probabilities
- **Operation:** linear combination → squash to [0,1] via sigmoid
- **Info loss:** assumes linear decision boundary — complex boundaries invisible
- **Learning:** gradient descent (no closed-form for cross-entropy)
- **Swap test:** swap sigmoid→softmax for multiclass; add L2 penalty → regularized boundary

---

## Tree-Based

### Decision Tree

```
Forward:  X → "feature_3 > 2.5?" → yes → "feature_1 < 0.8?" → ... → leaf → label
Backward: pick feature+threshold that maximizes purity (Gini/Entropy) → split → repeat
```

- **Shape:** input (n_samples, n_features) → output: class label or continuous value
- **Operation:** sequential if/else questions on features
- **Info loss:** each split ignores all features except the chosen one; discretizes continuous features
- **Learning:** greedy recursive splitting — NOT gradient. Measures impurity, picks best cut, repeats
- **Swap test:** Gini→Entropy rarely changes result; max_depth↓ → simpler tree, less overfit

### Random Forest

```
Forward:  X → Tree₁(random subset) → label₁
              Tree₂(random subset) → label₂
              ...
              TreeN(random subset) → labelN  → majority vote → final label
Backward: each tree learns independently on bootstrap sample + random feature subset
```

- **Shape:** same as single tree, but N parallel trees
- **Operation:** ensemble of decorrelated trees → vote
- **Info loss:** each tree sees partial data (bagging) + partial features → diversity through ignorance
- **Learning:** N independent greedy splits (parallelizable)
- **Swap test:** n_estimators↑ → more stable, diminishing returns; max_features↓ → more diversity

---

## Distance-Based

### KNN (K-Nearest Neighbors)

```
Forward:  X_new → compute distance to ALL stored points → sort → pick K nearest → vote
Backward: ❌ NONE — stores raw training data, no parameters to update
```

- **Shape:** input = single vector; compares against all (n_train, n_features) stored vectors
- **Operation:** distance metric (euclidean/manhattan/etc.) + weighted/unweighted vote
- **Info loss:** only K neighbors matter, all other points ignored at prediction time
- **Learning:** NONE — lazy learner. fit() = memorize. Cost is at predict time
- **Swap test:** K↑ → smoother boundary; weights='distance' → closer=stronger vote; metric change → different "closeness" definition

---

## Probabilistic

### Naive Bayes

```
Forward:  X → P(class|features) = P(features|class) × P(class) / P(features) → argmax → label
Backward: count frequencies in training data → compute P(feature|class) for each feature independently
```

- **Shape:** input (n_samples, n_features) → output: class probabilities
- **Operation:** Bayes theorem with naive independence assumption
- **Info loss:** assumes ALL features are independent given class — correlations invisible
- **Learning:** counting + division. One pass through data. No iteration, no gradient
- **Swap test:** Gaussian→Multinomial→Bernoulli changes feature distribution assumption; remove independence assumption → full Bayesian (expensive)

---

## Margin-Based

### SVM (Support Vector Machine)

```
Forward:  X → w·x + b → sign(result) → class (+1 or -1)
          (with kernel: X → φ(X) → w·φ(x) + b → sign → class)
Backward: find w,b that maximize margin between classes (quadratic programming or SGD)
```

- **Shape:** input (n_samples, n_features) → mapped to higher dim with kernel → output: class
- **Operation:** find maximum-margin hyperplane; kernel trick projects to higher dimensions without computing it
- **Info loss:** only support vectors (points near boundary) matter; all other points are irrelevant
- **Learning:** quadratic optimization (LibSVM) or SGD for linear SVM
- **Swap test:** linear→RBF kernel → curved boundary; C↑ → harder margin, more overfit; C↓ → softer, more tolerant

---

## Deep Learning

### Dense (Fully Connected)

```
Forward:  x → Wx + b → activation(z) → output
Backward: loss → ∂L/∂W via chain rule → W -= lr × ∂L/∂W
```

- **Shape:** (batch, features_in) → (batch, features_out)
- **Operation:** matrix multiply + bias + nonlinearity (ReLU/sigmoid/etc.)
- **Info loss:** ReLU kills negatives; dimensionality reduction loses detail
- **Learning:** gradient backpropagation
- **Swap test:** ReLU→GELU subtle difference; units↓ → bottleneck forces compression

### CNN (Convolutional)

```
Forward:  image → slide filter → feature map → pool → compressed feature map
Backward: gradient flows through conv + pool layers → update filter weights
```

- **Shape:** (batch, H, W, C) → conv → (batch, H', W', filters) → pool → (batch, H'', W'', filters)
- **Operation:** convolution (local dot product) + pooling (spatial compression)
- **Info loss:** pooling discards spatial precision; small filter = ignores global context
- **Learning:** gradient backprop; filter weights are shared across spatial positions
- **Swap test:** 3×3→5×5 filter = wider local view; MaxPool→AvgPool = sharp vs smooth

### RNN / LSTM

```
Forward:  x_t + h_{t-1} → h_t → ... → h_T → output
Backward: BPTT (backprop through time) — gradient flows backward through timesteps
```

- **Shape:** (batch, timesteps, features) → (batch, hidden_size) or (batch, timesteps, hidden_size)
- **Operation:** recurrent: current input + previous hidden state → new hidden state
- **Info loss:** vanilla RNN forgets distant past (vanishing gradient); LSTM gates control what to keep/forget
- **Learning:** BPTT — gradient unrolls through time; exploding/vanishing gradient problem
- **Swap test:** RNN→LSTM adds memory gates; →GRU simplifies to 2 gates; →Transformer removes recurrence entirely

### Transformer / Attention

```
Forward:  X → Q,K,V = XW_q, XW_k, XW_v → Attention(Q,K,V) = softmax(QKᵀ/√d)V → FFN → output
Backward: gradient through attention weights + FFN → update W_q, W_k, W_v, W_ffn
```

- **Shape:** (batch, seq_len, d_model) → attention → (batch, seq_len, d_model) → FFN → same
- **Operation:** every token attends to every other token; weighted sum based on relevance
- **Info loss:** softmax concentrates on few tokens; positional encoding is lossy for long sequences
- **Learning:** standard gradient backprop; multi-head = parallel diverse attention patterns
- **Swap test:** heads↑ → more diverse attention; →linear attention = faster but weaker; remove FFN → loses per-token processing

---

## Clustering

### K-Means

```
Forward:  X → compute distance to each centroid → assign to nearest cluster
Backward: recompute centroids as mean of assigned points → repeat until stable
```

- **Shape:** input (n_samples, n_features) → output: cluster label per sample
- **Operation:** distance to centroid + assignment + centroid update
- **Info loss:** forces K spherical clusters — irregular shapes invisible; sensitive to initialization
- **Learning:** iterative EM-style: assign → update → assign → update (no gradient)
- **Swap test:** K↑ → more granular; init='k-means++' → smarter start; →DBSCAN removes need to pick K

### DBSCAN

```
Forward:  X → find neighbors within ε → core/border/noise classification → expand clusters
Backward: ❌ NONE — single pass algorithm, no iteration to convergence
```

- **Shape:** input (n_samples, n_features) → output: cluster label (-1 = noise)
- **Operation:** density-based: if ≥ min_samples within ε radius → core point → expand cluster
- **Info loss:** points below density threshold = noise (discarded)
- **Learning:** no learning — deterministic single/multi pass
- **Swap test:** ε↑ → bigger clusters, less noise; min_samples↑ → stricter core definition

---

## Dimensionality Reduction

### PCA

```
Forward:  X → center data → compute covariance → eigendecomposition → project onto top-k eigenvectors
Backward: ❌ closed-form — solve once, no iteration
```

- **Shape:** (n_samples, n_features) → (n_samples, n_components) where n_components < n_features
- **Operation:** find axes of maximum variance; project data onto them
- **Info loss:** discards low-variance dimensions — assumes variance = importance
- **Learning:** eigendecomposition or SVD (one-shot computation)
- **Swap test:** n_components↑ → keep more info; →t-SNE preserves local structure instead of global variance; →UMAP balances local+global

---

## Ensemble Methods

### Bagging (Bootstrap Aggregating)

```
Forward:  X → Model₁(bootstrap₁) → ŷ₁ ... ModelN(bootstrapN) → ŷN → average/vote → final
Backward: each model trains independently on random subset (with replacement)
```

- **Key idea:** reduce variance through averaging independent models
- **Example:** Random Forest = bagging + random feature subsets

### Boosting (AdaBoost / XGBoost / LightGBM)

```
Forward:  X → Model₁ → residual₁ → Model₂ → residual₂ → ... → sum all models → final
Backward: each new model learns to correct ERRORS of previous models (sequential, not parallel)
```

- **Key idea:** reduce bias by focusing on mistakes
- **XGBoost specific:** uses gradient of loss to fit next tree (gradient boosting = gradient-based backward thread for trees!)
- **Swap test:** learning_rate↓ → slower but more precise correction; n_estimators↑ → more corrections

### Stacking

```
Forward:  X → Model_A → ŷ_A, Model_B → ŷ_B, Model_C → ŷ_C → Meta-model([ŷ_A, ŷ_B, ŷ_C]) → final
Backward: train base models first → use their outputs as features → train meta-model
```

- **Key idea:** let a meta-model learn which base model to trust for which input
- **Info loss:** base model predictions compress all information into single values
