---
name: thread-vector-thinking
description: >
  A mental framework for understanding, explaining, and reasoning about any ML/DL algorithm
  by decomposing it into two core threads: forward (data→prediction) and backward (learning/correction).
  Covers ALL ML families: deep learning (Dense, CNN, RNN, Transformer, Attention), classical ML
  (KNN, SVM, Decision Tree, Random Forest, Naive Bayes, Linear/Logistic Regression),
  unsupervised (K-Means, PCA, DBSCAN), and ensemble methods (Bagging, Boosting, Stacking).
  Use when asked to explain any ML algorithm, layer, or architecture — or when answering
  ML/DL interview questions, teaching, debugging model behavior, or building intuition.
  Triggers: ML/DL explanation, algorithm analysis, interview prep, "how does X work",
  model walkthrough, gradient flow, teaching/tutoring.
---

# Thread–Vector Thinking

Every ML algorithm does exactly 2 things: push data toward a prediction (forward thread),
then improve itself somehow (backward thread). Master these 2 threads for ANY algorithm.

## Core Principle

> ML is not "AI thinking" — it's "data being transformed under control."

## The 5 Root Questions

Reduce ANY ML question to one of these:

1. **Shape** — What does the data look like here? dimensions? type? meaning?
2. **Operation** — What math/logic does this step apply? multiply? split? count? distance?
3. **Info loss** — Where is information lost, compressed, or ignored?
4. **Learning** — How does it improve? gradient? splitting? counting? nothing?
5. **Swap test** — If you replace this component, what changes?

Always force questions into this frame before answering.

## Two Threads (Generalized)

### Thread 1: Forward (data → prediction)

Every algorithm has this. The shape varies:

```
DL:          Input → transform → transform → ... → probability
Tree:        Input → question → question → ... → leaf label
KNN:         Input → measure distances → vote → label
Regression:  Input → multiply weights → sum → output
Clustering:  Input → measure distances → assign cluster
```

Mantra: *"Trace the data from input to output — what happens at each step?"*

### Thread 2: Backward (learning/correction)

NOT every algorithm has the same backward thread. This is the key insight:

| Type | Backward Thread | Examples |
|---|---|---|
| **Gradient-based** | Error → gradient → weight update | DL, Linear/Logistic Reg, SVM (SGD) |
| **Split-based** | Impurity → find best split → grow tree | Decision Tree, Random Forest |
| **Count-based** | Count frequencies → compute probabilities | Naive Bayes |
| **Iterative** | Reassign → recompute centers → repeat | K-Means, EM |
| **Closed-form** | Solve equation directly (no iteration) | OLS Regression, PCA |
| **Lazy (none)** | No learning — memorize everything | KNN |

Mantra: *"What does this algorithm adjust, and how?"*

## 5-Line Explanation Template

For ANY algorithm or layer:

1. Input data shape?
2. Core operation (math/logic)?
3. How does output differ from input?
4. What information is discarded?
5. How does it learn (or not)?

## Algorithm Thread Map

Quick reference — see [references/ml-algorithms.md](references/ml-algorithms.md) for
detailed thread analysis of each algorithm family.

| Algorithm | Forward Thread | Backward Thread | Key Operation |
|---|---|---|---|
| Dense/MLP | matmul + activation | gradient backprop | vector mixing |
| CNN | convolution + pool | gradient backprop | local pattern detection |
| RNN/LSTM | sequential hidden state | BPTT | sequence memory |
| Transformer | attention + FFN | gradient backprop | global context mixing |
| KNN | distance + vote | ❌ none (lazy) | similarity comparison |
| Decision Tree | question chain | impurity splitting | feature thresholding |
| Random Forest | many trees → vote | independent splits | ensemble diversity |
| SVM | find margin boundary | quadratic opt / SGD | maximum margin |
| Naive Bayes | probability lookup | frequency counting | conditional independence |
| Linear Reg | Wx + b | gradient / closed-form | line fitting |
| Logistic Reg | Wx + b → sigmoid | gradient descent | probability boundary |
| K-Means | nearest centroid | centroid recompute | cluster assignment |
| PCA | project onto axes | eigendecomposition | variance maximization |
| XGBoost | sequential trees | residual fitting + gradient | boosted correction |

## Audience Adaptation

See [references/audience-levels.md](references/audience-levels.md) for tailored explanation
templates at three levels: beginner, developer, ML engineer.

## 30-Second Rapid Framework

When caught off guard, answer these 5 in order:

1. What data goes in? (shape, type)
2. What operation does the algorithm do?
3. What comes out? (how is it different?)
4. How does it learn? (gradient? split? count? nothing?)
5. Why does this algorithm exist? (what's it good at?)

Answering all 5 = complete answer.

## Mental Checklist

Before answering any ML question:

- [ ] Am I talking about concrete data or vague concepts?
- [ ] Did I mention data shape?
- [ ] Did I mention the core operation?
- [ ] Did I mention information loss?
- [ ] Did I explain the learning mechanism (or lack thereof)?

If any unchecked → answer is not deep enough.

## Master Mode

When fluent, the progression becomes automatic:

```
See code → see data flow → see operations → see learning mechanism → see behavior
```

This is the boundary between "using ML" and "understanding ML."
