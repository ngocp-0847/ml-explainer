# Audience-Level Explanation Templates

## 👶 High School Student

**Frame:** The machine looks at data, finds patterns, then guesses answers.

**Template:**
- Imagine a row of numbers (like a barcode)
- The algorithm processes that row in its own way (bending / comparing / splitting / counting)
- At the end, it picks an answer
- If wrong, it adjusts (or not — some algorithms don't learn at all!)

**Vocabulary:** numbers, list, bending, comparing, closest, splitting, pattern, answer, adjusting

**Avoid:** matrix, gradient, loss function, manifold, representation, hyperplane

**Analogies by algorithm:**
- DL: "bẻ cong dãy số nhiều lần rồi chọn đáp án"
- KNN: "hỏi 3 người đứng gần nhất, theo số đông"
- Decision Tree: "chơi 20 câu hỏi — hỏi yes/no liên tục"
- Naive Bayes: "đếm xem thường gặp nhất là gì rồi đoán"
- K-Means: "chia bạn bè thành nhóm theo chỗ đứng"
- SVM: "kẻ 1 đường chia 2 phe, cách xa nhất có thể"

---

## 👨‍💻 Developer

**Frame:** An algorithm is a function: `prediction = f(data, parameters)`

**Template:**
- Input: array/matrix of shape `(n_samples, n_features)`
- Operation: [specific to algorithm — matmul / distance / split / count]
- Output: prediction (class label, probability, continuous value, or cluster)
- Learning: how parameters update (gradient, split criterion, frequency count, or nothing)
- Complexity: time/space for train and predict

**Vocabulary:** array, shape, function, parameter, fit, predict, complexity

**Code-first:** always show the sklearn/torch equivalent and explain params.

---

## 👨‍🔬 ML Engineer

**Frame:** We're shaping decision boundaries in feature space via optimization.

**Template:**
- Input space: data manifold in R^n
- Hypothesis class: what boundaries can this model represent?
- Optimization: how does it search the hypothesis space?
- Inductive bias: what assumptions does this algorithm make?
- Generalization: bias-variance tradeoff and regularization

**Vocabulary:** hypothesis space, inductive bias, generalization bound, VC dimension, bias-variance, regularization

**Depth markers:**
- Discuss computational complexity (train vs predict)
- Compare sample efficiency across algorithms
- Mention when the algorithm breaks down (edge cases)
- Discuss relationship to other algorithms (e.g., Logistic Reg = 1-layer neural net)

---

## Switching Between Levels

| Signal | Level |
|---|---|
| "What is AI / ML?" | High school |
| "How do I implement this?" / "What params should I tune?" | Developer |
| "What's the inductive bias?" / "Bias-variance tradeoff?" | ML Engineer |

When unsure, start at developer level — safest middle ground.
