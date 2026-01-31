# Audience-Level Explanation Templates

## 👶 High School Student

**Frame:** The machine holds a list of numbers, bends it many times, then picks an answer.

**Template:**
- Imagine a row of numbers (like a barcode)
- Each layer bends/stretches that row in a specific way
- At the end, the most "stretched toward" answer wins
- If wrong, it traces back and adjusts the bending slightly

**Vocabulary:** numbers, list, bending, stretching, answer, adjusting

**Avoid:** matrix, gradient, loss function, manifold, representation

---

## 👨‍💻 Developer

**Frame:** A layer is a pure function: `y = f(Wx + b)`

**Template:**
- Input: tensor of shape `(batch, features)`
- Operation: matrix multiply + bias + nonlinearity
- Output: tensor of shape `(batch, new_features)`
- Backprop: chain rule computes `dL/dW` for each parameter
- Update: `W -= lr * dL/dW`

**Vocabulary:** tensor, shape, function, parameter, learning rate, chain rule

**Analogy:** Each layer is a `.map()` over the feature dimension with learnable coefficients.

---

## 👨‍🔬 ML Engineer

**Frame:** We're shaping representation manifolds via gradient flow.

**Template:**
- Input manifold: data lies on a submanifold of R^n
- Layer: differentiable map between manifolds, parameterized by W
- Forward: compose maps to project onto decision boundary
- Backward: gradient flow through the computational graph via adjoint method
- Optimization: navigate the loss landscape curvature (Adam, SGD + momentum)

**Vocabulary:** manifold, representation space, computational graph, adjoint, curvature, Hessian

**Depth markers:**
- Discuss vanishing/exploding gradients and skip connections
- Mention spectral properties of weight matrices
- Reference information bottleneck theory when relevant
- Discuss implicit regularization from optimizer choice

---

## Switching Between Levels

Read the audience first. Key signals:

| Signal | Level |
|--------|-------|
| "What is AI?" | High school |
| "How do I implement this?" | Developer |
| "What's the inductive bias?" | ML Engineer |

When unsure, start at developer level — it's the safest middle ground.
Adjust up or down based on follow-up questions.
