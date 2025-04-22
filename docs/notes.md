# Summary of Key Concepts

## Quadratic Functions & Derivatives
- A quadratic function has the form `f(x) = ax² + bx + c` (`a ≠ 0`). Its graph is a parabola.
- The **derivative** `f′(x)` measures the instantaneous rate of change (the slope of the tangent).
  - **Definition**:  
    `f′(x) = limₕ→0 [f(x + h) – f(x)] / h`
  - **Example**: For `f(x) = x²`,  
    `f′(x) = limₕ→0 [(x + h)² – x²] / h = 2x`
  - At `x = 3`, the slope is `f′(3) = 6`, meaning `y` rises 6 units for a 1‑unit move in `x`.

## Smoothness & Why Squared Errors Are Easy to Optimize
- A **smooth** function is continuous and differentiable everywhere—no “corners.”
- The sum of squared residuals `∑(y – y_pred)²` is a quadratic in the model parameters (`w`, `b`), hence smooth.
- Smooth, differentiable functions let us set derivatives to zero and solve for the global optimum in closed form.

## Convex vs. Concave Functions
- A **convex function** curves upward (“smile”), has a single global minimum, and no local minima.
- A **concave function** curves downward (“frown”) and has a single global maximum (e.g., `f(x) = –x²` is concave, not convex).
- Convexity guarantees that gradient-based methods (like gradient descent) won’t get stuck in local minima.

## Why Sum of Squared Residuals (OLS)
- Squaring makes all errors positive—avoids cancellation of positive/negative residuals.
- Penalizes larger errors more than smaller ones in a balanced way.
- Leads to a convex, differentiable loss that’s easy to minimize and corresponds to the MLE under normally distributed errors.

## Why Not Higher Powers (e.g., ^4 or ^99)?
- Higher powers over-penalize outliers, making the model overly sensitive to noise.
- Numerical stability suffers (huge gradient values, possible overflow).
- Only the squared-error loss matches the MLE for Gaussian noise.

## When Errors Aren’t Normally Distributed
- If residuals are heavy-tailed or contain outliers, OLS estimates can be biased.
- An alternative is **Least Absolute Deviations (LAD)**, which minimizes `∑|y – y_pred|` and is more robust to outliers.
- **Example**: Data with one extreme outlier will pull an OLS fit toward it, while LAD will largely ignore that single point.

## Choosing Between OLS and LAD
- **Check residual distribution and presence of outliers.**
- Compare performance with cross-validation:
  - OLS often wins on **Mean Squared Error (MSE)** when noise is Gaussian.
  - LAD often wins on **Mean Absolute Error (MAE)** in the presence of outliers or non-Gaussian noise.

---

This gives you the theoretical foundation for why linear regression uses the sum of squared errors, what convexity and smoothness mean for optimization, how derivatives work, and when you might choose alternatives like LAD.