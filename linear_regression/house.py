import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from visualizer.png_model_visualizer import PNGModelVisualizer


def compute_linear_regression_params(x, y):
    # === 2. Compute means ===
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # === 3. Compute w (slope) and b (intercept) ===
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    w = numerator / denominator
    b = y_mean - w * x_mean

    print(f"Learned model (via Gradient Descent): y = {w:.2f}x + {b:.2f}")

    return w, b


def compute_by_gradient_descent(x, y):
    # === 2. Initialize parameters for gradient descent ===
    w = 0.0
    b = 0.0
    learning_rate = 0.01
    iterations = 1000

    # === 3. Gradient Descent Loop ===
    for _ in range(iterations):
        # Predictions
        y_pred = w * x + b

        # Compute gradients (partial derivatives of MSE)
        dw = (-2 / len(x)) * np.sum(x * (y - y_pred))
        db = (-2 / len(x)) * np.sum(y - y_pred)

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

    print(f"Learned model (via Gradient Descent): y = {w:.2f}x + {b:.2f}")

    return w, b

# This code generates synthetic data representing house sizes and prices.
# It then fits a linear regression model using gradient descent to determine:
#   - The slope (w)
#   - The intercept (b)
# Finally, it makes predictions using the learned model and visualizes the results:
#   - Plots the actual data points.
#   - Plots the fitted regression line.


# === 1. Sample Data ===
# Generate data with 1 feature, 100 samples, and some noise
x, y = make_regression(n_samples=10, n_features=1, noise=100, random_state=42)
x = x.flatten()  # Flatten X to 1D for consistency
y = y / 10  # Scale y to match the original price range

# print x and y
print(f"x: {x}")
print(f"y: {y}")

# w, b = compute_linear_regression_params(x, y)
w, b = compute_by_gradient_descent(x, y)

# === 4. Make predictions ===
y_pred = w * x + b

# === 5. Plot result using the new visualizer ===
visualizer = PNGModelVisualizer(
    output_path="fitted_line.png",
    x_label="Size (m²)",
    y_label="Price ($1000s)"
)
visualizer.visualize(x, y, y_pred)
