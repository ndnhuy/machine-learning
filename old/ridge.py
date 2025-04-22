import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

# === 1. Sample Data ===
x, y = make_regression(n_samples=1000, n_features=1, noise=100, random_state=42)
x = x.flatten()    # Flatten X to 1D for consistency
y = y / 10         # Scale y to match the original price range

# === 2. Train Ridge Regression model ===
# Ridge applies L2 regularization to reduce large coefficient values.
ridge_model = Ridge(alpha=1.0)  # alpha is the regularization strength
ridge_model.fit(x.reshape(-1, 1), y)
print("Trained Ridge Regression model")

# === 3. Make predictions ===
# Sort x and corresponding predictions to plot a smooth line.
sorted_idx = np.argsort(x)
x_sorted = x[sorted_idx]
y_pred_sorted = ridge_model.predict(x_sorted.reshape(-1, 1))

# === 4. Plot result ===
plt.scatter(x, y, label='Data', alpha=0.5)
plt.plot(x_sorted, y_pred_sorted, color='red', label='Ridge Prediction', linewidth=2)
plt.xlabel("Size (m²)")
plt.ylabel("Price ($1000s)")
plt.title("Ridge Regression")
plt.legend()
plt.grid(True)
plt.savefig("ridge_fitted_line.png")
plt.show()