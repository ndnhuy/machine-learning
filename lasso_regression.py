import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso

# === 1. Sample Data ===
x, y = make_regression(n_samples=1000, n_features=1, noise=100, random_state=42)
x = x.flatten()    # Flatten X to 1D for consistency
y = y / 10         # Scale y to match the original price range

# === 2. Train Lasso Regression model ===
# Lasso applies L1 regularization which can drive some coefficients to zero.
lasso_model = Lasso(alpha=0.1)  # alpha is the regularization strength
lasso_model.fit(x.reshape(-1, 1), y)
print("Trained Lasso Regression model")

# === 3. Make predictions ===
# Sort x and corresponding predictions to plot a smooth line.
sorted_idx = np.argsort(x)
x_sorted = x[sorted_idx]
y_pred_sorted = lasso_model.predict(x_sorted.reshape(-1, 1))

# === 4. Plot result ===
plt.scatter(x, y, label='Data', alpha=0.5)
plt.plot(x_sorted, y_pred_sorted, color='red', label='Lasso Prediction', linewidth=2)
plt.xlabel("Size (m²)")
plt.ylabel("Price ($1000s)")
plt.title("Lasso Regression")
plt.legend()
plt.grid(True)
plt.savefig("lasso_fitted_line.png")
plt.show()