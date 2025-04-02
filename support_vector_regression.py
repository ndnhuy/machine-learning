import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.svm import SVR

# === 1. Sample Data ===
x, y = make_regression(n_samples=1000, n_features=1, noise=100, random_state=42)
x = x.flatten()  # Flatten X to 1D for consistency
y = y / 10       # Scale y to match the original price range

# === 2. Train Support Vector Regression model ===
# Using SVR with an RBF kernel which is effective for non-linear relationships.
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
# Reshape x to (-1, 1) because the regressor expects a 2D array as features.
svr_model.fit(x.reshape(-1, 1), y)
print("Trained Support Vector Regressor")

# === 3. Make predictions ===
# Sort x and corresponding predictions to plot a smooth curve.
sorted_idx = np.argsort(x)
x_sorted = x[sorted_idx]
y_pred_sorted = svr_model.predict(x_sorted.reshape(-1, 1))

# === 4. Plot result ===
plt.scatter(x, y, label='Data', alpha=0.5)
plt.plot(x_sorted, y_pred_sorted, color='red', label='SVR Prediction', linewidth=2)
plt.xlabel("Size (m²)")
plt.ylabel("Price ($1000s)")
plt.title("Support Vector Regression")
plt.legend()
plt.grid(True)
plt.savefig("svr_fitted_line.png")