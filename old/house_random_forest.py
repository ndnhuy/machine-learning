import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

# === 1. Sample Data ===
x, y = make_regression(n_samples=1000, n_features=1, noise=100, random_state=42)
x = x.flatten()  # Flatten X to 1D for consistency
y = y / 10       # Scale y to match the original price range

# === 2. Train Random Forest Regressor ===
# Random forests are robust to noise and can capture non-linear relationships.
model = RandomForestRegressor(n_estimators=100, random_state=42)
# Reshape x to (-1, 1) since the regressor expects a 2D array for features.
model.fit(x.reshape(-1, 1), y)
print("Trained Random Forest Regressor")

# === 3. Make predictions ===
# Sort x and corresponding predictions to plot a smooth line.
sorted_idx = np.argsort(x)
x_sorted = x[sorted_idx]
y_pred_sorted = model.predict(x_sorted.reshape(-1, 1))

# === 4. Plot result ===
plt.scatter(x, y, label='Data', alpha=0.5)
plt.plot(x_sorted, y_pred_sorted, color='red', label='Random Forest Prediction', linewidth=2)
plt.xlabel("Size (m²)")
plt.ylabel("Price ($1000s)")
plt.title("Random Forest Regression")
plt.legend()
plt.grid(True)
plt.savefig("random_forest_fitted_line.png")