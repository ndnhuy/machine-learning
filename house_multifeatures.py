import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

def compute_by_gradient_descent(X, y):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)  # one weight per feature
    b = 0.0
    learning_rate = 0.01
    iterations = 1000

    for _ in range(iterations):
        y_pred = np.dot(X, weights) + b
        error = y - y_pred
        dw = (-2 / n_samples) * np.dot(X.T, error)
        db = (-2 / n_samples) * np.sum(error)
        weights -= learning_rate * dw
        b -= learning_rate * db

    print(f"Learned model: y = {weights[0]:.2f}*size + {weights[1]:.2f}*bedrooms + {b:.2f}")
    return weights, b

# Generate data with 2 features (e.g., house size and number of bedrooms)
X, y = make_regression(n_samples=1000, n_features=2, noise=100, random_state=42)
y = y / 10  # Scale the prices

weights, b = compute_by_gradient_descent(X, y)
y_pred = np.dot(X, weights) + b

# 3D Scatter plot and regression plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, label='Data', alpha=0.6)

# Create a grid to plot the regression plane
x0_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
x1_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
X0, X1 = np.meshgrid(x0_range, x1_range)
plane_y = weights[0] * X0 + weights[1] * X1 + b

ax.plot_surface(X0, X1, plane_y, color='red', alpha=0.5)
ax.set_xlabel("Size (m²)")
ax.set_ylabel("Number of Bedrooms")
ax.set_zlabel("Price ($1000s)")
plt.legend()
plt.savefig("fitted_line_3d.png")