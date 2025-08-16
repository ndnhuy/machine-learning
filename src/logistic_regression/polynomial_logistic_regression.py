import copy
import math
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from logistic_regression.compute_functions import compute_cost_logistic, compute_gradient_logistic


class PolynomialLogisticRegression:
    """
    Logistic Regression with polynomial features for non-linear decision boundaries.
    This extends your existing gradient descent implementation.
    """

    def __init__(self, learning_rate: float = 0.01, iterations: int = 10000, degree: int = 2):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        self.iterationConsumer = lambda w, b: None
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Fit a polynomial logistic regression model on X and y.

        Args:
            X (np.ndarray): Input features, shape (m, n)
            y (np.ndarray): Target values, shape (m,)
        Returns:
            tuple[np.ndarray, float]: Model weights and bias
        """
        # Transform features to polynomial
        X_poly = self.poly_features.fit_transform(X)
        
        # Run gradient descent on polynomial features
        self.w, self.b, _ = self.gradient_descent(
            X_poly, y, np.zeros(X_poly.shape[1]), 0.0, 
            self.learning_rate, self.iterations
        )
        return self.w, self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        if self.w is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X_poly = self.poly_features.transform(X)
        z = np.dot(X_poly, self.w) + self.b
        return (self._sigmoid(z) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X."""
        if self.w is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X_poly = self.poly_features.transform(X)
        z = np.dot(X_poly, self.w) + self.b
        return self._sigmoid(z)

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        z = np.clip(z, -250, 250)  # Prevent overflow
        return 1 / (1 + np.exp(-z))

    def setIterationConsumer(self, consumer_fn):
        """Set function to consume (w, b) after each iteration."""
        def poly_consumer(w, b):
            # For visualization, we need to convert back to original feature space
            # This is complex for polynomial features, so we'll pass the polynomial weights
            consumer_fn(w, b)
        self.iterationConsumer = poly_consumer

    def gradient_descent(self, X, y, w_in, b_in, alpha, num_iters):
        """
        Performs batch gradient descent on polynomial features.
        Same as your original implementation but works on transformed features.
        """
        J_history = []
        w = copy.deepcopy(w_in)
        b = b_in

        for i in range(num_iters):
            # Calculate the gradient and update the parameters
            dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

            # Update Parameters
            w = w - alpha * dj_dw
            b = b - alpha * dj_db

            self.iterationConsumer(w, b)

            # Save cost
            if i < 100000:
                J_history.append(compute_cost_logistic(X, y, w, b))

            # Print cost every interval
            if i % math.ceil(num_iters / 10) == 0:
                print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

        return w, b, J_history

    def get_feature_names(self, input_features=None):
        """Get names of the polynomial features."""
        if input_features is None:
            input_features = [f'x{i}' for i in range(2)]  # Default for 2D
        return self.poly_features.get_feature_names_out(input_features)


# Utility function to visualize polynomial decision boundary
def visualize_polynomial_boundary(model, X, y, title="Polynomial Decision Boundary"):
    """
    Visualize the decision boundary for a polynomial logistic regression model.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    
    # Create a mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and regions
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap=plt.cm.RdYlBu)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
    
    # Plot data points
    colors = ['#0099FF', '#FF2222']
    markers = ['o', 'x']
    labels = ['y=0', 'y=1']
    
    for i, (color, marker, label) in enumerate(zip(colors, markers, labels)):
        mask = (y == i)
        plt.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker, 
                   s=100, linewidths=2, label=label, edgecolors='black')
    
    plt.colorbar(label='Predicted Probability')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
