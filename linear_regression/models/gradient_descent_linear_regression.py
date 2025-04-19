import numpy as np
from .linear_regression_interface import LinearRegression


class GradientDescentLinearRegression(LinearRegression):
    """
    Linear regression using batch gradient descent.
    """

    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        # Convert to numpy arrays
        x_arr = x
        y_arr = y
        # Feature normalization
        x_mean = np.mean(x_arr)
        x_std = np.std(x_arr) # standard deviation tells you, on average, how far the data points are from the mean
        if x_std != 0:
            x_norm = (x_arr - x_mean) / x_std
        else:
            x_norm = x_arr.copy()
        # Initialize parameters for normalized data
        w_norm = 0.0
        b_norm = 0.0
        n = len(x_norm)
        # Gradient descent on normalized feature
        for _ in range(self.iterations):
            y_pred = w_norm * x_norm + b_norm
            dw_norm = (-2 / n) * np.sum(x_norm * (y_arr - y_pred))
            db_norm = (-2 / n) * np.sum(y_arr - y_pred)
            w_norm -= self.learning_rate * dw_norm
            b_norm -= self.learning_rate * db_norm
        # Map parameters back to original scale
        if x_std != 0:
            w = w_norm / x_std
            b = b_norm - w_norm * x_mean / x_std
        else:
            w = w_norm
            b = b_norm
        return w, b