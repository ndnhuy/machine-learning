import numpy as np
from .linear_regression_interface import LinearRegression


class GradientDescentLinearRegression(LinearRegression):
    """
    Linear regression using batch gradient descent.
    """

    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000, normalize_features: bool = True):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.normalize_features = normalize_features
        self.iterationConsumer = lambda w, b: None

    def setIterationConsumer(self, consumer_fn):
        """
        Set the function to consume (w, b) after each iteration.
        """
        self.iterationConsumer = consumer_fn

    def fit(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        # Convert to numpy arrays
        x_arr = x
        y_arr = y

        # Feature normalization
        if self.normalize_features:
            x_mean = np.mean(x_arr)
            # standard deviation tells you, on average, how far the data points are from the mean
            x_std = np.std(x_arr)
            if x_std != 0:
                x_norm = (x_arr - x_mean) / x_std
            else:
                x_norm = x_arr.copy()

            w_norm, b_norm = self.doGradientDescent(x_norm, y_arr)

            # Rescale the parameters back to the original scale
            w = w_norm / x_std
            b = b_norm - w_norm * x_mean / x_std
            return w, b
        else:
            # Skip normalization if the flag is False
            return self.doGradientDescent(x_arr, y_arr)

    def doGradientDescent(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        w = 0.0
        b = 0.0
        n = len(x)
        for _ in range(self.iterations):
            y_pred = w * x + b
            dw = (-2 / n) * np.sum(x * (y - y_pred))
            db = (-2 / n) * np.sum(y - y_pred)
            w -= self.learning_rate * dw
            b -= self.learning_rate * db
            self.iterationConsumer(w, b)
        return w, b
