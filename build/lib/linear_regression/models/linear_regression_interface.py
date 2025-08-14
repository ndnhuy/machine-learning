from abc import ABC, abstractmethod
import numpy as np

class LinearRegression(ABC):
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """
        Fit a linear regression model on x and y and return (slope, intercept).
        """
        pass