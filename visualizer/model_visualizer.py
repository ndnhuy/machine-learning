from abc import ABC, abstractmethod
import numpy as np


class ModelVisualizer(ABC):
    """
    Interface for visualizing model predictions.
    """

    @abstractmethod
    def visualize(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Visualize the actual data points and the model predictions.

        Parameters:
        -----------
        x : np.ndarray
            The input features (e.g., house sizes)
        y : np.ndarray
            The actual target values (e.g., house prices)
        y_pred : np.ndarray
            The predicted values from the model
        """
        pass