import matplotlib
from visualizer.model_visualizer import ModelVisualizer
from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # or 'Qt5Agg', forces a specific backend


class InteractiveModelVisualizer(ModelVisualizer):
    """
    Implementation of ModelVisualizer that displays visualizations in an interactive window.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, x_label: str = "X", y_label: str = "Y"):
        """
        Initialize the interactive model visualizer.

        Parameters:
        -----------
        x : np.ndarray
            The input features (e.g., house sizes)
        y : np.ndarray
            The actual target values (e.g., house prices)
        x_label : str, optional
            The label for the x-axis, defaults to "X"
        y_label : str, optional
            The label for the y-axis, defaults to "Y"
        """
        self._x = x
        self._y = y
        self.x_label = x_label
        self.y_label = y_label
        self._y_pred = None

    def visualize(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        # For backward compatibility, update x and y if called
        self._x = x
        self._y = y
        self.accept(y_pred)
        self.show()

    def accept(self, y_hat: np.ndarray) -> None:
        self._y_pred = y_hat

    def show(self) -> None:
        if self._x is None or self._y is None or self._y_pred is None:
            raise ValueError(
                "Data and predictions must be provided before calling show().")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.get_yaxis().set_major_formatter(
            ticker.StrMethodFormatter('${x:,.0f}'))
        ax.scatter(self._x, self._y, label='Data')
        ax.plot(self._x, self._y_pred, color='red', label='Fitted Line')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.legend()
        ax.grid(True)
        ax.set_title("Linear Regression Visualization (Interactive)")
        plt.show()  # Show interactive window
