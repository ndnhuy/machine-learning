from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt

from visualizer.model_visualizer import ModelVisualizer


class InteractiveModelVisualizer(ModelVisualizer):
    """
    Implementation of ModelVisualizer that displays visualizations in an interactive window.
    """

    def __init__(self, x_label: str = "X", y_label: str = "Y"):
        """
        Initialize the interactive model visualizer.

        Parameters:
        -----------
        x_label : str, optional
            The label for the x-axis, defaults to "X"
        y_label : str, optional
            The label for the y-axis, defaults to "Y"
        """
        self.x_label = x_label
        self.y_label = y_label

    def visualize(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Visualize the data points and model predictions in an interactive window.

        Parameters:
        -----------
        x : np.ndarray
            The input features (e.g., house sizes)
        y : np.ndarray
            The actual target values (e.g., house prices)
        y_pred : np.ndarray
            The predicted values from the model
        """
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.get_yaxis().set_major_formatter(
            ticker.StrMethodFormatter('${x:,.0f}'))

        plt.scatter(x, y, label='Data')
        plt.plot(x, y_pred, color='red', label='Fitted Line')
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.legend()
        plt.grid(True)
        plt.title("Linear Regression Visualization (Interactive)")
        plt.show()  # Show interactive window
