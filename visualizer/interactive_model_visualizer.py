import numpy as np
import matplotlib.pyplot as plt
from .model_visualizer import ModelVisualizer


class InteractiveModelVisualizer(ModelVisualizer):
    """
    Implementation of ModelVisualizer that displays visualizations in an interactive matplotlib window.
    """

    def __init__(self, x_label: str = "X", y_label: str = "Y", title: str = "Model Visualization"):
        """
        Initialize the interactive model visualizer.

        Parameters:
        -----------
        x_label : str, optional
            The label for the x-axis, defaults to "X"
        y_label : str, optional
            The label for the y-axis, defaults to "Y"
        title : str, optional
            The title for the plot, defaults to "Model Visualization"
        """
        self.x_label = x_label
        self.y_label = y_label
        self.title = title

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
        plt.scatter(x, y, label='Data')
        plt.plot(x, y_pred, color='red', label='Fitted Line')
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.legend()
        plt.grid(True)
        plt.show()  # Display the plot in an interactive window