from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt

from visualizer.model_visualizer import ModelVisualizer


class PNGModelVisualizer(ModelVisualizer):
    """
    Implementation of ModelVisualizer that exports visualizations to PNG files.
    """

    def __init__(self, output_path: str, x_label: str = "X", y_label: str = "Y"):
        """
        Initialize the PNG model visualizer.

        Parameters:
        -----------
        output_path : str
            The file path where the PNG will be saved
        x_label : str, optional
            The label for the x-axis, defaults to "X"
        y_label : str, optional
            The label for the y-axis, defaults to "Y"
        """
        self.output_path = output_path
        self.x_label = x_label
        self.y_label = y_label

    def visualize(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Visualize the data points and model predictions, and save to PNG file.

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
        plt.savefig(self.output_path)
        plt.close()  # Close the figure to free memory
