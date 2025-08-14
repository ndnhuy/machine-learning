from matplotlib import animation, ticker
import numpy as np
import matplotlib.pyplot as plt

from visualizer.model_visualizer import ModelVisualizer


class PNGModelVisualizer(ModelVisualizer):
    """
    Implementation of ModelVisualizer that exports visualizations to PNG files.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, output_path: str, x_label: str = "X", y_label: str = "Y"):
        """
        Initialize the PNG model visualizer.

        Parameters:
        -----------
        x : np.ndarray
            The input features (e.g., house sizes)
        y : np.ndarray
            The actual target values (e.g., house prices)
        output_path : str
            The file path where the PNG will be saved
        x_label : str, optional
            The label for the x-axis, defaults to "X"
        y_label : str, optional
            The label for the y-axis, defaults to "Y"
        """
        self._x = x
        self._y = y
        self.output_path = output_path
        self.x_label = x_label
        self.y_label = y_label
        self._y_pred = None

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
        # For backward compatibility, update x and y if called
        self._x = x
        self._y = y
        self.accept(y_pred)
        self.show()

    def accept(self, y_hat: np.ndarray) -> None:
        """
        Accepts the predicted values only, as per interface.

        Parameters:
        -----------
        y_hat : np.ndarray
            The predicted values from the model
        """
        self._y_pred = y_hat

    def show(self) -> None:
        """
        Displays (saves) the visualizations to PNG file.
        """
        if self._x is None or self._y is None or self._y_pred is None:
            raise ValueError("Data and predictions must be provided before calling show().")
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.get_yaxis().set_major_formatter(
            ticker.StrMethodFormatter('${x:,.0f}'))
        plt.scatter(self._x, self._y, label='Data')
        plt.plot(self._x, self._y_pred, color='red', label='Fitted Line')
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_path)
        plt.close()  # Close the figure to free memory
