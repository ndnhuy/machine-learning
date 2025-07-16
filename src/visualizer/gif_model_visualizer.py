from matplotlib import animation, ticker
import numpy as np
import matplotlib.pyplot as plt

from visualizer.model_visualizer import ModelVisualizer


class GifModelVisualizer(ModelVisualizer):
    """
    Implementation of ModelVisualizer that exports visualizations to PNG files.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, output_path: str, x_label: str = "X", y_label: str = "Y"):
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
        self.x = x
        self.y = y
        self.y_pred_history = []

    def saveToFile(self):
        fig, ax = plt.subplots()
        ax.scatter(self.x, self.y, label='Data')
        line, = ax.plot([], [], 'r-', label='Fitted Line')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_title("Gradient Descent Progress")
        ax.grid(True)
        ax.legend()

        # Set axes limits for clarity
        xmin, xmax = self.x.min() - 1, self.x.max() + 1
        ymin, ymax = self.y.min() - 5, self.y.max() + 5
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        def update(frame):
            line.set_data(self.x, self.y_pred_history[frame])
            return line,

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.y_pred_history),
            interval=200,    # Time in ms between frames
            blit=True
        )
        ani.save(self.output_path, writer="pillow")

    def visualize(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        # Append y_pred to the history array
        self.y_pred_history.append(y_pred.copy())
