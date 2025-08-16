from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt

from visualizer.model_visualizer import ModelVisualizer



class InteractiveGifModelVisualizer(ModelVisualizer):
    """
    Implementation of ModelVisualizer that displays gradient descent progress as an interactive animation.
    Now supports a customizable scatter strategy for labeling points.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, x_label: str = "X", y_label: str = "Y", scatter_strategy=None):
        """
        Initialize the interactive GIF model visualizer.

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
        scatter_strategy : callable, optional
            A function(ax, x, y) that plots the data points on the axes. If None, uses the default scatter strategy.
        """
        self.x = x
        self.y = y
        self.x_label = x_label
        self.y_label = y_label
        self.x_pred_history = []
        self.y_pred_history = []
        if scatter_strategy is not None:
            self.scatter_strategy = scatter_strategy
        else:
            self.scatter_strategy = self.default_scatter_strategy

    @staticmethod
    def default_scatter_strategy(ax, x, y):
        """
        Default scatter strategy: simple scatter plot with one color/marker.
        """
        ax.scatter(x, y, label='Data')

    def visualize(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        # For backward compatibility, update x and y if called
        self.x = x
        self.y = y
        self.accept(y_pred)

    def accept(self, y_hat: np.ndarray) -> None:
        self.x_pred_history.append(self.x.copy())
        # Append y_hat to the history array
        self.y_pred_history.append(y_hat.copy())

    def accept_x_y(self, x_pred: np.ndarray, y_pred: np.ndarray) -> None:
        self.x_pred_history.append(x_pred.copy())
        self.y_pred_history.append(y_pred.copy())

    def show(self):
        fig, ax = plt.subplots()
        # Use the scatter strategy to plot the data points
        self.scatter_strategy(ax, self.x, self.y)
        line, = ax.plot([], [], 'r-', label='Fitted Line')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_title("Gradient Descent Progress (Interactive)")
        ax.grid(True)
        ax.legend()

        def update(frame):
            line.set_data(
                self.x_pred_history[frame], self.y_pred_history[frame])
            return line,

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.y_pred_history),
            interval=200,    # Time in ms between frames
            blit=True
        )
        plt.show()  # Show the animation interactively
        # Close the figure after showing to prevent accumulation
        plt.close(fig)
