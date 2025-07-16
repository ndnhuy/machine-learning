from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class XYGraph:
    def __init__(self, x_label: str, y_label: str, width=10, height=6):
        self.x_label = x_label
        self.y_label = y_label
        self.plot_elements = []
        self.width = width
        self.height = height

        self.figure: Figure = plt.figure(figsize=(self.width, self.height))
        plt.xlabel(self.x_label)
        self.ax: Axes = self.figure.gca()
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.legend()
        self.ax.grid(True)
