from graph.plot_element import PlotElement


class Scatter(PlotElement):
    """
    A class representing a line plot element.

    Attributes:
        x (list): The x-coordinates of the line.
        y (list): The y-coordinates of the line.
        color (str): The color of the line.
        label (str): The label for the line.
    """

    def __init__(self, x, y, color='blue', label=None):
        """
        Initialize the Line object.

        Args:
            x (list): The x-coordinates of the line.
            y (list): The y-coordinates of the line.
            color (str): The color of the line. Default is 'blue'.
            label (str): The label for the line. Default is None.
        """
        self.x = x
        self.y = y

    def draw(self, xy_graph):
        """
        scatter the data points to the given XYGraph.

        Args:
            xy_graph: An instance of XYGraph to draw on.
        """
