from abc import ABC, abstractmethod


class PlotElement(ABC):
    @abstractmethod
    def draw(self, xy_graph):
        """
        Render the plot element to the given XYGraph.

        Args:
          xy_graph: An instance of XYGraph to draw on.
        """
        pass
