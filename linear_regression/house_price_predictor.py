import numpy as np
from linear_regression_interface import LinearRegression
from closed_form_linear_regression import ClosedFormLinearRegression


class HousePricePredictor:
    """
    A class to predict house prices based on house size.
    """

    def __init__(self, sizes, prices, model: LinearRegression = None):
        """
        Initialize the predictor with training data and regression model.

        Parameters:
        -----------
        sizes : array-like
            The sizes of houses in the training data
        prices : array-like
            The prices of houses in the training data
        model : LinearRegression, optional
            The regression model to use for predictions. Defaults to ClosedFormLinearRegression.
        """
        self.sizes = np.array(sizes)
        self.prices = np.array(prices)
        self.model = model or ClosedFormLinearRegression()

    def predict(self, size):
        """
        Predict the price of a house based on its size.

        Parameters:
        -----------
        size : float
            The size of the house

        Returns:
        --------
        float
            The predicted price of the house
        """
        x = self.sizes
        y = self.prices
        w, b = self.model.fit(x, y)
        return w * size + b
