import unittest
import numpy as np
from house_price_predictor import HousePricePredictor
from models.closed_form_linear_regression import ClosedFormLinearRegression
from models.gradient_descent_linear_regression import GradientDescentLinearRegression
from sklearn.datasets import make_regression


class TestDataProvider:
    @staticmethod
    def trainingDataOfZeroDeriatives():
        # Generate data points lying on a line through origin (noise=0, bias=0)
        X, y = make_regression(n_samples=20, n_features=1,
                               noise=0.0, random_state=42, bias=0.0)
        sizes = X.flatten()
        prices = y
        return sizes, prices

    @staticmethod
    def trainingDataWithBias():
        # Generate data points lying on a line with positive bias (noise=0)
        X, y = make_regression(n_samples=20, n_features=1,
                               noise=0.0, random_state=42, bias=1000.0)
        sizes = X.flatten()
        prices = y
        return sizes, prices

    @staticmethod
    def trainingDataWithNoiseAndBias():
        # Generate data points lying on a line with positive bias and noise
        X, y = make_regression(n_samples=20, n_features=1,
                               noise=100, random_state=42, bias=1000.0)
        sizes = X.flatten()
        prices = y
        return sizes, prices

    @staticmethod
    def trainingDataDummy():
        # Generate dummy data points
        sizes = np.array([800, 1200, 1500, 1800, 2200])  # in square feet
        prices = np.array([150000, 210000, 260000, 310000, 380000])  # in USD
        return sizes, prices


class TestHousePricePredictor(unittest.TestCase):

    def setUp(self):
        """Set up test data"""
        # Retrieve synthetic training data
        self.sizes, self.prices = TestDataProvider.trainingDataOfZeroDeriatives()

    def test_initialization(self):
        """Test that the constructor correctly stores the input data"""
        sizes, prices = TestDataProvider.trainingDataOfZeroDeriatives()
        predictor = HousePricePredictor(sizes, prices)
        np.testing.assert_array_equal(predictor.sizes, np.array(self.sizes))
        np.testing.assert_array_equal(
            predictor.prices, np.array(self.prices))

    def test_predict_with_close_form_least_squares_method(self):
        sizes, prices = TestDataProvider.trainingDataOfZeroDeriatives()
        predictor = HousePricePredictor(
            sizes, prices, ClosedFormLinearRegression())

        self.assertAlmostEqual(
            predictor.predict(sizes[0]), prices[0],
            delta=0.5,
        )
        self.assertAlmostEqual(
            predictor.predict(sizes[1]), prices[1],
            delta=0.5,
        )

    def test_predict_with_gradient_descent_method(self):
        """Test gradient descent method on all provided training data"""
        methods = [
            TestDataProvider.trainingDataOfZeroDeriatives,
            TestDataProvider.trainingDataWithBias,
            TestDataProvider.trainingDataWithNoiseAndBias,
            TestDataProvider.trainingDataDummy
        ]
        for method in methods:
            with self.subTest(data=method.__name__):
                sizes, prices = method()
                predictor = HousePricePredictor(
                    sizes, prices, GradientDescentLinearRegression(
                        learning_rate=0.01, iterations=10000)
                )
                for x, y in zip(sizes, prices):
                    self.assertAlmostEqual(
                        predictor.predict(x), y,
                        delta=5000,
                        msg=f"Failed for {method.__name__} with tolerance"
                    )


if __name__ == "__main__":
    unittest.main()
