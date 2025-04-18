import unittest
import numpy as np
from house_price_predictor import HousePricePredictor
from closed_form_linear_regression import ClosedFormLinearRegression
from gradient_descent_linear_regression import GradientDescentLinearRegression
from sklearn.datasets import make_regression


class TestDataProvider:
    @staticmethod
    def trainingDataOfZeroDeriatives():
        # Generate data points lying on a line through origin (noise=0, bias=0)
        X, y = make_regression(n_samples=5, n_features=1,
                               noise=0.0, random_state=42, bias=0.0)
        sizes = X.flatten()
        prices = y
        return sizes, prices

    @staticmethod
    def trainingDataWithBias():
        # Generate data points lying on a line with positive bias (noise=0)
        X, y = make_regression(n_samples=4, n_features=1,
                               noise=0.0, random_state=42, bias=1000.0)
        sizes = X.flatten()
        prices = y
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
        prediction = predictor.predict(sizes[0])
        self.assertEqual(prediction, prices[0])
        prediction = predictor.predict(sizes[1])
        self.assertEqual(prediction, prices[1])

    def test_predict_with_gradient_descent_method(self):
        """Test gradient descent method on all provided training data"""
        methods = [
            TestDataProvider.trainingDataOfZeroDeriatives,
            TestDataProvider.trainingDataWithBias
        ]
        for method in methods:
            with self.subTest(data=method.__name__):
                sizes, prices = method()
                predictor = HousePricePredictor(
                    sizes, prices, GradientDescentLinearRegression(
                        learning_rate=0.01, iterations=10000)
                )
                for x, y in zip(sizes, prices):
                    self.assertAlmostEqual(predictor.predict(x), y, places=0,
                                           msg=f"Failed for {method.__name__}")


if __name__ == "__main__":
    unittest.main()
