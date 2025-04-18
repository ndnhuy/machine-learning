import unittest
import numpy as np
from house_price_predictor import HousePricePredictor


class TestHousePricePredictor(unittest.TestCase):

    def setUp(self):
        """Set up test data"""
        # Generate synthetic training data using y = w * x
        w = 3000
        self.sizes = [50, 75, 100, 150, 200]
        self.prices = [150000, 225000, 300000, 450000, 600000]
        self.predictor = HousePricePredictor(self.sizes, self.prices)

    def test_initialization(self):
        """Test that the constructor correctly stores the input data"""
        np.testing.assert_array_equal(
            self.predictor.sizes, np.array(self.sizes))
        np.testing.assert_array_equal(
            self.predictor.prices, np.array(self.prices))

    def test_predict_with_close_form_least_squares_method(self):
        """Test that the predict method returns a value (dummy test)"""
        prediction = self.predictor.predict(50)
        self.assertEqual(prediction, 150000)
        prediction = self.predictor.predict(150)
        self.assertEqual(prediction, 450000)


if __name__ == "__main__":
    unittest.main()
