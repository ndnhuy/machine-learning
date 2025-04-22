import unittest
import numpy as np
import os
import matplotlib.pyplot as plt
from visualizer.png_model_visualizer import PNGModelVisualizer
from visualizer.interactive_model_visualizer import InteractiveModelVisualizer
from sklearn.datasets import make_regression
from linear_regression.models.closed_form_linear_regression import ClosedFormLinearRegression


class TestDataProvider:
    @staticmethod
    def simple_house_data():
        # Generate data points with noise using scikit-learn
        X, y = make_regression(
            n_samples=50,
            n_features=1,
            noise=10,  # Add substantial noise
            bias=10,  # Set a bias/intercept
            random_state=42  # For reproducibility
        )
        # Scale X to represent house sizes in square feet
        X = (X * 500) + 1500  # Center around 1500 sq ft
        # Ensure y values are positive and in a reasonable house price range
        y = y + 250000  # Center around $250,000

        return X.flatten(), y


class TestPNGModelVisualizer(unittest.TestCase):

    def setUp(self):
        """Set up test data and model"""
        self.sizes, self.prices = TestDataProvider.simple_house_data()

        # Define test output directory path
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_output')

        # Create the directory if it doesn't exist
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """Clean up after test, but preserve output files"""
        # Close any open matplotlib figures
        plt.close('all')

    def test_visualization_saves_to_png_file(self):
        """Test that the PNG visualizer correctly saves a file"""
        output_path = os.path.join(self.test_dir, "test_visualization.png")

        # Create visualizer
        visualizer = PNGModelVisualizer(
            output_path=output_path,
            x_label="House Size (sq ft)",
            y_label="Price ($)"
        )

        model = ClosedFormLinearRegression()
        w, b = model.fit(self.sizes, self.prices)
        y_pred = w * self.sizes + b

        # Generate visualization
        visualizer.visualize(self.sizes, self.prices, y_pred)

        # Check that file exists
        self.assertTrue(os.path.exists(output_path),
                        "PNG file was not created")

        # Check that file is not empty
        self.assertGreater(os.path.getsize(
            output_path), 0, "PNG file is empty")

        print(f"Test visualization saved to: {output_path}")

    def test_interactive_visualization(self):
        """Test that the interactive visualizer"""
        # Create visualizer
        visualizer = InteractiveModelVisualizer(
            x_label="House Size (sq ft)",
            y_label="Price ($)"
        )

        model = ClosedFormLinearRegression()
        w, b = model.fit(self.sizes, self.prices)
        y_pred = w * self.sizes + b

        # Generate visualization
        visualizer.visualize(self.sizes, self.prices, y_pred)


if __name__ == "__main__":
    unittest.main()
