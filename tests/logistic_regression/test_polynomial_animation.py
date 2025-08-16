import numpy as np
from sklearn.datasets import make_classification

from logistic_regression.polynomial_logistic_regression import PolynomialLogisticRegression, visualize_polynomial_boundary
from logistic_regression.gradient_logistic_regression import GradientLogisticRegression
from visualizer.interactive_gif_model_visualizer import InteractiveGifModelVisualizer


def test_polynomial_vs_linear_comparison():
    """Compare your original linear model with polynomial model on the same data."""
    
    # Use your exact same data generation
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=2,
        flip_y=0.2,
        class_sep=0.7,
        random_state=42
    )

    print("Testing Polynomial Logistic Regression...")
    print("=" * 50)
    
    # Test different polynomial degrees
    for degree in [1, 2, 3]:
        print(f"\nDegree {degree} Polynomial Features:")
        print("-" * 30)
        
        model = PolynomialLogisticRegression(
            learning_rate=0.1,  # Reduced learning rate for stability
            iterations=5000,    # Reduced iterations
            degree=degree
        )
        
        # Fit the model
        w, b = model.fit(X, y)
        
        # Calculate accuracy
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Feature names: {model.get_feature_names(['x0', 'x1'])}")
        print(f"Number of features: {len(w)}")
        
        # Visualize the decision boundary
        visualize_polynomial_boundary(
            model, X, y, 
            title=f"Polynomial Logistic Regression (degree={degree})\nAccuracy: {accuracy:.3f}"
        )


def test_polynomial_with_simple_visualization():
    """Test polynomial logistic regression with better visualization."""
    
    # Same data as your test
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=2,
        flip_y=0.2,
        class_sep=0.7,
        random_state=42
    )

    print("Training Polynomial Logistic Regression...")
    
    # Compare linear vs polynomial
    from logistic_regression.gradient_logistic_regression import GradientLogisticRegression
    
    # Linear model
    linear_model = GradientLogisticRegression(learning_rate=0.5)
    linear_w, linear_b = linear_model.fit(X, y)
    
    # Polynomial model
    poly_model = PolynomialLogisticRegression(
        learning_rate=0.1,
        iterations=2000,
        degree=2
    )
    poly_w, poly_b = poly_model.fit(X, y)
    
    # Calculate accuracies
    linear_predictions = (1 / (1 + np.exp(-(X @ linear_w + linear_b))) >= 0.5).astype(int)
    linear_accuracy = np.mean(linear_predictions == y)
    
    poly_predictions = poly_model.predict(X)
    poly_accuracy = np.mean(poly_predictions == y)
    
    print(f"Linear Accuracy: {linear_accuracy:.3f}")
    print(f"Polynomial Accuracy: {poly_accuracy:.3f}")
    print(f"Improvement: {poly_accuracy - linear_accuracy:.3f}")
    
    # Show final polynomial visualization
    visualize_polynomial_boundary(
        poly_model, X, y,
        title=f"Polynomial vs Linear Decision Boundary\nLinear: {linear_accuracy:.3f}, Polynomial: {poly_accuracy:.3f}"
    )


if __name__ == "__main__":
    # Test polynomial vs linear comparison
    test_polynomial_vs_linear_comparison()
    
    # Test with simpler visualization
    print("\n" + "="*60)
    print("POLYNOMIAL VS LINEAR COMPARISON")
    print("="*60)
    test_polynomial_with_simple_visualization()
