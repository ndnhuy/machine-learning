import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def create_test_datasets():
    """Create different types of datasets to demonstrate boundary types"""
    
    # 1. Your current dataset (linearly separable-ish)
    X1, y1 = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_clusters_per_class=2,
        flip_y=0.2, class_sep=0.7, random_state=42
    )
    
    # 2. Circular boundary needed
    X2, y2 = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=42)
    
    # 3. Moon-shaped boundary needed
    X3, y3 = make_moons(n_samples=200, noise=0.2, random_state=42)
    
    return [(X1, y1, "Your Current Data"), (X2, y2, "Circular Data"), (X3, y3, "Moon Data")]

def test_different_boundaries():
    """Compare linear vs non-linear boundaries on different datasets"""
    
    datasets = create_test_datasets()
    
    # Different models
    models = [
        ("Linear Logistic", LogisticRegression(random_state=42)),
        ("Polynomial Features", Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('logistic', LogisticRegression(random_state=42, max_iter=1000))
        ])),
        ("SVM (RBF)", SVC(kernel='rbf', random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Neural Network", MLPClassifier(hidden_layer_sizes=(10, 10), random_state=42, max_iter=1000))
    ]
    
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(20, 12))
    fig.suptitle('Decision Boundaries: Linear vs Non-Linear Models', fontsize=16)
    
    for i, (X, y, title) in enumerate(datasets):
        for j, (model_name, model) in enumerate(models):
            ax = axes[i, j]
            
            # Fit the model
            model.fit(X, y)
            
            # Create a mesh for decision boundary
            h = 0.02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            # Predict on mesh
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = model.predict(mesh_points)
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
            
            # Plot data points
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
            
            # Calculate accuracy
            accuracy = model.score(X, y)
            ax.set_title(f'{model_name}\n{title}\nAccuracy: {accuracy:.3f}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('/home/huynguyen/workspace/machinelearning/decision_boundaries_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def test_polynomial_logistic_regression():
    """Demonstrate how to add polynomial features to your existing logistic regression"""
    
    # Use your exact same data generation
    X, y = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_clusters_per_class=2,
        flip_y=0.2, class_sep=0.7, random_state=42
    )
    
    # Create polynomial features (degree=2 adds x1^2, x2^2, x1*x2)
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    print("Original features shape:", X.shape)
    print("Polynomial features shape:", X_poly.shape)
    print("Feature names:", poly_features.get_feature_names_out(['x0', 'x1']))
    
    # Train both models
    linear_model = LogisticRegression(random_state=42)
    poly_model = LogisticRegression(random_state=42, max_iter=1000)
    
    linear_model.fit(X, y)
    poly_model.fit(X_poly, y)
    
    # Compare accuracies
    linear_acc = linear_model.score(X, y)
    poly_acc = poly_model.score(X_poly, y)
    
    print(f"\nLinear Logistic Regression Accuracy: {linear_acc:.3f}")
    print(f"Polynomial Logistic Regression Accuracy: {poly_acc:.3f}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot linear boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Linear model
    Z1 = linear_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)
    ax1.contourf(xx, yy, Z1, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    ax1.set_title(f'Linear Boundary\nAccuracy: {linear_acc:.3f}')
    ax1.set_xlabel('x0')
    ax1.set_ylabel('x1')
    
    # Polynomial model  
    mesh_poly = poly_features.transform(np.c_[xx.ravel(), yy.ravel()])
    Z2 = poly_model.predict(mesh_poly)
    Z2 = Z2.reshape(xx.shape)
    ax2.contourf(xx, yy, Z2, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax2.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    ax2.set_title(f'Polynomial Boundary (degree=2)\nAccuracy: {poly_acc:.3f}')
    ax2.set_xlabel('x0')
    ax2.set_ylabel('x1')
    
    plt.tight_layout()
    plt.savefig('/home/huynguyen/workspace/machinelearning/polynomial_vs_linear.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return linear_acc, poly_acc

if __name__ == "__main__":
    print("Testing different decision boundaries...")
    test_different_boundaries()
    
    print("\nTesting polynomial logistic regression...")
    test_polynomial_logistic_regression()
