"""
Direct comparison using your exact test case to answer your question about decision boundaries.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from logistic_regression.gradient_logistic_regression import GradientLogisticRegression
from logistic_regression.polynomial_logistic_regression import PolynomialLogisticRegression


def answer_your_question():
    """
    Answer: Are decision boundaries always linear? Should yours be non-linear?
    """
    print("ANSWERING YOUR QUESTION ABOUT DECISION BOUNDARIES")
    print("=" * 60)
    
    # Use your EXACT same data generation from your test
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=2,
        flip_y=0.2,
        class_sep=0.7,
        random_state=42
    )
    
    # Train both models
    print("Training both models on your exact dataset...")
    
    # Your original linear model
    linear_model = GradientLogisticRegression(learning_rate=0.5)
    linear_w, linear_b = linear_model.fit(X, y)
    
    # Polynomial model for curved boundaries
    poly_model = PolynomialLogisticRegression(learning_rate=0.1, iterations=2000, degree=2)
    poly_w, poly_b = poly_model.fit(X, y)
    
    # Calculate accuracies
    linear_pred = (1 / (1 + np.exp(-(X @ linear_w + linear_b))) >= 0.5).astype(int)
    poly_pred = poly_model.predict(X)
    
    linear_acc = np.mean(linear_pred == y)
    poly_acc = np.mean(poly_pred == y)
    
    print(f"\nRESULTS:")
    print(f"Linear boundary accuracy: {linear_acc:.3f} ({linear_acc*100:.1f}%)")
    print(f"Curved boundary accuracy:  {poly_acc:.3f} ({poly_acc*100:.1f}%)")
    print(f"Improvement with curves:   {poly_acc-linear_acc:.3f} ({(poly_acc-linear_acc)*100:.1f} percentage points)")
    
    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Common plot parameters
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Plot 1: Linear boundary (your current approach)
    Z1 = (1 / (1 + np.exp(-(np.c_[xx.ravel(), yy.ravel()] @ linear_w + linear_b))) >= 0.5).astype(int)
    Z1 = Z1.reshape(xx.shape)
    ax1.contourf(xx, yy, Z1, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax1.contour(xx, yy, Z1, levels=[0.5], colors='red', linestyles='-', linewidths=3)
    
    # Plot data points
    for label, color, marker, name in [(0, '#0099FF', 'o', 'y=0'), (1, '#FF2222', 'x', 'y=1')]:
        mask = (y == label)
        if marker == 'x':
            ax1.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker, s=100, linewidths=2, label=name)
        else:
            ax1.scatter(X[mask, 0], X[mask, 1], facecolors='none', edgecolors=color, marker=marker, 
                       s=100, linewidths=2, label=name)
    
    ax1.set_title(f'LINEAR Boundary (Your Current Method)\nAccuracy: {linear_acc:.3f} ({linear_acc*100:.1f}%)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('x0', fontsize=12)
    ax1.set_ylabel('x1', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Curved boundary (better approach)
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z2 = poly_model.predict(mesh_points)
    Z2 = Z2.reshape(xx.shape)
    ax2.contourf(xx, yy, Z2, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax2.contour(xx, yy, Z2, levels=[0.5], colors='red', linestyles='-', linewidths=3)
    
    # Plot data points
    for label, color, marker, name in [(0, '#0099FF', 'o', 'y=0'), (1, '#FF2222', 'x', 'y=1')]:
        mask = (y == label)
        if marker == 'x':
            ax2.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker, s=100, linewidths=2, label=name)
        else:
            ax2.scatter(X[mask, 0], X[mask, 1], facecolors='none', edgecolors=color, marker=marker, 
                       s=100, linewidths=2, label=name)
    
    ax2.set_title(f'CURVED Boundary (Polynomial Features)\nAccuracy: {poly_acc:.3f} ({poly_acc*100:.1f}%)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('x0', fontsize=12)
    ax2.set_ylabel('x1', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Answer: Should Your Decision Boundary Be Non-Linear?', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/huynguyen/workspace/machinelearning/linear_vs_curved_boundary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyze the misclassified points
    linear_errors = np.sum(linear_pred != y)
    poly_errors = np.sum(poly_pred != y)
    
    print(f"\nMISCLASSIFICATION ANALYSIS:")
    print(f"Linear boundary misclassifies: {linear_errors} out of {len(y)} points")
    print(f"Curved boundary misclassifies:  {poly_errors} out of {len(y)} points")
    print(f"Curved boundary fixes:         {linear_errors - poly_errors} classification errors")
    
    print(f"\nCONCLUSION:")
    print("=" * 40)
    print("1. Decision boundaries are NOT always linear in real life!")
    print("2. Your data has multiple clusters per class - this often needs curved boundaries")
    print(f"3. For your specific dataset, a curved boundary improves accuracy by {(poly_acc-linear_acc)*100:.1f} percentage points")
    print("4. Polynomial features (degree=2) add x0^2, x1^2, and x0*x1 terms")
    print("5. This allows the model to learn elliptical, parabolic, and other curved shapes")
    
    return linear_acc, poly_acc


if __name__ == "__main__":
    answer_your_question()
