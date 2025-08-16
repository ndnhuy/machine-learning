"""
Real-world examples of when decision boundaries are non-linear.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def real_world_examples():
    """Show examples where linear boundaries completely fail."""
    
    # Example 1: Concentric circles (like medical imaging)
    print("REAL-WORLD EXAMPLE 1: Medical Imaging / Radar Detection")
    print("Scenario: Detecting tumors (inner circle) vs healthy tissue (outer ring)")
    
    X_circles, y_circles = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)
    
    # Example 2: Moons (like customer segmentation)  
    print("\nREAL-WORLD EXAMPLE 2: Customer Behavior Segmentation")
    print("Scenario: Premium customers vs regular customers based on spending patterns")
    
    X_moons, y_moons = make_moons(n_samples=300, noise=0.15, random_state=42)
    
    # Example 3: Multiple clusters (like image recognition)
    print("\nREAL-WORLD EXAMPLE 3: Image Recognition")
    print("Scenario: Distinguishing cats vs dogs based on features")
    
    # Create clusters where each class appears in multiple regions
    centers_class0 = [[-2, -2], [2, 2]]
    centers_class1 = [[-2, 2], [2, -2]]
    
    X_class0, _ = make_blobs(n_samples=150, centers=centers_class0, cluster_std=0.8, random_state=42)
    X_class1, _ = make_blobs(n_samples=150, centers=centers_class1, cluster_std=0.8, random_state=42)
    
    X_multi = np.vstack([X_class0, X_class1])
    y_multi = np.hstack([np.zeros(150), np.ones(150)])
    
    # Test linear vs polynomial on all examples
    datasets = [
        (X_circles, y_circles, "Medical: Tumor Detection", "Concentric circles need circular boundaries"),
        (X_moons, y_moons, "Business: Customer Segmentation", "Moon shapes need curved boundaries"),
        (X_multi, y_multi, "AI: Image Recognition", "Multiple clusters need complex boundaries")
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('When Linear Decision Boundaries Completely Fail', fontsize=16, fontweight='bold')
    
    for i, (X, y, title, explanation) in enumerate(datasets):
        # Fit models
        linear_model = LogisticRegression(random_state=42)
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('logistic', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        linear_model.fit(X, y)
        poly_model.fit(X, y)
        
        linear_acc = linear_model.score(X, y)
        poly_acc = poly_model.score(X, y)
        
        # Create decision boundary plots
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Data plot
        ax = axes[i, 0]
        colors = ['#0099FF', '#FF2222']
        for class_idx in [0, 1]:
            mask = y == class_idx
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[class_idx], s=50, alpha=0.7, 
                      label=f'Class {class_idx}')
        ax.set_title(f'{title}\n{explanation}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Linear boundary plot
        ax = axes[i, 1]
        Z_linear = linear_model.predict(mesh_points).reshape(xx.shape)
        ax.contourf(xx, yy, Z_linear, alpha=0.3, cmap=plt.cm.RdYlBu)
        ax.contour(xx, yy, Z_linear, levels=[0.5], colors='black', linestyles='-', linewidths=2)
        
        for class_idx in [0, 1]:
            mask = y == class_idx
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[class_idx], s=50, alpha=0.7)
        ax.set_title(f'Linear Boundary\nAccuracy: {linear_acc:.1%}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Polynomial boundary plot  
        ax = axes[i, 2]
        Z_poly = poly_model.predict(mesh_points).reshape(xx.shape)
        ax.contourf(xx, yy, Z_poly, alpha=0.3, cmap=plt.cm.RdYlBu)
        ax.contour(xx, yy, Z_poly, levels=[0.5], colors='black', linestyles='-', linewidths=2)
        
        for class_idx in [0, 1]:
            mask = y == class_idx
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[class_idx], s=50, alpha=0.7)
        ax.set_title(f'Curved Boundary\nAccuracy: {poly_acc:.1%}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Print comparison
        improvement = poly_acc - linear_acc
        print(f"\n{title}:")
        print(f"  Linear accuracy:  {linear_acc:.1%}")  
        print(f"  Curved accuracy:  {poly_acc:.1%}")
        print(f"  Improvement:      {improvement:.1%} ({improvement*100:.1f} percentage points)")
    
    plt.tight_layout()
    plt.savefig('/home/huynguyen/workspace/machinelearning/real_world_nonlinear_examples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nKEY INSIGHTS FOR YOUR PROJECT:")
    print("=" * 40)
    print("• Your dataset has n_clusters_per_class=2, creating multiple regions per class")
    print("• This is similar to Example 3 (multiple clusters) - very common in real ML")
    print("• Linear boundaries force a single straight line separation")
    print("• Polynomial features allow elliptical, curved, and complex boundaries")
    print("• Even small improvements (1.5%) can be significant in production systems")
    print("• The visual improvement in decision boundary fit is even more important")


if __name__ == "__main__":
    real_world_examples()
