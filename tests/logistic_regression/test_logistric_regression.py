import numpy as np

from logistic_regression.gradient_logistic_regression import GradientLogisticRegression
from logistic_regression.main import sigmoid
from logistic_regression.compute_functions import compute_cost_logistic, compute_gradient_logistic
from visualizer.interactive_gif_model_visualizer import InteractiveGifModelVisualizer
from visualizer.interactive_model_visualizer import InteractiveModelVisualizer


def test_sigmoid():
    # Generate an array of evenly spaced values between -10 and 10
    z_tmp = np.arange(-10, 11)
    # Use the function implemented above to get the sigmoid values
    y = sigmoid(z_tmp)
    # Check that all values are between 0 and 1 (inclusive)
    assert np.all((y >= 0) & (y <= 1)), "Sigmoid output not in [0, 1] range"
    # Print for visual inspection
    np.set_printoptions(precision=3)
    print("Input (z), Output (sigmoid(z))")
    print(np.c_[z_tmp, y])


def test_compute_cost_logistic():

    # Example data
    X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    # 1D array to 2D column array
    y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)
    w = np.array([1, 1])  # w0=1 and w1=1
    b = -3

    # Compute cost
    cost = compute_cost_logistic(X, y, w, b)

    # Print the cost for visual inspection
    print(f"Computed cost: {cost}")


def test_compute_gradient_logistic():
    # Example data
    X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)
    w = np.array([2., 3.])  # w0=2 and w1=3
    b = 1.

    # Compute gradient
    dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

    # Print the gradients for visual inspection
    print(f"Gradient dj_db: {dj_db}")
    print(f"Gradient dj_dw: {dj_dw}")


def test_batch_gradient_descent():
    # Example data
    X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = GradientLogisticRegression()
    w, b = model.fit(X, y)

    visualizer = InteractiveModelVisualizer(
        X[:, 0],
        X[:, 1],
        x_label="x0",
        y_label="x1"
    )

    # w0*x0 + w1*x1 + b = 0
    # x1 = -(w[0] * x0 + b) / w[1]
    x0_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x1_vals = -(w[0] * x0_vals + b) / w[1]
    visualizer.accept_x_y(x0_vals, x1_vals)
    visualizer.show()

    # Print final weights and bias for visual inspection
    print(f"Final weights: {w}")
    print(f"Final bias: {b}")


# pytest tests/logistic_regression/test_logistric_regression.py::test_batch_gradient_descent_with_animation -vs
def test_batch_gradient_descent_with_animation():
    # Example data
    X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = GradientLogisticRegression(learning_rate=0.5)

    visualizer = InteractiveGifModelVisualizer(
        X[:, 0],
        X[:, 1],
        x_label="x0",
        y_label="x1"
    )

    def visualizer_callback(w, b):
        x0_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        x1_vals = -(w[0] * x0_vals + b) / w[1]
        visualizer.accept_x_y(x0_vals, x1_vals)

    model.setIterationConsumer(visualizer_callback)
    model.fit(X, y)

    # w0*x0 + w1*x1 + b = 0
    # x1 = -(w[0] * x0 + b) / w[1]
    # x0_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    # x1_vals = -(w[0] * x0_vals + b) / w[1]
    # visualizer.accept_x_y(x0_vals, x1_vals)
    visualizer.show()
