import copy
import math
import numpy as np

from logistic_regression.compute_functions import compute_cost_logistic, compute_gradient_logistic


class GradientLogisticRegression():

    def __init__(self, learning_rate: float = 0.01, iterations: int = 10000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.iterationConsumer = lambda w, b: None

    def fit(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Fit a logistic regression model on x and y and return (weights, bias).

        Args:
            x (np.ndarray): Input features, shape (m, n) where m is the number of samples and n is the number of features.
            y (np.ndarray): Target values, shape (m,) where m is the number of samples.
        Returns:
            tuple[np.ndarray, float]: A tuple containing the model weights (shape (n,)) and the bias (scalar).
        """
        w, b, _ = self.gradient_descent(x, y, np.zeros(
            x.shape[1]), 0.0, self.learning_rate, self.iterations)
        return w, b

    def setIterationConsumer(self, consumer_fn):
        """
        Set the function to consume (w, b) after each iteration.
        """
        self.iterationConsumer = consumer_fn

    def gradient_descent(self, X, y, w_in, b_in, alpha, num_iters):
        """
        Performs batch gradient descent

        Args:
        X (ndarray (m,n)   : Data, m examples with n features
        y (ndarray (m,))   : target values
        w_in (ndarray (n,)): Initial values of model parameters  
        b_in (scalar)      : Initial values of model parameter
        alpha (float)      : Learning rate
        num_iters (scalar) : number of iterations to run gradient descent

        Returns:
        w (ndarray (n,))   : Updated values of parameters
        b (scalar)         : Updated value of parameter 
        """
        # An array to store cost J and w's at each iteration primarily for graphing later
        J_history = []
        w = copy.deepcopy(w_in)  # avoid modifying global w within function
        b = b_in

        for i in range(num_iters):
            # Calculate the gradient and update the parameters
            dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

            # Update Parameters using w, b, alpha and gradient
            w = w - alpha * dj_dw
            b = b - alpha * dj_db

            self.iterationConsumer(w, b)

            # Save cost J at each iteration
            if i < 100000:      # prevent resource exhaustion
                J_history.append(compute_cost_logistic(X, y, w, b))

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i % math.ceil(num_iters / 10) == 0:
                print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

        return w, b, J_history  # return final w,b and J history for graphing
