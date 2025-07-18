import numpy as np

from logistic_regression.main import sigmoid

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
