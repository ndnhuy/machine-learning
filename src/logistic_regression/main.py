from matplotlib import pyplot as plt
import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """

    g = 1/(1+np.exp(-z))

    return g


x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0

plt.figure(figsize=(10, 6))

ax = plt.gca()
# Plot points: y > 0 as red cross, y = 0 as blue circle
for x, y in zip(x_train, y_train):
    if y > 0:
        plt.scatter(x, y, color='#FF2222', marker='x', s=200, linewidths=4, label='y=1' if 'y=1' not in ax.get_legend_handles_labels()[1] else "")
    else:
        plt.scatter(x, y, facecolors='none', edgecolors='#0099FF', marker='o', s=200, linewidths=3, label='y=0' if 'y=0' not in ax.get_legend_handles_labels()[1] else "")
plt.plot(x_train, sigmoid(w_in * x_train + b_in), color='black', linewidth=2, label='Sigmoid Function')
plt.xlabel("Tumor Size")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig("./output.png")  # Save to the same folder as this script
plt.close()  # Close the figure to free memory