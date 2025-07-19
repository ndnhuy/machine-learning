from logistic_regression.main import sigmoid
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive window


X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)
# plot the points
plt.figure(figsize=(4, 4))
ax = plt.gca()
for x, label in zip(X, y):
    if label == 1:
        plt.scatter(x[0], x[1], color='#FF2222', marker='x', s=200, linewidths=4,
                    label='y=1' if 'y=1' not in ax.get_legend_handles_labels()[1] else "")
    else:
        plt.scatter(x[0], x[1], facecolors='none', edgecolors='#0099FF', marker='o', s=200,
                    linewidths=3, label='y=0' if 'y=0' not in ax.get_legend_handles_labels()[1] else "")
ax.axis([0, 4, 0, 3.5])
ax.set_xlabel("$x_0$")
ax.set_ylabel("$x_1$")

# plot the decision boundary
x0 = np.arange(-10, 11)
x1 = 3 - x0
ax.plot(x0, x1, c="r", label="Decision Boundary")
ax.set_title("Decision Boundary")
ax.legend()

plt.show()
