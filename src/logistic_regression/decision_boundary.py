from logistic_regression.main import sigmoid
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from logistic_regression.compute_functions import compute_cost_logistic
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive window


# m examples with n features (m=6, n=2)
X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
# .reshape(-1, 1) is a quick way to turn a flat list into a column vector.
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

# Calculate costs for boundaries before plotting
w1 = np.array([1, 1])
b1 = -3
cost1 = compute_cost_logistic(X, y, w1, b1)
w2 = np.array([1, 1])
b2 = -4
cost2 = compute_cost_logistic(X, y, w2, b2)

# plot the decision boundary
x0 = np.arange(-10, 11)
x1 = 3 - x0
ax.plot(x0, x1, c="r", label="Decision Boundary")
# Annotate cost1 on the first decision boundary
mid_x0 = (x0[0] + x0[-1]) / 2
mid_x1 = (x1[0] + x1[-1]) / 2
ax.text(mid_x0, mid_x1, f"Cost: {cost1.item():.2f}", color="r", fontsize=10, ha="center", va="bottom", backgroundcolor="white")
ax.set_title("Decision Boundary")
ax.legend()

# plot the other boundary, suppose b = -4, w = [1, 1]
# => we will see it fit the data worse than the first one
x1_other = 4 - x0
ax.plot(x0, x1_other, c="pink", linestyle="--", label="Other Boundary")
# Annotate cost2 on the second decision boundary
mid_x1_other = (x1_other[0] + x1_other[-1]) / 2
ax.text(mid_x0, mid_x1_other, f"Cost: {cost2.item():.2f}", color="pink", fontsize=10, ha="center", va="top", backgroundcolor="white")

# print the cost of each case
print(f"Cost for first boundary (w={w1}, b={b1}): {float(cost1):.2f}")
print(f"Cost for second boundary (w={w2}, b={b2}): {float(cost2):.2f}")

plt.show()
