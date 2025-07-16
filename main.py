import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import make_regression

fig, ax = plt.subplots(2, 2, figsize=(10, 8))  # Create a 2x2 grid of subplots
ax[0, 0].plot([1, 2, 3], [4, 5, 6])           # Top-left subplot
ax[0, 1].scatter([1, 2, 3], [6, 5, 4])        # Top-right subplot
ax[1, 0].bar([1, 2, 3], [7, 8, 9])            # Bottom-left subplot
ax[1, 1].hist([1, 2, 2, 3, 3, 3])             # Bottom-right subplot
# plt.scatter(np.array([1, 2, 3]), np.array([6, 7, 8]), label='Data')
# plt.plot(x, y_pred, color='red', label='Fitted Line')
# plt.plot(np.array([1, 2, 6]), np.array([6, 7, 8]), color='red', label='Fitted Line')

plt.xlabel("House Size (sq ft)")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.savefig("./test_visualization.png")
plt.close()  # Close the figure to free memory
