import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive window
from matplotlib import pyplot as plt
import numpy as np

from logistic_regression.main import sigmoid

z = np.arange(-10, 11)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))

ax.plot(z, sigmoid(z), c="b", label="sigmoid(z)")
ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
ax.legend()

plt.show()
