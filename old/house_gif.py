import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import make_regression

# === 1. Sample Data ===
x, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
x = x.flatten()  # Flatten X to 1D
y = y / 10       # Scale y for visualization

# === 2. Initialize parameters for gradient descent ===
w = 0.0
b = 0.0
learning_rate = 0.1
iterations = 100  # Fewer iterations so the GIF is manageable

# === 3. Set up the figure and axes ===
fig, ax = plt.subplots()
ax.scatter(x, y, label='Data')
line, = ax.plot([], [], 'r-', label='Fitted Line')
ax.set_xlabel("Size (m²)")
ax.set_ylabel("Price ($1000s)")
ax.set_title("Gradient Descent Progress")
ax.grid(True)
ax.legend()

# Set axes limits for clarity
xmin, xmax = x.min() - 1, x.max() + 1
ymin, ymax = y.min() - 5, y.max() + 5
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# === 4. Animation function ===
def update(frame):
    global w, b
    # One step of gradient descent
    y_pred = w * x + b
    dw = (-2 / len(x)) * np.sum(x * (y - y_pred))
    db = (-2 / len(x)) * np.sum(y - y_pred)
    w -= learning_rate * dw
    b -= learning_rate * db

    # Update line data
    line.set_data(x, w*x + b)
    return line,

# === 5. Create animation ===
ani = animation.FuncAnimation(
    fig,
    update,
    frames=iterations,
    interval=200,    # Time in ms between frames
    blit=True
)

# === 6. Save animation as GIF ===
ani.save("gradient_descent.gif", writer="pillow")