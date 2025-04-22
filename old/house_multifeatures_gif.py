import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import make_regression
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

def gradient_descent_frames(X, y, learning_rate=0.01, iterations=1000, frame_step=10):
    """Perform gradient descent and collect parameters every 'frame_step' iterations."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    b = 0.0
    frames = []
    for i in range(iterations):
        y_pred = np.dot(X, weights) + b
        error = y - y_pred
        dw = (-2 / n_samples) * np.dot(X.T, error)
        db = (-2 / n_samples) * np.sum(error)
        weights -= learning_rate * dw
        b -= learning_rate * db

        if i % frame_step == 0:
            # Save a copy of parameters and iteration number
            frames.append((weights.copy(), b, i))
    return frames

# Generate data with 2 features: house size and number of bedrooms
X, y = make_regression(n_samples=1000, n_features=2, noise=100, random_state=42)
y = y / 10  # Scale the prices

# Get intermediate parameter values (frames) via gradient descent
frames = gradient_descent_frames(X, y, learning_rate=0.01, iterations=1000, frame_step=10)
# Get final parameters for reporting
final_weights, final_b, _ = frames[-1]
print(f"Learned model: y = {final_weights[0]:.2f}*size + {final_weights[1]:.2f}*bedrooms + {final_b:.2f}")

# Set up the figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(X[:, 0], X[:, 1], y, label='Data', alpha=0.6)

# Create a grid for plotting the regression plane
x0_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
x1_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
X0, X1 = np.meshgrid(x0_range, x1_range)

# Initialize the surface plot (empty so we can update it later)
surf = [None]  # use a mutable object to hold the current surface

ax.set_xlabel("Size (m²)")
ax.set_ylabel("Number of Bedrooms")
ax.set_zlabel("Price ($1000s)")
plt.legend()

def update(frame_data):
    weights, b, iteration = frame_data
    # Compute current regression plane
    plane_y = weights[0] * X0 + weights[1] * X1 + b

    # Remove previous surface if it exists by calling its remove() method
    if surf[0] is not None:
        surf[0].remove()
    
    # Plot new surface and save its handle
    surf[0] = ax.plot_surface(X0, X1, plane_y, color='red', alpha=0.5)
    ax.set_title(f"Iteration: {iteration}")
    return surf[0],

# Create and save the animation as a GIF
ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
writer = PillowWriter(fps=10)
ani.save("fitted_line_multifeatures.gif", writer=writer)