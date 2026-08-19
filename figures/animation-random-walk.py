import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

# Parameters
n_steps = 500              # total number of steps in the walk
step_std = 0.50               # standard deviation of each step (controls "wiggliness")
start_point = (-5, -5)       # (x, y) where the walk begins
end_point = (5, 5)           # (x, y) where the walk ends
seed = 42                     # random seed (None for fully random)

# Simulate a standard 2D random walk, then bend it into a Brownian bridge
# that starts exactly at start_point and ends exactly at end_point.
rng = np.random.default_rng(seed)
dx = rng.normal(0, step_std, size=n_steps)
dy = rng.normal(0, step_std, size=n_steps)

w_x = np.concatenate([[0], np.cumsum(dx)])  # free random walk, x
w_y = np.concatenate([[0], np.cumsum(dy)])  # free random walk, y

i = np.arange(n_steps + 1)
frac = i / n_steps  # 0 -> 1 as the walk progresses

x_vals = start_point[0] + (w_x - frac * w_x[-1]) + frac * (end_point[0] - start_point[0])
y_vals = start_point[1] + (w_y - frac * w_y[-1]) + frac * (end_point[1] - start_point[1])

# Fixed square bounds that comfortably contain the whole path
margin = 1
x_range = max(np.abs(x_vals).max(), abs(start_point[0]), abs(end_point[0]))
y_range = max(np.abs(y_vals).max(), abs(start_point[1]), abs(end_point[1]))
half_range = max(x_range, y_range) + margin

# Animation setup
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-half_range, half_range)
ax.set_ylim(-half_range, half_range)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axis('off')
fig.patch.set_facecolor("#FFFFFF")
fig.tight_layout(pad=0, w_pad=0, h_pad=0)

# Start and end point markers (static, gray)
ax.plot(*start_point, 'o', markersize=8, color='gray', zorder=1)
ax.plot(*end_point, 'X', markersize=10, color='gray', zorder=1)

# Path trail, colored by step order (gradient), and current-position dot
trail = LineCollection([], cmap='viridis', linewidth=2, alpha=0.85, zorder=2)
trail.set_array(np.array([]))
ax.add_collection(trail)
head = ax.scatter([], [], s=90, color='#058743', zorder=3, edgecolors='white',
                   linewidths=0.8)

def init():
    trail.set_segments([])
    head.set_offsets(np.empty((0, 2)))
    return trail, head

def update(frame):
    n = frame + 1
    points = np.column_stack([x_vals[:n + 1], y_vals[:n + 1]])
    segments = np.stack([points[:-1], points[1:]], axis=1)

    trail.set_segments(segments)
    trail.set_array(np.arange(len(segments)))
    trail.set_clim(0, n_steps)

    head.set_offsets([[x_vals[n], y_vals[n]]])

    return trail, head

ani = FuncAnimation(fig, update, frames=n_steps, init_func=init, blit=True)

# Save as GIF
ani.save('random-walk.gif', writer=PillowWriter(fps=60), dpi=150)
