import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters
n_points = 500          # total number of random dots
mean = [-1, 1]            # mean of the distribution
std_x = 0.75               # standard deviation in x
std_y = 0.75               # standard deviation in y
rho = 0                 # correlation between x and y
seed = 42                 # random seed (None for fully random)

# Covariance matrix
cov = [[std_x**2, rho * std_x * std_y],
       [rho * std_x * std_y, std_y**2]]

# Sample points from the multivariate normal distribution
rng = np.random.default_rng(seed)
xy_vals = rng.multivariate_normal(mean, cov, size=n_points)
x_vals = xy_vals[:, 0]
y_vals = xy_vals[:, 1]

# Distance of each point from the center (used to color the gradient)
dist_vals = np.sqrt((x_vals - mean[0])**2 + (y_vals - mean[1])**2)

# Animation setup
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axis('off')
fig.patch.set_facecolor("#FFFFFF")
fig.tight_layout(pad=0, w_pad=0, h_pad=0)

# Concentric "dart target" circles in the background
circle_center = [0,0]
circle_radii = [1, 2, 3, 4]
for r in circle_radii:
    circle = plt.Circle(circle_center, r, fill=False, color='gray', linewidth=1,
                         zorder=1)
    ax.add_patch(circle)

# Dots component (one scatter artist holding all revealed points so far,
# colored by distance from the center)
dots = ax.scatter([], [], s=50, c=[], cmap='viridis', alpha=0.85,
                   vmin=dist_vals.min(), vmax=dist_vals.max(), zorder=2)

def init():
    dots.set_offsets(np.empty((0, 2)))
    dots.set_array(np.array([]))
    return dots,

def update(frame):
    xy = xy_vals[:frame + 1]
    dots.set_offsets(xy)
    dots.set_array(dist_vals[:frame + 1])
    return dots,

ani = FuncAnimation(fig, update, frames=len(x_vals), init_func=init, blit=True)

# Save as GIF
ani.save('bias-variance.gif', writer=PillowWriter(fps=60), dpi=150)
