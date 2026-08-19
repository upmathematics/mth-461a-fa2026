import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ============================================================
# Parameters
# ============================================================

n_steps = 500
seed = 42

n_states = 5

# Transition probabilities
#
# Each row = current state
# Each column = next state

transition_matrix = np.array([
    # S1     S2     S3     S4     S5
    [0.20,  0.10,  0.15,  0.20,  0.35],  # S1 → S5
    [0.10,  0.20,  0.15,  0.40,  0.15],  # S2 → S4
    [0.15,  0.20,  0.20,  0.30, 0.15],  # S3 → S4/S2
    [0.15,  0.40,  0.15,  0.20,  0.10],  # S4 → S2
    [0.35,  0.20,  0.15,  0.10, 0.20],  # S5 → S1
])

start_state = 0


# ============================================================
# Validate parameters
# ============================================================

if transition_matrix.shape != (n_states, n_states):
    raise ValueError(
        f"Transition matrix must have shape "
        f"({n_states}, {n_states})"
    )

if not np.allclose(transition_matrix.sum(axis=1), 1):
    raise ValueError(
        "Each row of the transition matrix must sum to 1."
    )

if np.any(transition_matrix < 0):
    raise ValueError(
        "Transition probabilities cannot be negative."
    )

if not 0 <= start_state < n_states:
    raise ValueError(
        "start_state must be between 0 and n_states - 1."
    )


# ============================================================
# Generate state positions
# ============================================================

angles = np.linspace(
    0,
    2 * np.pi,
    n_states,
    endpoint=False
)

radius = 4

positions = np.column_stack([
    radius * np.cos(angles),
    radius * np.sin(angles)
])


# ============================================================
# Simulate Markov chain
# ============================================================

rng = np.random.default_rng(seed)

state_sequence = np.empty(
    n_steps + 1,
    dtype=int
)

state_sequence[0] = start_state

for i in range(n_steps):
    current_state = state_sequence[i]

    state_sequence[i + 1] = rng.choice(
        n_states,
        p=transition_matrix[current_state]
    )


# ============================================================
# Animation setup
# ============================================================

fig, ax = plt.subplots(figsize=(5, 5))

margin = 1.5

ax.set_xlim(-radius - margin, radius + margin)
ax.set_ylim(-radius - margin, radius + margin)
ax.set_aspect('equal')

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.axis('off')

fig.patch.set_facecolor("#FFFFFF")
fig.tight_layout(pad=0, w_pad=0, h_pad=0)


# ============================================================
# Draw directed transition arrows
# ============================================================

# Store every arrow so its appearance can be updated later.
arrows = {}

for i in range(n_states):
    for j in range(n_states):

        if i != j and transition_matrix[i, j] > 0:

            start = positions[i]
            end = positions[j]

            arrow = ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="lightgray",
                    alpha=0.35,
                    linewidth=1.2,
                    shrinkA=35,
                    shrinkB=35,
                    connectionstyle="arc3,rad=0.08"
                ),
                zorder=0
            )

            arrows[(i, j)] = arrow


# ============================================================
# State nodes
# ============================================================

initial_node_size = 100

nodes = ax.scatter(
    positions[:, 0],
    positions[:, 1],
    s=np.full(n_states, initial_node_size),
    color="#EEEEEE",
    edgecolors="#555555",
    linewidths=1.5,
    zorder=1
)


# ============================================================
# Current-state indicator
# ============================================================

head = ax.scatter(
    [],
    [],
    s=initial_node_size,
    color='#058743',
    zorder=4,
    edgecolors='white',
    linewidths=1.5
)


# ============================================================
# Transition count storage
# ============================================================

transition_counts = np.zeros(
    (n_states, n_states),
    dtype=int
)


# ============================================================
# Animation
# ============================================================

def init():

    # Reset transition counts
    transition_counts.fill(0)

    # Reset all arrows
    for (i, j), arrow in arrows.items():

        arrow.arrow_patch.set_color("lightgray")
        arrow.arrow_patch.set_alpha(0.35)
        arrow.arrow_patch.set_linewidth(1.2)

    # Reset nodes
    nodes.set_sizes(
        np.full(n_states, initial_node_size)
    )

    # Hide current-state indicator
    head.set_offsets(
        np.empty((0, 2))
    )

    return tuple(arrows.values()) + (nodes, head)


def update(frame):

    n = frame + 1

    # --------------------------------------------------------
    # Count state visits
    # --------------------------------------------------------

    visit_counts = np.bincount(
        state_sequence[:n + 1],
        minlength=n_states
    )

    # --------------------------------------------------------
    # Scale node sizes according to visit counts
    # --------------------------------------------------------

    min_node_size = 100
    max_node_size = 3000

    max_visits = max(visit_counts.max(), 1)

    node_sizes = (
        min_node_size
        + (visit_counts / max_visits)
        * (max_node_size - min_node_size)
    )

    nodes.set_sizes(node_sizes)

    # --------------------------------------------------------
    # Record the most recent transition
    # --------------------------------------------------------

    previous_state = state_sequence[n - 1]
    current_state = state_sequence[n]

    transition_counts[
        previous_state,
        current_state
    ] += 1

    # --------------------------------------------------------
    # Update directed connection appearance
    # --------------------------------------------------------

    trans_counts_copy = transition_counts
    for i in range(0,n_states):
        trans_counts_copy[i,i] = 0
        
    max_transitions = max(
        trans_counts_copy.max(),
        1
    )

    for (i, j), arrow in arrows.items():

        count = transition_counts[i, j]

        if count == 0:

            # Connection has not yet been used
            arrow.arrow_patch.set_color("lightgray")
            arrow.arrow_patch.set_alpha(0.35)
            arrow.arrow_patch.set_linewidth(1.2)

        else:

            # Normalize transition frequency
            intensity = count / max_transitions

            # Green becomes stronger as transition is used more
            arrow.arrow_patch.set_color(
                plt.cm.viridis(
                    intensity
                )
            )

            arrow.arrow_patch.set_linewidth(
                4.0 * intensity
            )

    # --------------------------------------------------------
    # Current-state indicator
    # --------------------------------------------------------

    head.set_sizes([
        node_sizes[current_state]
    ])

    head.set_offsets([
        positions[current_state]
    ])

    return tuple(arrows.values()) + (nodes, head)


ani = FuncAnimation(
    fig,
    update,
    frames=n_steps,
    init_func=init,
    blit=True
)


# ============================================================
# Save as GIF
# ============================================================

ani.save(
    'markov-chain.gif',
    writer=PillowWriter(fps=60),
    dpi=150
)
