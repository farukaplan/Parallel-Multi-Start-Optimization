from main import custom_multi_modal, gradient_descent, gradient_custom_multi_modal
from parameters import n, w, positions, sigma_x, sigma_y
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import random
import time
import sys

def multi_start_sequential(f, grad_f, num_starts, learning_rate, max_iters, tolerance):
    best_position = None
    best_fitness = np.inf
    history_all = []
    start_time = time.time()

    for i in range(num_starts):
        x0 = random.uniform(-5, 5)
        y0 = random.uniform(-5, 5)
        initial_point = (x0, y0)

        # Run gradient descent from the starting point
        position, fitness, history = gradient_descent(
            f=f,
            grad_f=grad_f,
            initial_point=initial_point,
            learning_rate=learning_rate,
            max_iters=max_iters,
            tolerance=tolerance
        )

        history_all.append(history)

        # Update the best found solution
        if fitness < best_fitness:
            best_fitness = fitness
            best_position = position

    end_time = time.time()
    total_time = end_time - start_time

    return best_position, best_fitness, history_all, total_time

# Read the total number of starts from command line arguments
if len(sys.argv) > 1:
    num_starts = int(sys.argv[1])
else:
    num_starts = 128

best_position_seq, best_fitness_seq, history_seq, time_seq = multi_start_sequential(
    f=lambda x, y: custom_multi_modal(x, y, w, positions, sigma_x, sigma_y),
    grad_f=lambda x, y: gradient_custom_multi_modal(x, y, w, positions, sigma_x, sigma_y),
    num_starts=num_starts,
    learning_rate=0.05,
    max_iters=1000,
    tolerance=1e-6
)

print(f'Sequential Multi-Start Optimization Time: {time_seq:.4f} seconds')
print(f'Best position (Sequential): {best_position_seq}')
print(f'Best fitness (Sequential): {best_fitness_seq}')

# Create a grid for visualization
x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x, y)
Z = custom_multi_modal(X, Y, w, positions, sigma_x, sigma_y)

# Plotting with Plotly for Interactivity
surface = go.Surface(
    x=X,
    y=Y,
    z=Z,
    colorscale='Viridis',
    showscale=True,
    opacity=0.8 
)

layout = go.Layout(
    title='Sequential Multi-Start Optimization Paths',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='f(X, Y)'
    ),
    autosize=True,
    showlegend=True
)

# Initialize the figure with the surface plot
fig = go.Figure(data=[surface], layout=layout)

# Prepare path traces
path_traces = []
for idx, history in enumerate(history_seq):
    history = np.array(history)
    x_path = history[:, 0]
    y_path = history[:, 1]
    z_path = history[:, 2]

    path_trace = go.Scatter3d(
        x=x_path,
        y=y_path,
        z=z_path,
        mode='lines+markers',
        name=f'Path {idx+1}',
        line=dict(
            width=4
        ),
        marker=dict(
            size=3,
            symbol='circle'
        )
    )
    path_traces.append(path_trace)

# Add the path traces to the figure
for trace in path_traces:
    fig.add_trace(trace)

# Save and display the image
# pio.write_html(fig, file='data/multi_sequential_gd.html', auto_open=True)

# For Linux based systems, change last line with this line
pio.write_html(fig, file='../data/multi_sequential_gd.html', auto_open=True)