from parameters import n, w, positions, sigma_x, sigma_y
from main import custom_multi_modal, gradient_descent, gradient_custom_multi_modal
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import random

# Define initial point
x = random.uniform(-5, 5)
y = random.uniform(-5, 5)
initial_point = (x, y)

# Run Gradient Descent
best_position, best_fitness, history = gradient_descent(
    f = lambda x, y: custom_multi_modal(x, y, w, positions, sigma_x, sigma_y),
    grad_f = lambda x, y: gradient_custom_multi_modal(x, y, w, positions, sigma_x, sigma_y),
    initial_point = initial_point,
    learning_rate = 0.001,
    max_iters = 1000,
    tolerance = 1e-6
)

print(f'Best position: {best_position}')
print(f'Best fitness: {best_fitness}')

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
    title='Gradient Descent Optimization Path',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='f(X, Y)'
    ),
    autosize=True
)

# Initialize the figure with the surface plot
fig = go.Figure(data=[surface], layout=layout)

# Extract optimization path
history = np.array(history)
x_path = history[:, 0]
y_path = history[:, 1]
z_path = history[:, 2]

# Create the path trace
path_trace = go.Scatter3d(
    x=x_path,
    y=y_path,
    z=z_path,
    mode='lines+markers',
    name='Optimization Path',
    line=dict(
        width=4,
        color='red'
    ),
    marker=dict(
        size=3,
        symbol='circle',
        color='red'
    )
)

# Add the path trace to the figure
fig.add_trace(path_trace)

# Save and display the image
pio.write_html(fig, file='data/single_start_gd.html', auto_open=True)

# For linux-based systems, change the last line with this line
# pio.write_html(fig, file='../data/single_start_gd.html', auto_open=True)