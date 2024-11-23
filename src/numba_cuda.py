# cuda_multi_start.py

import numpy as np
import time
from numba import cuda, float64
import sys

# Define parameters for the Gaussian wells
n = 7  # Total number of minima (1 global + 6 local)

w = np.array([1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], dtype=np.float64)
positions = np.array([
    (2.5, 2.5),
    (1, 1),
    (4, 1),
    (1, 4),
    (4, 4),
    (3, 3),
    (2, 4)
], dtype=np.float64)
sigma_x = np.array([0.5] * n, dtype=np.float64)
sigma_y = np.array([0.5] * n, dtype=np.float64)

# Transfer data to device memory
d_w = cuda.to_device(w)
d_positions = cuda.to_device(positions)
d_sigma_x = cuda.to_device(sigma_x)
d_sigma_y = cuda.to_device(sigma_y)

@cuda.jit(device=True)
def custom_multi_modal_device(x, y, w, positions, sigma_x, sigma_y, n):
    total = 0.0
    for i in range(n):
        exponent = -(((x - positions[i, 0]) ** 2) / (2 * sigma_x[i] ** 2) +
                     ((y - positions[i, 1]) ** 2) / (2 * sigma_y[i] ** 2))
        total += w[i] * cuda.math.exp(exponent)
    return -total

@cuda.jit(device=True)
def gradient_custom_multi_modal_device(x, y, grad, w, positions, sigma_x, sigma_y, n):
    df_dx = 0.0
    df_dy = 0.0
    for i in range(n):
        exponent = -(((x - positions[i, 0]) ** 2) / (2 * sigma_x[i] ** 2) +
                     ((y - positions[i, 1]) ** 2) / (2 * sigma_y[i] ** 2))
        common_factor = w[i] * cuda.math.exp(exponent)
        df_dx += common_factor * ((x - positions[i, 0]) / (sigma_x[i] ** 2))
        df_dy += common_factor * ((y - positions[i, 1]) / (sigma_y[i] ** 2))
    grad[0] = df_dx
    grad[1] = df_dy

@cuda.jit
def gradient_descent_kernel(x0_array, y0_array, best_positions, best_fitnesses, histories, w, positions, sigma_x, sigma_y, n, learning_rate, max_iters, tolerance):
    idx = cuda.grid(1)
    if idx < x0_array.size:
        x = x0_array[idx]
        y = y0_array[idx]
        grad = cuda.local.array(2, dtype=float64)
        history = cuda.local.array(3000, dtype=float64)  # Adjust size as needed

        f_val = custom_multi_modal_device(x, y, w, positions, sigma_x, sigma_y, n)
        history_idx = 0
        history[history_idx * 3 + 0] = x
        history[history_idx * 3 + 1] = y
        history[history_idx * 3 + 2] = f_val
        history_idx += 1

        for i in range(max_iters):
            gradient_custom_multi_modal_device(x, y, grad, w, positions, sigma_x, sigma_y, n)
            x_new = x - learning_rate * grad[0]
            y_new = y - learning_rate * grad[1]
            f_val = custom_multi_modal_device(x_new, y_new, w, positions, sigma_x, sigma_y, n)

            history[history_idx * 3 + 0] = x_new
            history[history_idx * 3 + 1] = y_new
            history[history_idx * 3 + 2] = f_val
            history_idx += 1

            # Check for convergence
            if cuda.math.sqrt((x_new - x) ** 2 + (y_new - y) ** 2) < tolerance:
                break

            x, y = x_new, y_new

        best_positions[idx, 0] = x
        best_positions[idx, 1] = y
        best_fitnesses[idx] = f_val

        # Store history (up to history_idx entries)
        for j in range(history_idx):
            histories[idx, j, 0] = history[j * 3 + 0]
            histories[idx, j, 1] = history[j * 3 + 1]
            histories[idx, j, 2] = history[j * 3 + 2]

        # Record the length of the history
        histories[idx, 0, 3] = history_idx  # Use an extra slot to store history length

def generate_random_points(num_points):
    x_points = np.random.uniform(0, 5, num_points).astype(np.float64)
    y_points = np.random.uniform(0, 5, num_points).astype(np.float64)
    return x_points, y_points

def custom_multi_modal_array(x, y, w, positions, sigma_x, sigma_y):
    total = np.zeros_like(x)
    for i in range(n):
        exponent = -(((x - positions[i, 0]) ** 2) / (2 * sigma_x[i] ** 2) +
                     ((y - positions[i, 1]) ** 2) / (2 * sigma_y[i] ** 2))
        total += w[i] * np.exp(exponent)
    return -total

if __name__ == "__main__":
    import sys

    # Read the total number of starts from command line arguments
    if len(sys.argv) > 1:
        num_starts_total = int(sys.argv[1])
    else:
        num_starts_total = 10  # Default value

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate random starting points
    x_starts, y_starts = generate_random_points(num_starts_total)

    # Device arrays for starting points
    d_x_starts = cuda.to_device(x_starts)
    d_y_starts = cuda.to_device(y_starts)

    # Allocate device arrays for results
    best_positions = np.zeros((num_starts_total, 2), dtype=np.float64)
    best_fitnesses = np.zeros(num_starts_total, dtype=np.float64)
    histories = np.zeros((num_starts_total, 1000, 4), dtype=np.float64)  # Adjust sizes as needed

    d_best_positions = cuda.to_device(best_positions)
    d_best_fitnesses = cuda.to_device(best_fitnesses)
    d_histories = cuda.to_device(histories)

    # CUDA kernel parameters
    threads_per_block = 128
    blocks_per_grid = (num_starts_total + (threads_per_block - 1)) // threads_per_block

    # Start timing
    start_time = time.time()

    # Run the kernel
    gradient_descent_kernel[blocks_per_grid, threads_per_block](
        d_x_starts,
        d_y_starts,
        d_best_positions,
        d_best_fitnesses,
        d_histories,
        d_w,
        d_positions,
        d_sigma_x,
        d_sigma_y,
        n,
        0.05,  # learning_rate
        1000,  # max_iters
        1e-6   # tolerance
    )

    # Copy results back to host
    d_best_positions.copy_to_host(best_positions)
    d_best_fitnesses.copy_to_host(best_fitnesses)
    d_histories.copy_to_host(histories)

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    # Find the overall best position
    min_idx = np.argmin(best_fitnesses)
    best_position = best_positions[min_idx, :]
    best_fitness = best_fitnesses[min_idx]

    print(f'CUDA Multi-Start Optimization Time: {total_time:.4f} seconds')
    print(f'Best position (CUDA): {best_position}')
    print(f'Best fitness (CUDA): {best_fitness}')

    # Visualization (Optional)
    import plotly.graph_objs as go
    import plotly.io as pio

    # Create a grid for visualization
    x_grid = np.linspace(0, 5, 200)
    y_grid = np.linspace(0, 5, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = custom_multi_modal_array(X, Y, w, positions, sigma_x, sigma_y)

    # Plotting with Plotly
    surface = go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        showscale=True,
        opacity=0.8
    )

    layout = go.Layout(
        title='CUDA Multi-Start Optimization Paths',
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

    # Extract the best history
    best_history = histories[min_idx]

    # Trim the history based on the stored length
    history_length = int(best_history[0, 3])
    best_history = best_history[:history_length]

    x_path = best_history[:, 0]
    y_path = best_history[:, 1]
    z_path = best_history[:, 2]

    # Add the best path trace
    path_trace = go.Scatter3d(
        x=x_path,
        y=y_path,
        z=z_path,
        mode='lines+markers',
        name='Best Optimization Path',
        line=dict(width=4, color='red'),
        marker=dict(size=3, color='red')
    )
    fig.add_trace(path_trace)

    # Save the figure
    pio.write_html(fig, file='cuda_optimization_paths.html', auto_open=True)
