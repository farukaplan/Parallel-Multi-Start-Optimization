import numpy as np
import time
from numba import njit, prange
from parameters import n, w, positions, sigma_x, sigma_y

def custom_multi_modal_array(x, y, w, positions, sigma_x, sigma_y):
    total = np.zeros_like(x)
    for i in range(n):
        exponent = -(((x - positions[i, 0]) ** 2) / (2.0 * sigma_x[i] ** 2) +
                     ((y - positions[i, 1]) ** 2) / (2.0 * sigma_y[i] ** 2))
        total += w[i] * np.exp(exponent)
    # Sinusoidal Components
    sin_component = 0.2 * np.sin(4.0 * np.pi * x) * np.cos(3.0 * np.pi * y)
    # Polynomial Trend
    poly_component = 0.1 * (0.2 * x**2 + 0.8 * y**2)
    return -(total + sin_component - poly_component)

@njit
def custom_multi_modal(x, y, w, positions, sigma_x, sigma_y):
    total = 0.0
    for i in range(n):
        exponent = -(((x - positions[i, 0]) ** 2) / (2.0 * sigma_x[i] ** 2) +
                     ((y - positions[i, 1]) ** 2) / (2.0 * sigma_y[i] ** 2))
        total += w[i] * np.exp(exponent)
    # Sinusoidal Components
    sin_component = 0.2 * np.sin(4.0 * np.pi * x) * np.cos(3.0 * np.pi * y)
    # Polynomial Trend
    poly_component = 0.1 * (0.2 * x**2 + 0.8 * y**2)
    return -(total + sin_component - poly_component)

@njit
def gradient_custom_multi_modal(x, y, w, positions, sigma_x, sigma_y):
    df_dx_total = 0.0
    df_dy_total = 0.0
    for i in range(n):
        exponent = -(((x - positions[i, 0]) ** 2) / (2.0 * sigma_x[i] ** 2) +
                     ((y - positions[i, 1]) ** 2) / (2.0 * sigma_y[i] ** 2))
        common_factor = w[i] * np.exp(exponent)
        df_dx_total += common_factor * ((x - positions[i, 0]) / (sigma_x[i] ** 2))
        df_dy_total += common_factor * ((y - positions[i, 1]) / (sigma_y[i] ** 2))

    # Gradient of Sinusoidal Component
    sin_coeff = 0.2
    df_dx_sin = sin_coeff * 4.0 * np.pi * np.cos(4.0 * np.pi * x) * np.cos(3.0 * np.pi * y)
    df_dy_sin = -sin_coeff * 3.0 * np.pi * np.sin(4.0 * np.pi * x) * np.sin(3.0 * np.pi * y)

    # Gradient of Polynomial Trend
    poly_coeff = 0.1
    df_dx_poly = 2.0 * poly_coeff * 0.2 * x
    df_dy_poly = 2.0 * poly_coeff * 0.8 * y

    # Combine gradients with correct signs
    df_dx = df_dx_total + df_dx_sin - df_dx_poly
    df_dy = df_dy_total + df_dy_sin - df_dy_poly

    return np.array([df_dx, df_dy])

@njit
def gradient_descent(f, grad_f, initial_point, w, positions, sigma_x, sigma_y, learning_rate=0.01, max_iters=1000, tolerance=1e-6):
    x, y = initial_point
    history = np.zeros((max_iters + 1, 3))
    history[0, :] = np.array([x, y, f(x, y, w, positions, sigma_x, sigma_y)])
    for i in range(1, max_iters + 1):
        gradient = grad_f(x, y, w, positions, sigma_x, sigma_y)
        x_new = x - learning_rate * gradient[0]
        y_new = y - learning_rate * gradient[1]
        current_val = f(x_new, y_new, w, positions, sigma_x, sigma_y)
        history[i, :] = np.array([x_new, y_new, current_val])
        # Check for convergence
        if np.sqrt((x_new - x)**2 + (y_new - y)**2) < tolerance:
            history = history[:i+1, :]
            break
        x, y = x_new, y_new
    return (x, y), current_val, history

def generate_random_points(num_points):
    x_points = np.random.uniform(-5, 5, num_points)
    y_points = np.random.uniform(-5, 5, num_points)
    return x_points, y_points

@njit(parallel=True)
def multi_start_parallel(x_starts, y_starts, w, positions, sigma_x, sigma_y, learning_rate, max_iters, tolerance):
    num_starts = x_starts.shape[0]
    best_positions = np.zeros((num_starts, 2))
    best_fitnesses = np.zeros(num_starts)
    histories = [np.zeros((max_iters + 1, 3)) for _ in range(num_starts)]

    for i in prange(num_starts):
        initial_point = (x_starts[i], y_starts[i])

        # Run gradient descent from the starting point
        position, fitness, history = gradient_descent(
            custom_multi_modal,
            gradient_custom_multi_modal,
            initial_point,
            w,
            positions,
            sigma_x,
            sigma_y,
            learning_rate,
            max_iters,
            tolerance
        )

        best_positions[i, :] = position
        best_fitnesses[i] = fitness
        histories[i][:history.shape[0], :] = history[:]

    # Find the overall best position
    min_idx = np.argmin(best_fitnesses)
    best_position = best_positions[min_idx, :]
    best_fitness = best_fitnesses[min_idx]
    return best_position, best_fitness, histories

if __name__ == "__main__":
    import sys

    # Read the total number of starts from command line arguments
    if len(sys.argv) > 1:
        num_starts_total = int(sys.argv[1])
    else:
        num_starts_total = 75  # Default value

    # Set random seed for reproducibility
    #np.random.seed(42)

    # Generate random starting points
    x_starts, y_starts = generate_random_points(num_starts_total)

    # Start timing
    start_time = time.time()

    # Run the parallel multi-start optimization
    best_position, best_fitness, histories = multi_start_parallel(
        x_starts,
        y_starts,
        w,
        positions,
        sigma_x,
        sigma_y,
        learning_rate=0.05,
        max_iters=1000,
        tolerance=1e-6
    )

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    # Print the results
    print(f'Parallel Multi-Start Optimization Time: {total_time:.4f} seconds')
    print(f'Best position (Parallel): {best_position}')
    print(f'Best fitness (Parallel): {best_fitness}')

    # Visualization (Optional)
    import plotly.graph_objs as go
    import plotly.io as pio

    # Create a grid for visualization
    x_grid = np.linspace(-5, 5, 500)
    y_grid = np.linspace(-5, 5, 500)
    X, Y = np.meshgrid(x_grid, y_grid)
    # Compute Z using the non-JIT function
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
        title='Parallel Multi-Start Optimization Paths (Numba)',
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
    for idx, history in enumerate(histories):
        # Trim zeros from preallocated history
        history = history[~np.all(history == 0, axis=1)]
        x_path = history[:, 0]
        y_path = history[:, 1]
        z_path = history[:, 2]

        path_trace = go.Scatter3d(
            x=x_path,
            y=y_path,
            z=z_path,
            mode='lines+markers',
            name=f'Path {idx+1}',
            line=dict(width=4),
            marker=dict(size=3)
        )
        fig.add_trace(path_trace)

    # Save the figure
    pio.write_html(fig, file='../data/numba_parallel_optimization_paths.html', auto_open=True)
