from parameters import n, w, positions, sigma_x, sigma_y
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np


def custom_multi_modal(x, y, w, positions, sigma_x, sigma_y):
    total = 0

    # Gaussian Wells
    for i in range(n):
        total += w[i] * np.exp(-(((x - positions[i][0])**2) / (2 * sigma_x[i]**2) +
                                 ((y - positions[i][1])**2) / (2 * sigma_y[i]**2)))
    
    # Sinusoidal Components
    sin_component = 0.2 * np.sin(4 * np.pi * x) * np.cos(3 * np.pi * y)
    
    # Polynomial Trend
    poly_component = 0.1 * (1/5*(x**2) + 4/5*(y**2))

    return -(total + sin_component - poly_component)

def gradient_descent(f, grad_f, initial_point, learning_rate=0.01, max_iters=1000, tolerance=1e-6):
    x, y = initial_point
    history = [(x, y, f(x, y))]
    
    for i in range(max_iters):
        gradient = grad_f(x, y)
        
        x_new = x - learning_rate * gradient[0]
        y_new = y - learning_rate * gradient[1]
        current_val = f(x_new, y_new)
        history.append((x_new, y_new, current_val))
        
        # Check for position convergence
        if np.linalg.norm([x_new - x, y_new - y]) < tolerance:
            print(f'Position change small; converged in {i+1} iterations.')
            break
        
        x, y = x_new, y_new
    
    return (x, y), current_val, history


def gradient_custom_multi_modal(x, y, w, positions, sigma_x, sigma_y):
    df_dx_total = 0.0
    df_dy_total = 0.0

    # Gradient of Gaussian Wells (total)
    for i in range(n):
        exponent = -(((x - positions[i, 0])**2) / (2 * sigma_x[i]**2) +
                     ((y - positions[i, 1])**2) / (2 * sigma_y[i]**2))
        common_factor = w[i] * np.exp(exponent)
        df_dx_total += common_factor * ((x - positions[i, 0]) / (sigma_x[i]**2))
        df_dy_total += common_factor * ((y - positions[i, 1]) / (sigma_y[i]**2))

    # Gradient of Sinusoidal Component
    sin_coeff = 0.2
    df_dx_sin = sin_coeff * 4 * np.pi * np.cos(4 * np.pi * x) * np.cos(3 * np.pi * y)
    df_dy_sin = -sin_coeff * 3 * np.pi * np.sin(4 * np.pi * x) * np.sin(3 * np.pi * y)

    # Gradient of Polynomial Trend
    poly_coeff = 0.1
    df_dx_poly = 1/5 * 2 * poly_coeff * x
    df_dy_poly = 2 * 4/5 * poly_coeff * y

    # Combine gradients with correct signs
    df_dx = df_dx_total + df_dx_sin - df_dx_poly
    df_dy = df_dy_total + df_dy_sin - df_dy_poly

    return np.array([df_dx, df_dy])


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
    opacity=0.9
)

layout = go.Layout(
    title='Complex Multi-Modal Function',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Cost Function Value'
    ),
    autosize=True
)

# Draw the figure
fig = go.Figure(data=[surface], layout=layout)

# Save and display the image
# pio.write_html(fig, file='data/cost_function.html', auto_open=True)

# For linux-based systems, change last line with this line
pio.write_html(fig, file='../data/cost_function.html', auto_open=False)