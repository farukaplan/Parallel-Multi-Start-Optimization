import numpy as np

# Define parameters for the Gaussian wells
n = 15  # Number of minima

# Define weights (depths)
# The global minimum has the largest weight
w = np.array([4.5, 3.3, 3.2, 2.1, 4.0, 1.9, 3.8, 2.7, 3.6, 0.5, 1.4, 3.3, 0.2, 1.1, 2.05])


# Define positions (mu_x, mu_y)
positions = np.array([
    (-2.5, -2.5),  # Global minimum
    (1, 1),
    (4, -1),
    (-1, 4),
    (-4, -4),
    (-3, -3),
    (0, 0),
    (3.5, 2),
    (-1.5, -4),
    (2, 1.5),
    (-3, -4.5),
    (-0.5, -2),
    (4.5, 3),
    (2, 0.5),
    (-4, -0.5)
])

# Define standard deviations
sigma_x = np.array([0.3, 0.4, 0.35, 0.5, 0.45, 0.4, 0.35, 0.3, 0.5, 0.4, 0.3, 0.35, 0.45, 0.5, 0.4])
sigma_y = np.array([0.35, 0.3, 0.4, 0.45, 0.5, 0.35, 0.4, 0.3, 0.5, 0.35, 0.45, 0.4, 0.5, 0.3, 0.35])