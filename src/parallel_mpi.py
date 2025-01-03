from main import custom_multi_modal, gradient_descent, gradient_custom_multi_modal
from parameters import n, w, positions, sigma_x, sigma_y
from mpi4py import MPI
import numpy as np
import random

def multi_start_parallel(f, grad_f, num_starts_per_process, learning_rate, max_iters, tolerance):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_best_position = None
    local_best_fitness = np.inf
    local_histories = []

    # Each process runs gradient descent from multiple starting points
    for _ in range(num_starts_per_process):
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

        local_histories.append(history)

        # Update the local best solution
        if fitness < local_best_fitness:
            local_best_fitness = fitness
            local_best_position = position

    # Find the global minimum fitness via allreduce with MPI.MIN
    global_best_fitness = comm.allreduce(local_best_fitness, op=MPI.MIN)

    # Determine which rank "owns" the global best fitness
    candidate_rank = rank if np.isclose(local_best_fitness, global_best_fitness) else -1

    # Find the maximum of candidate_rank across all ranks
    owner_rank = comm.allreduce(candidate_rank, op=MPI.MAX)

    # Broadcast the best position from the owner rank
    if rank == owner_rank:
        global_best_position = local_best_position
    else:
        global_best_position = None

    global_best_position = comm.bcast(global_best_position, root=owner_rank)

    # For the histories, do an allgather
    gathered_histories = comm.allgather(local_histories)
    if rank == 0:
        flattened_histories = [hist for proc_histories in gathered_histories for hist in proc_histories]
        return global_best_position, global_best_fitness, flattened_histories
    else:
        return None, None, None

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    import sys

    if len(sys.argv) > 1:
        num_starts_total = int(sys.argv[1])
    else:
        num_starts_total = 128

    # Determine the number of starts per process
    num_starts_per_process = num_starts_total // size
    remainder = num_starts_total % size
    if rank < remainder:
        num_starts_per_process += 1

    start_time = MPI.Wtime()

    # Run the parallel multi-start optimization
    best_position, best_fitness, all_histories = multi_start_parallel(
        f=lambda x, y: custom_multi_modal(x, y, w, positions, sigma_x, sigma_y),
        grad_f=lambda x, y: gradient_custom_multi_modal(x, y, w, positions, sigma_x, sigma_y),
        num_starts_per_process=num_starts_per_process,
        learning_rate=0.05,
        max_iters=1000,
        tolerance=1e-6
    )

    end_time = MPI.Wtime()
    total_time = end_time - start_time

    # Only the root process prints the results
    if rank == 0:
        print(f'Parallel Multi-Start Optimization Time: {total_time:.4f} seconds')
        print(f'Best position (Parallel): {best_position}')
        print(f'Best fitness (Parallel): {best_fitness}')

        import plotly.graph_objs as go
        import plotly.io as pio

        # Create a grid for visualization
        x_grid = np.linspace(-5, 5, 500)
        y_grid = np.linspace(-5, 5, 500)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = custom_multi_modal(X, Y, w, positions, sigma_x, sigma_y)

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
            title='Parallel Multi-Start Optimization Paths',
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
        for idx, history in enumerate(all_histories):
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
                line=dict(width=4),
                marker=dict(size=3, symbol='circle')
            )
            path_traces.append(path_trace)

        # Add the path traces to the figure
        for trace in path_traces:
            fig.add_trace(trace)

        # Save the figure to an HTML file
        pio.write_html(fig, file='../data/parallel_optimization_paths.html', auto_open=False)
