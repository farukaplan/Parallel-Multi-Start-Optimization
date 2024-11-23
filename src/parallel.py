from mpi4py import MPI
import numpy as np
import random
import time

# mpi_multi_start.py
def multi_start_parallel(f, grad_f, num_starts_per_process, learning_rate, max_iters, tolerance):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_best_position = None
    local_best_fitness = np.inf
    local_history = []
    start_time = MPI.Wtime()

    for i in range(num_starts_per_process):
        # Each process generates its own starting points
        x0 = random.uniform(0, 5)
        y0 = random.uniform(0, 5)
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

        local_history.append(history)

        # Update the local best solution
        if fitness < local_best_fitness:
            local_best_fitness = fitness
            local_best_position = position

    # Gather all local best positions and fitnesses at root
    all_best_positions = comm.gather(local_best_position, root=0)
    all_best_fitnesses = comm.gather(local_best_fitness, root=0)

    end_time = MPI.Wtime()
    total_time = end_time - start_time

    # Only the root process will find the overall best
    if rank == 0:
        best_index = np.argmin(all_best_fitnesses)
        best_position = all_best_positions[best_index]
        best_fitness = all_best_fitnesses[best_index]
        return best_position, best_fitness, local_history, total_time
    else:
        return None, None, local_history, total_time
