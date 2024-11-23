from mpi4py import MPI

comm = MPI.COMM_WORLD
print(f"Hello from rank {comm.Get_rank()} out of {comm.Get_size()}")