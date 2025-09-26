import dolfinx
from mpi4py import MPI

print("✅ FEniCSx version:", dolfinx.__version__)
print("✅ MPI world size:", MPI.COMM_WORLD.size)
