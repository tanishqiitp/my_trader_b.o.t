from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
import ufl
from mpi4py import MPI

# 1. Create a very simple mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)

# 2. Define Function Space
V = fem.functionspace(domain, ("Lagrange", 1))

# 3. Define the problem (u=1 on boundary, -div(grad(u))=0 inside)
uD = fem.Constant(domain, 1.0)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
bc = fem.dirichletbc(uD, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, 0.0)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# 4. Solve the problem
problem = LinearProblem(a, L, bcs=[bc])
uh = problem.solve()

# 5. Save the result
with XDMFFile(domain.comm, "minimal_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

print("Minimal test finished. Output saved to minimal_solution.xdmf")