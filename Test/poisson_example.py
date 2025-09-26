import numpy as np
import pyvista
import ufl
from dolfinx import fem, mesh, plot
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem

# --- Visualization Function ---
def create_plot(solution_function, filename):
    print("--- Entering visualization function ---")
    try:
        pyvista.start_xvfb()
        plotter = pyvista.Plotter(off_screen=True)
        
        V = solution_function.function_space
        topology, cell_types, geometry = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data["u"] = solution_function.x.array
        
        plotter.add_mesh(grid, show_edges=True)
        plotter.view_xy()
        plotter.screenshot(filename)
        print(f"--- Plot successfully saved to {filename} ---")
    except Exception as e:
        print(f"--- An error occurred during visualization: {e} ---")

# --- Main part of the script ---
print("Script started.")
domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
V = fem.functionspace(domain, ("Lagrange", 1))

# --- Boundary Conditions ---
def bottom(x):
    return np.isclose(x[1], 0)
def top(x):
    return np.isclose(x[1], 1)
bottom_dofs = fem.locate_dofs_geometrical(V, bottom)
top_dofs = fem.locate_dofs_geometrical(V, top)
bc_bottom = fem.dirichletbc(fem.Constant(domain, 0.0), bottom_dofs, V)
bc_top = fem.dirichletbc(fem.Constant(domain, 1.0), top_dofs, V)
bcs = [bc_bottom, bc_top]

# --- Variational Problem ---
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)
# Make the internal heat source 50 times stronger
# Correct code
f = 500 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# --- Solve ---
problem = LinearProblem(a, L, bcs=bcs)
uh = problem.solve()
print("Poisson problem solved.")

# --- Call the visualization function ---
create_plot(uh, "poisson_solution.png")

print("Script finished.")