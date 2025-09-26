from fenics import *
import matplotlib.pyplot as plt

# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)   # 32x32 grid on unit square
V = FunctionSpace(mesh, 'P', 1) # linear elements

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution
plot(u)
plt.title("Poisson equation solution")
plt.colorbar(plot(u))
plt.show()

# Save result to file for ParaView
vtkfile = File('poisson_solution.pvd')
vtkfile << u

# Compute error against exact solution
u_exact = interpolate(u_D, V)
error = errornorm(u_exact, u, 'L2')
print("L2 error =", error)
