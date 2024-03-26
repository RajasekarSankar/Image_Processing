import numpy as np
'''
# Parameters
rho = 1.0  # Fluid density
mu = 1.0   # Dynamic viscosity
K = 1.0    # Permeability of the porous medium
epsilon = 0.5  # Porosity
alpha = 1.75 / np.sqrt(150 * epsilon**3)  # Dimensionless friction coefficient


# Define grid parameters
Nx = 100  # Number of grid points in x-direction
Ny = 100  # Number of grid points in y-direction
Lx = 1.0  # Length of the domain in x-direction
Ly = 1.0  # Length of the domain in y-direction
dx = Lx / (Nx - 1)  # Grid spacing in x-direction
dy = Ly / (Ny - 1)  # Grid spacing in y-direction
dt = 0.001  # Time step size

# Initialize velocity field
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))

# Initialize pressure field
p = np.zeros((Nx, Ny))

# Time stepping loop
for t in range(1000):
    # Compute velocity gradients
    du_dx = (u[1:, :] - u[:-1, :]) / dx
    dv_dy = (v[:, 1:] - v[:, :-1]) / dy
    du_dy = (u[:, 1:] - u[:, :-1]) / dy
    dv_dx = (v[1:, :] - v[:-1, :]) / dx
    
    # Compute pressure gradient
    dp_dx = (p[1:, :] - p[:-1, :]) / dx
    dp_dy = (p[:, 1:] - p[:, :-1]) / dy
    
    # Compute Forchheimer term
    Forchheimer_term_x = rho * epsilon * alpha * np.sqrt(K / mu) * np.abs(u) * u
    Forchheimer_term_y = rho * epsilon * alpha * np.sqrt(K / mu) * np.abs(v) * v
    
    # Compute velocity updates
    #u[1:-1, 1:-1] += dt * (1 / rho) * (-dp_dx[1:-1, :-1] + mu * (du_dx[1:, :] - du_dx[:-1, :]) / dx - Forchheimer_term_x[1:-1, :-1])
    u[1:-1, 1:-1] += dt * (1 / rho) * (-(dp_dx[1:-1, :-1] + dp_dx[1:-1, 1:]) + mu / epsilon * (du_dx[1:-1, 1:] - du_dx[1:-1, :-1]) / dx - Forchheimer_term_x[1:-1, 1:-1])
    #u[1:-1, 1:-1] += dt * (1 / rho) * (-dp_dx[1:-1, 1:-1] + mu * (du_dx[1:-1, 1:-1] - du_dx[1:-1, 1:-1]) / dx - Forchheimer_term_x[1:-1, 1:-1])
    v[1:-1, 1:-1] += dt * (1 / rho) * (-dp_dy[:-1, 1:-1] + mu * (dv_dy[:, 1:] - dv_dy[:, :-1]) / dy - Forchheimer_term_y[:-1, 1:-1])
    
    # Apply boundary conditions (e.g., no-slip)
    u[:, 0] = 0
    u[:, -1] = 0
    u[0, :] = 0
    u[-1, :] = 0
    
    v[:, 0] = 0
    v[:, -1] = 0
    v[0, :] = 0
    v[-1, :] = 0
    
    # Update pressure using continuity equation
    p[1:-1, 1:-1] += dt * (-rho * ((du_dx[1:, :] - du_dx[:-1, :]) / dx + (dv_dy[:, 1:] - dv_dy[:, :-1]) / dy))
    
    # Apply boundary conditions for pressure (e.g., zero gradient)
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]
    p[:, 0] = p[:, 1]
    p[:, -1] = p[:, -2]
'''

import numpy as np
import matplotlib.pyplot as plt

# Parameters
epsilon = 1.0
rho = 1.0
mu = 1.0
K = 1.0
alpha = 1.0

# Grid parameters
nx = 100  # Number of grid points in the x-direction
ny = 100  # Number of grid points in the y-direction
dx = 1.0  # Grid spacing in the x-direction
dy = 1.0  # Grid spacing in the y-direction

# Time parameters
dt = 0.01  # Time step size
nt = 100   # Number of time steps

# Initialize velocity field u and pressure field p
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

# Time stepping loop
for t in range(nt):
    # Calculate gradients
    du_dx = np.gradient(u, axis=1) / dx
    dv_dy = np.gradient(v, axis=0) / dy
    dp_dx = np.gradient(p, axis=1) / dx
    dp_dy = np.gradient(p, axis=0) / dy
    #print(Forchheimer_term_x[1:-1, 1:-1].shape)
    #print(dv_dy.shape)
    

    
    # Calculate Forchheimer term
    Forchheimer_term_x = mu / (epsilon * K) * (du_dx + du_dx.T)
    Forchheimer_term_y = mu / (epsilon * K) * (dv_dy + dv_dy.T)
    #print(Forchheimer_term_x[1:-1, 1:-1].shape)
    # Update velocity field
    #u[1:-1, 1:-1] += dt * (1 / rho) * (-(dp_dx[1:-1, 1:-1] + dp_dx[1:-1, 1:-1]) + mu / epsilon * (du_dx[1:-1, 1:-1] - du_dx[1:-1, 1:-1]) / dx - Forchheimer_term_x[1:-1, 1:-1])
    u[1:-1, 1:-1] += dt * (1 / rho) * (-(dp_dx[1:-1, :-1] + dp_dx[1:-1, 1:]) + mu / epsilon * (du_dx[1:-1, 1:] - du_dx[1:-1, :-1]) / dx - Forchheimer_term_x[1:-1, 1:-1])
    v[1:-1, 1:-1] += dt * (1 / rho) * (-(dp_dy[1:-1, 1:-1] + dp_dy[1:-1, 1:-1]) + mu / epsilon * (dv_dy[1:-1, 1:-1] - dv_dy[1:-1, 1:-1]) / dy - Forchheimer_term_y[1:-1, 1:-1])
    #u[1:-1, 1:-1] += dt * (1 / rho) * (-dp_dx[1:-1, :-1] + mu * (du_dx[1:, :] - du_dx[:-1, :]) / dx - Forchheimer_term_x[1:-1, :-1])
    #u[1:-1, 1:-1] += dt * (1 / rho) * (-(dp_dx[1:-1, :-1] + dp_dx[1:-1, 1:]) + mu / epsilon * (du_dx[1:-1, 1:] - du_dx[1:-1, :-1]) / dx - Forchheimer_term_x[1:-1, 1:-1])
    #u[1:-1, 1:-1] += dt * (1 / rho) * (-dp_dx[1:-1, 1:-1] + mu * (du_dx[1:-1, 1:-1] - du_dx[1:-1, 1:-1]) / dx - Forchheimer_term_x[1:-1, 1:-1])
    # Apply boundary conditions (if needed)
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]
    p[:, 0] = p[:, 1]
    p[:, -1] = p[:, -2]
    #print(du_dx[1:-1, 1:-1].shape)
    # Update pressure field (if needed)
    p[1:-1, 1:-1] += dt * (-rho * ((du_dx[1:-1, 1:-1] - du_dx[1:-1, 1:-1]) / dx + (dv_dy[1:-1, 1:-1] - dv_dy[1:-1, 1:-1]) / dy))
# Visualization
# Plot velocity field
X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
plt.figure()
plt.quiver(X, Y, u, v, scale=20)
plt.title('Velocity Field (u, v)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
