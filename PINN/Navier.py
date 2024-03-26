import tensorflow as tf
import numpy as np

# Define the PINN model architecture
class NavierStokesPINN(tf.keras.Model):
    def __init__(self):
        super(NavierStokesPINN, self).__init__()
        self.dense_layer_1 = tf.keras.layers.Dense(50, activation='tanh')
        self.dense_layer_2 = tf.keras.layers.Dense(50, activation='tanh')
        self.output_layer_u = tf.keras.layers.Dense(1)  # Output for velocity u
        self.output_layer_v = tf.keras.layers.Dense(1)  # Output for velocity v
        self.output_layer_p = tf.keras.layers.Dense(1)  # Output for pressure

    def call(self, x):
        x = self.dense_layer_1(x)
        x = self.dense_layer_2(x)
        u = self.output_layer_u(x)
        v = self.output_layer_v(x)
        p = self.output_layer_p(x)
        return u, v, p

# Define the loss function incorporating the Navier-Stokes equations and boundary conditions
def navier_stokes_loss(model, x_interior, x_boundary, rho, nu):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_interior)
        tape.watch(x_boundary)
        u_interior, v_interior, p_interior = model(x_interior)
        u_boundary, v_boundary, p_boundary = model(x_boundary)
        du_dx = tape.gradient(u_interior, x_interior)
        dv_dy = tape.gradient(v_interior, x_interior)
        dp_dx = tape.gradient(p_interior, x_interior)
        dp_dy = tape.gradient(p_interior, x_interior)

    # Define the Navier-Stokes residuals
    pde_residual_u = du_dx + dv_dy
    pde_residual_v = du_dx + dv_dy
    pde_residual_p = dp_dx + dp_dy

    # Define the boundary condition residuals (for example)
    boundary_residual_u = u_boundary - 0.0
    boundary_residual_v = v_boundary - 0.0
    boundary_residual_p = p_boundary - 0.0

    # Compute the total loss
    loss = (tf.reduce_mean(tf.square(pde_residual_u)) +
            tf.reduce_mean(tf.square(pde_residual_v)) +
            tf.reduce_mean(tf.square(pde_residual_p)) +
            tf.reduce_mean(tf.square(boundary_residual_u)) +
            tf.reduce_mean(tf.square(boundary_residual_v)) +
            tf.reduce_mean(tf.square(boundary_residual_p)))

    return loss

# Generate training data (adjust as needed)
num_interior_points = 1000
num_boundary_points = 2
x_interior = np.random.uniform(0.1, 0.9, (num_interior_points, 2))  # Assuming 2D domain
x_boundary = np.array([[0., 0.], [1., 1.]])  # Example boundary points
rho_value = 1.0  # Fluid density
nu_value = 1.0   # Kinematic viscosity

# Convert training data to TensorFlow tensors
x_interior_tf = tf.constant(x_interior, dtype=tf.float32)
x_boundary_tf = tf.constant(x_boundary, dtype=tf.float32)

# Create and train the PINN model
model = NavierStokesPINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
epochs = 1000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = navier_stokes_loss(model, x_interior_tf, x_boundary_tf, rho_value, nu_value)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')

import matplotlib.pyplot as plt

# Generate points for visualization
x_vis = np.linspace(0, 1, 100)  # Adjust the range as needed
y_vis = np.linspace(0, 1, 100)  # Adjust the range as needed
x_vis_grid, y_vis_grid = np.meshgrid(x_vis, y_vis)
xy_vis = np.column_stack((x_vis_grid.flatten(), y_vis_grid.flatten()))

# Convert visualization points to TensorFlow tensor
xy_vis_tf = tf.constant(xy_vis, dtype=tf.float32)

# Predict velocity and pressure fields using the trained model
u_pred, v_pred, p_pred = model(xy_vis_tf)

# Reshape predictions to grid for plotting
u_pred_grid = u_pred.numpy().reshape(x_vis_grid.shape)
v_pred_grid = v_pred.numpy().reshape(y_vis_grid.shape)
p_pred_grid = p_pred.numpy().reshape(x_vis_grid.shape)

# Plot velocity field
plt.figure(figsize=(10, 6))
plt.quiver(x_vis_grid, y_vis_grid, u_pred_grid, v_pred_grid, scale=20)
plt.title('Velocity Field (u, v)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Plot pressure field
plt.figure(figsize=(10, 6))
contour = plt.contourf(x_vis_grid, y_vis_grid, p_pred_grid, cmap='viridis')
plt.colorbar(contour, label='Pressure')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Pressure Distribution')
plt.show()
