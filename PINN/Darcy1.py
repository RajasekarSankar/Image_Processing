import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the PINN model architecture
class PorousMediaPINN(tf.keras.Model):
    def __init__(self):
        super(PorousMediaPINN, self).__init__()
        self.dense_layer_1 = tf.keras.layers.Dense(50, activation='tanh')
        self.dense_layer_2 = tf.keras.layers.Dense(50, activation='tanh')
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.dense_layer_1(x)
        x = self.dense_layer_2(x)
        return self.output_layer(x)

# Define the loss function incorporating the PDE and boundary conditions
def porous_media_loss(model, x_interior, x_boundary, epsilon, rho, mu, K):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_interior)
        tape.watch(x_boundary)
        p_interior = model(x_interior)
        p_boundary = model(x_boundary)
        dp_dx = tape.gradient(p_interior, x_interior)
        dp2_dx2 = tape.gradient(dp_dx, x_interior)
        dp_dx_tensor_shape = tf.shape(tf.transpose(dp2_dx2))
        print('tens:',dp_dx_tensor_shape)

    # Define the expressions for alpha and beta
    alpha = 1.75 / tf.sqrt(150 * epsilon**3)
    beta = rho * epsilon * alpha / tf.sqrt(K)
    
    eye_matrix = tf.eye(1)
    eye_matrix_reshaped = tf.tile(tf.expand_dims(eye_matrix, axis=0), [1000, 1, 1])
    identity_matrix_broadcasted = tf.broadcast_to(eye_matrix, (1000, 2, 2))



    # Define the PDE residuals
    pde_residual_1 = (1 / epsilon) * rho * tf.matmul(dp_dx, tf.transpose(dp_dx))
    pde_residual_2 = tf.linalg.trace(-p_interior + (mu / epsilon) * (dp2_dx2 + tf.transpose(dp2_dx2))) - beta * tf.norm(p_interior) * tf.norm(dp_dx)
    pde_residual = tf.reduce_mean(tf.square(pde_residual_1 - pde_residual_2))

    # Define the divergence-free condition residual
    div_residual = tf.reduce_mean(tf.square(tf.math.reduce_sum(dp_dx, axis=1)))

    # Define the boundary condition residuals (for example)
    boundary_residual_left = p_boundary[0] - 0.0
    boundary_residual_right = p_boundary[1] - 1.0

    # Compute the total loss
    loss = pde_residual + div_residual + tf.square(boundary_residual_left) + tf.square(boundary_residual_right)

    return loss

# Generate training data
num_interior_points = 1000
num_boundary_points = 2
x_interior = np.random.uniform(0.1, 0.9, (num_interior_points, 2))  # Assuming 2D domain
x_boundary = np.array([[0., 0.], [1., 1.]])  # Example boundary points
epsilon_value = 0.1
rho_value = 1.0
mu_value = 1.0
K_value = 1.0

# Convert training data to TensorFlow tensors
x_interior_tf = tf.constant(x_interior, dtype=tf.float32)
x_boundary_tf = tf.constant(x_boundary, dtype=tf.float32)

# Create and train the PINN model
model = PorousMediaPINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
epochs = 1000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = porous_media_loss(model, x_interior_tf, x_boundary_tf, epsilon_value, rho_value, mu_value, K_value)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')

# Visualization (if needed)
# Generate points for visualization
x_vis = np.linspace(0, 1, 100)  # Adjust the range as needed
y_vis = np.linspace(0, 1, 100)  # Adjust the range as needed
x_vis_grid, y_vis_grid = np.meshgrid(x_vis, y_vis)
xy_vis = np.column_stack((x_vis_grid.flatten(), y_vis_grid.flatten()))

# Convert visualization points to TensorFlow tensor
xy_vis_tf = tf.constant(xy_vis, dtype=tf.float32)

# Predict pressure using the trained model
pressure_prediction = model(xy_vis_tf).numpy().reshape(x_vis_grid.shape)

# Plot the pressure distribution
plt.figure(figsize=(8, 6))
contour = plt.contourf(x_vis_grid, y_vis_grid, pressure_prediction, cmap='viridis')
plt.colorbar(contour, label='Pressure')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Pressure Distribution')
plt.grid(True)
plt.show()

