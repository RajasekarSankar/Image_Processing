import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the thermal diffusivity function for porous media
def thermal_diffusivity(u, alpha=1.0, beta=0.1):
    return alpha * (1 + beta * u**2)

# Define the porous media heat equation
def porous_media_heat_equation(u, x, t):
    alpha_u = thermal_diffusivity(u)
    with tf.GradientTape() as tape:
        tape.watch(x)
        du_dx = tape.gradient(u, x)
    with tf.GradientTape() as tape:
        tape.watch(du_dx)
        du_dx2 = tape.gradient(du_dx, x)
    du_dt = tf.gradients(u, t)[0]
    pde_residual = du_dt - alpha_u * du_dx2
    return pde_residual

# Generate training data
def generate_training_data(num_interior_points, num_boundary_points):
    x_interior = np.random.uniform(0.1, 0.9, num_interior_points)[:, None]
    t_interior = np.random.uniform(0, 1, num_interior_points)[:, None]
    x_boundary = np.array([[0.], [1.]])  # Fixed boundary points
    t_boundary = np.random.uniform(0, 1, (num_boundary_points, 1))
    return x_interior, t_interior, x_boundary, t_boundary

# Define the PINN model architecture
class PINNModel(tf.keras.Model):
    def __init__(self):
        super(PINNModel, self).__init__()
        self.dense_layer_1 = tf.keras.layers.Dense(50, activation='tanh')
        self.dense_layer_2 = tf.keras.layers.Dense(50, activation='tanh')
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x, t = inputs
        x = tf.concat([x, t], axis=1)  # Concatenate x and t as input
        x = self.dense_layer_1(x)
        x = self.dense_layer_2(x)
        return self.output_layer(x)

# Define the loss function for the porous media heat equation
def porous_media_heat_equation_loss(model, x_interior, t_interior, x_boundary, t_boundary):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_interior)
        tape.watch(t_interior)
        u_interior = model([x_interior, t_interior])
        u_boundary = model([x_boundary, t_boundary])

    pde_residual = porous_media_heat_equation(u_interior, x_interior, t_interior)

    boundary_residual_left = u_boundary[0]  # Boundary condition at x=0
    boundary_residual_right = u_boundary[1]  # Boundary condition at x=1

    loss = tf.reduce_mean(tf.square(pde_residual)) + \
           tf.reduce_mean(tf.square(boundary_residual_left)) + \
           tf.reduce_mean(tf.square(boundary_residual_right))

    return loss

# Train the PINN model
def train(model, x_train_interior, t_train_interior, x_train_boundary, t_train_boundary, epochs):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = porous_media_heat_equation_loss(model, x_train_interior, t_train_interior, x_train_boundary, t_train_boundary)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.numpy()}')

# Generate training data
num_interior_points = 1000
num_boundary_points = 2
x_train_interior, t_train_interior, x_train_boundary, t_train_boundary = generate_training_data(num_interior_points, num_boundary_points)

# Convert training data to TensorFlow tensors
x_train_interior_tf = tf.convert_to_tensor(x_train_interior, dtype=tf.float32)
t_train_interior_tf = tf.convert_to_tensor(t_train_interior, dtype=tf.float32)
x_train_boundary_tf = tf.convert_to_tensor(x_train_boundary, dtype=tf.float32)
t_train_boundary_tf = tf.convert_to_tensor(t_train_boundary, dtype=tf.float32)

# Create and train the PINN model
model = PINNModel()
train(model, x_train_interior_tf, t_train_interior_tf, x_train_boundary_tf, t_train_boundary_tf, epochs=1000)

# Generate points for visualization
x_plot = np.linspace(0, 1, 100)[:, None]
t_plot = np.linspace(0, 1, 100)[:, None]
x_mesh, t_mesh = np.meshgrid(x_plot, t_plot)

# Predict temperature using the trained model
temperature_prediction = model([x_mesh.flatten(), t_mesh.flatten()]).numpy().reshape(x_mesh.shape)

# Plot the temperature distribution
plt.figure(figsize=(10, 6))
plt.contourf(x_mesh, t_mesh, temperature_prediction, cmap='coolwarm')
plt.colorbar(label='Temperature')
plt.xlabel('Position (x)')
plt.ylabel('Time (t)')
plt.title('Temperature Distribution in Porous Media')
plt.grid(True)
plt.show()
