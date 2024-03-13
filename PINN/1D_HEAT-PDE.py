import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Step 2: Generate training data
def generate_training_data(num_interior_points, num_boundary_points):
    x_interior = np.random.uniform(0.1, 0.9, num_interior_points)[:, None]
    t_interior = np.random.uniform(0.1, 0.9, num_interior_points)[:, None]
    x_boundary = np.array([[0.], [1.]])  # Fixed boundary points
    t_boundary = np.random.uniform(0., 1., num_boundary_points)[:, None]  # Random time for boundary points
    return x_interior, t_interior, x_boundary, t_boundary

# Step 3: Define the PINN Model
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
    
def initial_condition(x):
    return tf.sin(np.pi * x)

# Step 4: Define the Loss Function
def heat_equation_loss(model, x_interior, t_interior, x_boundary, t_boundary):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_interior)
        tape.watch(t_interior)
        u_interior = model([x_interior, t_interior])
        u_boundary = model([x_boundary, t_boundary])
        u_interior_dx2 = tape.gradient(tape.gradient(u_interior, x_interior), x_interior)
    
    # Define the PDE residual
    alpha = 0.1  # Thermal diffusivity
    pde_residual = tf.reduce_mean(tf.square(tape.gradient(u_interior, t_interior) - alpha * u_interior_dx2))

    # Define the boundary condition constraints
    boundary_residual_left = u_boundary[0]  # Boundary condition at x=0
    boundary_residual_right = u_boundary[1]  # Boundary condition at x=1

    # Define the initial condition constraint
    initial_condition_residual = tf.reduce_mean(tf.square(u_interior - initial_condition(x_interior)))
    
    # Compute the loss as a combination of PDE and boundary residuals
    loss = pde_residual + tf.reduce_mean(tf.square(boundary_residual_left)) + tf.reduce_mean(tf.square(boundary_residual_right)) + initial_condition_residual
    
    return loss

# Step 5: Train the Model
def train_model(model, x_train_interior, t_train_interior, x_train_boundary, t_train_boundary, epochs):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = heat_equation_loss(model, x_train_interior, t_train_interior, x_train_boundary, t_train_boundary)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.numpy()}')

# Step 6: Evaluation and Visualization
def plot_temperature_distribution(model):
    x_plot = np.linspace(0, 1, 100)[:, None]
    t_plot = np.linspace(0, 1, 100)[:, None]
    x_mesh, t_mesh = np.meshgrid(x_plot, t_plot)
    x_plot_tf = tf.convert_to_tensor(x_mesh.reshape(-1, 1), dtype=tf.float32)
    t_plot_tf = tf.convert_to_tensor(t_mesh.reshape(-1, 1), dtype=tf.float32)
    temperature_prediction = model([x_plot_tf, t_plot_tf]).numpy().reshape(x_mesh.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(x_mesh, t_mesh, temperature_prediction, cmap='coolwarm')
    plt.colorbar(label='Temperature')
    plt.xlabel('Position (x)')
    plt.ylabel('Time (t)')
    plt.title('Temperature Distribution over Time')
    plt.grid(True)
    plt.show()

# Main
if __name__ == "__main__":
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
    train_model(model, x_train_interior_tf, t_train_interior_tf, x_train_boundary_tf, t_train_boundary_tf, epochs=1000)

    # Plot temperature distribution
    plot_temperature_distribution(model)
