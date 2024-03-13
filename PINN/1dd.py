import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_domain = 100
minval = 0
maxval = 2

def generate_uniform_data(num_domain, minval, maxval):
    data_init = tf.random_uniform_initializer(minval=minval, maxval=maxval, seed=0)
    return tf.Variable(data_init(shape=[num_domain, 1]), dtype=tf.float32)

x = generate_uniform_data(num_domain, minval, maxval)

def jacobian(tape, y, x, i=0, j=0):
    if y.shape[1] > 1:
        y = y[:, i : i + 1]
    
    grads = tape.gradient(y, x)
    return grads[:, j : j + 1]

def pde(tape, x, y):
    dy_dx = jacobian(tape, y, x)
    return dy_dx - y

def ic_0(x):
    return tf.exp(x)

def boundary_conditions(model, x_min, x_max):
    return model(x_min), model(x_max)

x_ic = tf.constant([[0]], dtype=tf.float32)
x_min = tf.constant([[minval]], dtype=tf.float32)
x_max = tf.constant([[maxval]], dtype=tf.float32)

n_inputs = 1
n_outputs = 1
activation = 'tanh'

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((n_inputs,)),
    tf.keras.layers.Dense(units=32, activation=activation),
    tf.keras.layers.Dense(units=32, activation=activation),
    tf.keras.layers.Dense(units=n_outputs, activation='linear')
])

epochs = 500
learning_rate = 0.005
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

for i in range(epochs + 1):
    with tf.GradientTape() as tape_model:
        with tf.GradientTape() as tape_pde:
            y = model(x, training=True)
            y_ic = model(x_ic, training=True)
            y_min, y_max = boundary_conditions(model, x_min, x_max)
        dy_dx = tape_pde.gradient(y, x)
        loss = tf.reduce_mean(tf.square(dy_dx - y))
        ic_error = tf.reduce_mean(tf.square(y_ic - ic_0(x_ic)))
        boundary_loss = tf.reduce_mean(tf.square(y_min - ic_0(x_min)) + tf.square(y_max - ic_0(x_max)))
        total_loss = loss + ic_error + boundary_loss
    
    if i % 100 == 0:
        print('Epoch: {}\t Total Loss = {}'.format(i, total_loss))
    
    model_gradients = tape_model.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))

# Evaluation
x_test = np.linspace(minval, maxval, 100)[:, None]
y_true = np.exp(x_test)
y_pred = model(x_test).numpy()

plt.plot(x_test, y_true, label='True')
plt.plot(x_test, y_pred, label='Predicted')
plt.title('Evaluation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
