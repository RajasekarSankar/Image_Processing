import tensorflow as tf
 
num_domain = 100
minval = 0
maxval = 1
 
def generate_uniform_data(num_domain, minval, maxval):
  data_init = tf.random_uniform_initializer(minval=minval, maxval=maxval, seed=0)
  return tf.Variable(data_init(shape=[num_domain, 1]), dtype=tf.float32)
 
x = generate_uniform_data(num_domain, minval, maxval)
x
 
def jacobian(tape, y, x, i=0, j=0):
  if y.shape[1] > 1:
    y = y[:, i : i + 1]
 
  grads = tape.gradient(y, x)
  return grads[:, j : j + 1]
 
def pde(tape, x, y):
  dy_x = jacobian(tape, y, x)
  return dy_x - y
 
def ic_0(x):
  return 1
 
x_ic = tf.constant(0, shape=[1,1])
 
n_inputs = 1
n_outputs = 1
activation = 'tanh'
 
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input((n_inputs,)))
model.add(tf.keras.layers.Dense(units=32, activation=activation))
model.add(tf.keras.layers.Dense(units=32, activation=activation))
model.add(tf.keras.layers.Dense(units=n_outputs))
 
 
epochs = 1000
learning_rate = 0.005
 
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
 
for i in range(epochs + 1):
  with tf.GradientTape() as tape_model:
   with tf.GradientTape() as tape_pde:
    y = model(x, training=True)
    y_ic = model(x_ic, training=True)
   du_dx = tape_pde.gradient(y,x)
   loss = tf.reduce_mean((du_dx-y)**2)
   ic_error = tf.reduce_mean((y_ic - ic_0(x_ic))**2)
  
   total_mse = loss + ic_error
 
   if i % 100 == 0:
     print('Epoch: {}\t MSE Loss = {}'.format(i, total_mse))
 
   model_update_gradients = tape_model.gradient(total_mse, model.trainable_variables)
   optimizer.apply_gradients(
    zip(model_update_gradients, model.trainable_variables)
  )
 
import numpy as np
import matplotlib.pyplot as plt
 
x_test = np.linspace(0, 2, 100)
y_true = np.exp(x_test)
y_pred = model(x_test)
 
plt.plot(y_true)
plt.plot(y_pred)
plt.title('Evaluation')
plt.legend(['Real', 'Predicted'])
plt.show()