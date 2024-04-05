# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:56:48 2024

@author: sank_ra
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize
from matplotlib.gridspec import GridSpec

num_train_samples=1000
num_test_samples=1000
#kinematic viscoscity
nu= 0.01/np.pi

class Network:
  @classmethod
  def build(cls, num_inputs=2,layers=[16,32,64],activation='tanh', num_outputs=1):
    #input layer.
    inputs= tf.keras.layers.Input(shape=(num_inputs,))
    #hidden layers
    x=inputs
    for layer in layers:
      x=tf.keras.layers.Dense(layer, activation=activation)(x)
    #output layer
    outputs=tf.keras.layers.Dense(num_outputs)(x)
    return tf.keras.models.Model(inputs=inputs,outputs=outputs)


network=Network.build()
network.summary()

class GradientLayer(tf.keras.layers.Layer):
  def __init__(self,model,**kwargs):
    self.model=model
    super().__init__(**kwargs)
  def call(self,x):
    with tf.GradientTape() as g:
      g.watch(x) 
      with tf.GradientTape() as gg: 
        gg.watch(x)
        u = self.model(x)
      du_dtx = gg.batch_jacobian(u, x) 
      du_dt = du_dtx[..., 0]  # du/dt du/dx
      du_dx = du_dtx[..., 1]
    d2u_dx2 = g.batch_jacobian(du_dx, x)[..., 1]#du/dt d2u/dx2
    return u, du_dt, du_dx, d2u_dx2


class PINN:
  def __init__(self, network, nu):
        self.network = network
        self.nu = nu
        self.grads = GradientLayer(self.network)
  def build(self):
 
        # equation input: (t, x)
        tx_eqn = tf.keras.layers.Input(shape=(2,))
        # initial condition input: (t=0, x)
        tx_ini = tf.keras.layers.Input(shape=(2,))
        # boundary condition input: (t, x=-1) or (t, x=+1)
        tx_bnd = tf.keras.layers.Input(shape=(2,))

        # compute gradients. we use the gradient class previously built.
        u, du_dt, du_dx, d2u_dx2 = self.grads(tx_eqn)

        # equation output being zero
        u_eqn = du_dt + u*du_dx - (self.nu)*d2u_dx2
        # initial condition output for t =0, x=x
        u_ini = self.network(tx_ini) #passing thru the inner network i.e. hidden layers.
        # boundary condition output for x=-1 and x=1 and t=t
        u_bnd = self.network(tx_bnd)

        return tf.keras.models.Model(inputs=[tx_eqn, tx_ini, tx_bnd], outputs=[u_eqn, u_ini, u_bnd])
    

pinn=PINN(network,nu).build()
pinn.summary()

#creating training input
tx_eqn=np.random.rand(num_train_samples,2) #t_eqn = 0 to 1
tx_eqn[...,1]=2*tx_eqn[...,1]-1 #x_eqn -1 to 1
tx_ini=2*np.random.rand(num_train_samples,2)-1 #x_ini -1 to 1
tx_ini[...,0]=0 #t_ini=0
tx_bnd=np.random.rand(num_train_samples,2) #t_bnd 0 to 1
tx_bnd[...,1]=2*np.round(tx_bnd[...,1])-1 #x_bnd = -1 or +1

#creating training output
u_eqn=np.zeros((num_train_samples,1)) #u_eqn=0
u_ini = np.sin(-np.pi * tx_ini[..., 1, np.newaxis])    # u_ini = -sin(pi*x_ini)
u_bnd = np.zeros((num_train_samples, 1))               # u_bnd = 0


#L-BFGS OPTIMIZATION
class L_BFGS_B:


    def __init__(self, model, x_train, y_train, factr=1e7, m=50, maxls=50, maxiter=5000):
 
        # set attributes
        self.model = model
        self.x_train = [ tf.constant(x, dtype=tf.float32) for x in x_train ]
        self.y_train = [ tf.constant(y, dtype=tf.float32) for y in y_train ]
        self.factr = factr
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.metrics = ['loss']
        self.progbar = tf.keras.callbacks.ProgbarLogger(
            count_mode='steps', stateful_metrics=self.metrics)
        self.progbar.set_params( {
            'verbose':1, 'epochs':1, 'steps':self.maxiter, 'metrics':self.metrics})

    def set_weights(self, flat_weights):
    

        shapes = [ w.shape for w in self.model.get_weights() ]
        split_ids = np.cumsum([ np.prod(shape) for shape in [0] + shapes ])
        weights = [ flat_weights[from_id:to_id].reshape(shape)
            for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes) ]
        self.model.set_weights(weights)

    @tf.function
    def tf_evaluate(self, x, y):

        with tf.GradientTape() as g:
            loss = tf.reduce_mean(tf.keras.losses.mse(self.model(x), y))
        grads = g.gradient(loss, self.model.trainable_variables)
        return loss, grads

    def evaluate(self, weights):

        self.set_weights(weights)
        loss, grads = self.tf_evaluate(self.x_train, self.y_train)
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([ g.numpy().flatten() for g in grads ]).astype('float64')

        return loss, grads

    def callback(self, weights):
  
        self.progbar.on_batch_begin(0)
        loss, _ = self.evaluate(weights)
        self.progbar.on_batch_end(0, logs=dict(zip(self.metrics, [loss])))

    def fit(self):
    
        initial_weights = np.concatenate(
            [ w.flatten() for w in self.model.get_weights() ])
        print('Optimizer: L-BFGS-B (maxiter={})'.format(self.maxiter))
        self.progbar.on_train_begin()
        self.progbar.on_epoch_begin(1)
        scipy.optimize.fmin_l_bfgs_b(func=self.evaluate, x0=initial_weights,
            factr=self.factr, m=self.m, maxls=self.maxls, maxiter=self.maxiter,
            callback=self.callback)
        self.progbar.on_epoch_end(1)
        self.progbar.on_train_end()

x_train=[tx_eqn,tx_ini,tx_bnd]#training input
y_train=[u_eqn,u_ini,u_bnd]#training output
lbfgs= L_BFGS_B(model=pinn,x_train=x_train,y_train=y_train)
lbfgs.fit()


# predict u(t,x) distribution
t_flat = np.linspace(0, 1, num_test_samples)
x_flat = np.linspace(-1, 1, num_test_samples)
t, x = np.meshgrid(t_flat, x_flat)
tx = np.stack([t.flatten(), x.flatten()], axis=-1)
u = network.predict(tx, batch_size=num_test_samples)
u = u.reshape(t.shape)

fig = plt.figure(figsize=(7,4))
gs = GridSpec(2, 5)
plt.subplot(gs[0, :])
plt.pcolormesh(t, x, u, cmap='rainbow')
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
cbar.set_label('u(t,x)')
cbar.mappable.set_clim(-1, 1)
# plot u(t=const, x) cross-sections
t_cross_sections = [0,0.25, 0.5,0.75,1]
for i, t_cs in enumerate(t_cross_sections):
  plt.subplot(gs[1, i])
  tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
  u = network.predict(tx, batch_size=num_test_samples)
  plt.plot(x_flat, u)
  plt.title('t={}'.format(t_cs))
  plt.xlabel('x')
  plt.ylabel('u(t,x)')
plt.tight_layout()
plt.show()
