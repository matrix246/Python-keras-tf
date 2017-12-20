# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:46:18 2017

@author: Administrator
"""

# define my own layer to compute loss function
from keras import backend as K
from keras.engine.topology import Layer
class my_layer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(my_layer, self).__init__(**kwargs)
    
    
    def vae_loss(self, x, x_decoded):
        x = K.flatten(x)
        x_decoded = K.flatten(x_decoded)
        loss = K.mean(K.abs(x - x_decoded))
        return loss
    
    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        loss = self.vae_loss(x, x_decoded)
        
        # this sentence is a must
        self.add_loss(loss, inputs = inputs)
        return loss

n = 4
# construct the model which use the self defined layer 
from keras.layers import Input,Dense
x_ten = Input(shape=(n,n))
x_d_ten = Input(shape=(n,n))

x_ten1 = Dense(32,activation='relu')(x_ten)
x_d_ten1 = Dense(32,activation='relu')(x_d_ten)

y  = my_layer()([x_ten1 , x_d_ten1])
#print(y.get_shape())
from keras.models import Model
vae = Model(inputs =[x_ten,x_d_ten], outputs=y)
# notice : loss =  None
vae.compile(optimizer='rmsprop', loss = None) 


import numpy as np
x = np.ones((1,n,n))
x_decoded = np.zeros((1,n,n))

#vae.fit([x, x_decoded], 4)
yhat = vae.predict([x, x_decoded])

print(yhat)