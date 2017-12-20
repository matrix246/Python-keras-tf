# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:46:18 2017

@author: Administrator
"""


import tensorflow as tf
import keras.backend as K

import numpy as np
n = 10
a = np.ones((n,n))
b = K.sum(tf.pow(a,2), keepdims=True)




print(b.shape)
print(b)

