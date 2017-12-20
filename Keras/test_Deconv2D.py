# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:16:51 2017

@author: Administrator
"""

# 测试 Deconv2D 


# define a layer to compute the angle between features
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.layers import Input , Dense, Dropout, MaxPooling2D
from keras.models import Model

def tensor_Angle(a):
    # 输入是一个张量，shape=（num,n,n,...）
    '''
    先将张量转化为矩阵  num * cols
    然后求，任意两行的内积 a_prod
    然后求，每一行的二范数 a_norm
    然后求，任意两行的余弦 a_cos
    然后求，任意两行的角度 angle 
    最后得到的是一个 num*num 的矩阵 angle
    '''
    # reshape tensor into a matrix, each row is a feature vector
    a = K.reshape(a , [K.shape(a)[0],-1])
    # inner prodcut of vectors 
    a_prod = tf.matmul(a,a,transpose_b = True)
    # compute the norm or each feature vector
    a_norm = K.sqrt(K.clip(K.sum(tf.pow(a,2), axis = 1, keepdims=True), 
                  K.epsilon(),None))
    # norm product of each vector
    a_norm = tf.matmul(a_norm, a_norm, transpose_b = True)
    
    a_cos = a_prod/K.clip(a_norm, K.epsilon(),None)
    a_cos = K.clip(a_cos,-1,1)
    angle = tf.acos(a_cos)
    return angle
    
    
n = 4
x_input = Input(shape=(n,n,3))
x = Dense(3,activation='relu')(x_input)
x = Dropout(0.5)(x)
x = MaxPooling2D((3,3), padding = 'same')(x)

# using Lambda to compute the angle between two vectors 
from keras.layers import Lambda 
y = Lambda(tensor_Angle)(x)


vae = Model(inputs =x_input, outputs=y)
vae.compile(optimizer='rmsprop', loss = 'binary_crossentropy')
 
num = 3
x = np.ones((num,n,n,3))
y = np.ones((num, num))
vae.fit(x , y)
yhat = vae.predict(x)

print(yhat)