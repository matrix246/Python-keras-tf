# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:46:18 2017

@author: Administrator
"""

"""
这里实现了 参考文献中的损失函数
sum_ij(xita_hat_ij - t_ij)^2

 t_ij = λ + (1 - λ) * xita  y_i == y_j
       -1                   otherwise
 
使用到的工具，包括， Lambda 层
自己定义一个层，用来计算损失函数的值，以及计算精度
      
ref:Geometry-aware Deep Transform;Jiaji Huang Qiang Qiu Robert Calderbank Guillermo Sapiro
"""

# define a layer to compute the angle between features
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Dropout, MaxPooling2D
from keras.models import Model
from keras.layers import Lambda, Flatten,Layer
import keras

def tensor_Cos(a):
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
    #angle = tf.acos(a_cos)
    return a_cos

def compute_t_ij(inputs):
    a_cos = inputs[0]
    y_true = inputs[1]
    lambda_value = inputs[2]    
    lambda_value = lambda_value[0]
    # 这里 y_true 是真实类别矩阵，每一行是一个onehot类型向量
    xita_index = tf.matmul(y_true,y_true,transpose_b = True)
    mins_index = 1 - xita_index
    t_ij = (lambda_value + (1-lambda_value)*a_cos)*xita_index - mins_index
    return t_ij
# 利用层定义自己的损失函数
# Custom loss layer  
class CustomVariationalLayer(Layer):  
    def __init__(self, **kwargs):  
        self.is_placeholder = True  
        super(CustomVariationalLayer, self).__init__(**kwargs)  
  
    def mins_loss(self, xita_hat, tij):
        return K.sum(tf.pow(xita_hat - tij,2) )
      
    def call(self, inputs):  
        xita_hat = inputs[0]  
        tij = inputs[1]  
        loss = self.mins_loss(xita_hat, tij)  
        self.add_loss(loss, inputs=inputs)  
        return loss

# 自己定义一个层，用来计算精度,binary_crossentropy
class Compute_accuracy(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(Compute_accuracy, self).__init__(**kwargs)  
        
    def call(self, inputs):  
        y_true = inputs[0]  
        y_pred = inputs[1]  
        accuracy = keras.metrics.binary_crossentropy(y_true, y_pred)  
        self.add_loss(accuracy, inputs=inputs)  
        return accuracy

n = 4
num_class = 5


x_input = Input(shape=(n,n,3))
x = Dense(3,activation='relu')(x_input)
x = Dropout(0.5)(x)
x = Dense(3,activation='relu')(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
y_pred = Dense(num_class,activation='softmax')(x)

# using Lambda to compute the angle between two vectors 
xita_hat = Lambda(tensor_Cos)(y_pred)


y_true = Input(shape=(num_class,))

lambda_value = Input(shape = (1,))


xita = Lambda(tensor_Cos)(x_input)
tij = Lambda(compute_t_ij)([xita, y_true,lambda_value])

# 这个是计算损失的
myloss = CustomVariationalLayer()([xita_hat, tij])
myaccu = Compute_accuracy()([y_true, y_pred])


vae = Model(inputs =[x_input,y_true,lambda_value], 
            outputs=[myloss, myaccu, y_pred] )


vae.compile(optimizer='rmsprop', loss = None)
 

num = 10
x_input = np.ones((num,n,n,3))
y_true = np.random.randint(1,num_class,size=(num,1))
y_true = keras.utils.to_categorical(y_true, num_classes= num_class)

# 由于使用优化方法的时候，每次使用 mini batch 时，都会需要 λ，
# 因此，需要输入一个与所有训练样本数目一致的长度向量的 λ 
lambda_value = 0.4*np.ones((num,1))

vae.fit([x_input,y_true,lambda_value] ,epochs=1)
yhat = vae.predict([x_input,y_true,lambda_value] )
print(y_true)
print(yhat)




