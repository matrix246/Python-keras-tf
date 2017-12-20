# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:56:34 2017

@author: Administrator
"""

from keras.layers import Input, Dense
from keras.models import Model

# 这个返回一个 张量 inputs
# 定义一个输入（Input），将长度为 784 的数列，转化为一个维度为 784 的张量（inputs)
inputs = Input(shape=(784,))


# 输入 张量 inputs， 输出 张量 x
x = Dense(64, activation= 'relu')(inputs)
x = Dense(64, activation= 'relu')(x)
predictions = Dense(10, activation= 'softmax')(x)


model= Model(inputs= inputs, outputs= predictions)
model.compile(optimizer='rmsprop',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

import numpy as np
data = np.random.random((2,784))
labels = np.random.random((2,10)) # 注意，这里标签的维度需要与网络的输出维度一致

model.fit(data , labels)


x = Input(shape=(784,))
y = model(x)



# 将 图像分类模型变为一个对视频分类的模型
from keras.layers import TimeDistributed

input_sequences = Input(shape=(20, 784))

processed_sequences = TimeDistributed(model)(input_sequences)

print(processed_sequences)