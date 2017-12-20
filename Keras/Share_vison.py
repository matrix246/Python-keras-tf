# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:54:03 2017

@author: Administrator
"""

# 这里展示了 手册中的 共享视觉模型
# 这里 简历了一个模型，然后利用这个模型两次，分别得到输出，
# 将两个模型的输出作为 新模型的输入，最后判断两个数字是否是同一个数字

from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model

image_size = 28

digit_input = Input(shape=(image_size,image_size,1))
x = Conv2D(64, (3,3),activation='relu')(digit_input)
x = MaxPooling2D((3,3),strides=(2,2))(x)
x = Conv2D(32, (3,3),activation='relu')(x)
x = MaxPooling2D((2,2))(x)
out = Flatten()(x)

vision_model = Model(digit_input, out)

digit_a = Input(shape=  (image_size, image_size, 1))
digit_b = Input(shape=  (image_size, image_size, 1))

out_a = vision_model(digit_a)
out_b = vision_model(digit_b)

import keras
concatenated = keras.layers.concatenate([out_a, out_b])
out = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model([digit_a, digit_b], out)

# 利用 mnist 数据集制作数据，来训练这个网络，并进行测试

import numpy as np
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)


num_test = len(x_test)

train_index = np.arange(x_train.shape[0])
np.random.shuffle(train_index)

# 将训练数据重排，做成对比数据
x_train_b = x_train[train_index]

# 1 同一个数字  0 不同的数字
label_train =  y_train[train_index] != y_train
label_train = 1 - label_train

test_index = np.arange(x_test.shape[0])
np.random.shuffle(test_index)

x_test_b = x_test[test_index]

label_test = y_test[test_index] != y_test
label_test = 1 - label_test

# 训练模型
classification_model.compile(optimizer='rmsprop',
                             loss = 'binary_crossentropy',
                             metrics=['accuracy'])

classification_model.fit([x_train, x_train_b], label_train,batch_size=256)
History = classification_model.evaluate([x_test, x_test_b], label_test,batch_size=256)
print(History)


# 这里 将所有测测试样本对都搞成一样的，结果测试结果精度为 0 
# 可能是训练数据中 相同的样本对 数目太少导致
# 也可能是模型本身不好，导致  （这个可能性更大）
label = np.ones((len(label_test) , 1))
History = classification_model.evaluate([x_test, x_test], label,batch_size=256)
print(History)


