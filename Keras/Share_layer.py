# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:05:47 2017

@author: Administrator
"""

# 测试共享层

import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

tweet_a = Input(shape=(140, 256))
tweet_b = Input(shape=(140, 256))

# 定义一个 LSTM 层，并重命名为“shared_lstm”，并没有指定输入输出
shared_lstm = LSTM(64)
# 调用 “shared_lstm” 层，输入分别为 tweet_a 和 tweet_b
# 输出分别为 encoded_a 和 encoded_b
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis = -1)

predictions = Dense(1, activation='sigmoid')(merged_vector)

model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# 构造数据，测试网络
import numpy as np
x_test = np.random.random((140, 256))
x_test = np.expand_dims(x_test, 0)
y_hat = model.predict([x_test,x_test])
print(y_hat)

# 理解层节点的概念
# 这里 shared_lstm 调用了两次，则 其有两个节点
# 获取第一个节点的输出 
print(shared_lstm.get_output_at(0) )
