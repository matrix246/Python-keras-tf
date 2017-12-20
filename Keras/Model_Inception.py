# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:51:16 2017

@author: Administrator
"""

# 这里只是展示了 Inception 的构造方法，
# 将图片分别进行处理，然后将处理的结果进行叠加，作为新的特征继续进行处理
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate, Dense,Flatten

input_img = Input(shape=(256,256,3))

tower_1 = Conv2D(64, (1,1), padding= 'same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3,3), padding= 'same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1,1), padding= 'same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5,5), padding= 'same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3,3), strides=(1,1), padding= 'same')(input_img)
tower_3 = Conv2D(64, (1,1), padding= 'same', activation='relu')(tower_3)

output = concatenate([tower_1, tower_2, tower_3], axis = -1)

output = Flatten()(output)
output = Dense(2,activation='softmax')(output)


model = Model(inputs= input_img, outputs= output)


import numpy as np

img = np.random.random((256,256,3))
img = np.expand_dims(img,0)


y_hat = model.predict(img)

# 编译模型
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])
# 训练模型
model.fit(img,y_hat)

# 保存模型
model.save('F:\\Gdownloadpaper\\model.h5')
