# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 20:24:59 2017

@author: Administrator
"""
# 效果非常差，这里仅仅是展示一下如何构造模型并应用到实际问题中

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD


from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train= keras.utils.to_categorical(y_train,num_classes=10)
y_test= keras.utils.to_categorical(y_test,num_classes=10)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape = (32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu' ))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu' ))
model.add(Conv2D(64, (3, 3), activation='relu' ))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation= 'softmax'))

#sgd = SGD(lr = 0.01, decay = 1e-6, momentum= 0.9, nesterov=True)
#model.compile(loss = 'categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.compile(loss = 'categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=128, epochs=2)
score = model.evaluate(x_test, y_test, batch_size=256)

print("\n",
      "score is: ",score)
