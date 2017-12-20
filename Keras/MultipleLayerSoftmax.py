# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:53:54 2017

@author: Administrator
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import keras as keras

import numpy as np
x_train = np.random.random((1000,20))
y_train = keras.utils.to_categorical(
        np.random.randint(10, size=(1000,1)), num_classes = 10)
x_test = np.random.random((10,20))
y_test = keras.utils.to_categorical(
        np.random.randint(10, size=(10,1)), num_classes = 10)


model = Sequential()

model.add(Dense(64,activation='relu', input_dim = 20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', ))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr = 0.01, momentum=0.9, decay=1e-6, nesterov=True)
model.compile(loss = 'categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=10,
          batch_size= 128)

score = model.evaluate(x_test, y_test, batch_size=3)
print("            \n")
print("score is: ", score)

print("            \n")
# 类别数目 是从 0开始 到 9
y_hat = model.predict_classes(x_test, batch_size=3)
print("\n y_hat is: ", y_hat)

print("            \n")
print("y_test is: \n ", y_test)


y_hat2 = model.predict(x_test, batch_size=3)
print("            \n")
print("y_hat2 is: \n ", y_hat2)
