# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:53:54 2017

@author: Administrator
"""
# 这个实验表明，利用不同的优化方法，对结果影响较大

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD

# 导入数据
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 将数据转化为 4D 的，因为使用 Conv2D 的时候要求数据是 4D的
# 并且，这里将通道放到最后一个维度 （单通道，对应 1）
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train= keras.utils.to_categorical(y_train,num_classes=10)
y_test= keras.utils.to_categorical(y_test,num_classes=10)

model = Sequential()

model.add(Conv2D(4 , (5,5), activation='relu', 
                 input_shape = (28, 28,1), name = 'conv1_1'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(12 , (5,5), activation='relu', name = 'conv2_1'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation= 'softmax'))

# 实验发现， 利用 sgd 进行优化，效果差很多，而利用 Adadelta() 效果则有较大提升
# sgd = SGD(lr = 0.01, decay = 1e-6, momentum= 0.9, nesterov=True)
# model.compile(loss = 'categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# model.compile(loss = 'categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])


# 下面是利用自己定义的损失函数进行训练网络
# 发现，网络的性能与，损失函数的定义也有较大关系，好的损失函数可以得到更好的效果
from keras import backend as K
def my_loss(y_true, y_perd):
    return K.mean((y_perd - y_true), axis= -1)
# 'categorical_crossentropy'
model.compile(loss = 'categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])


# 实验发现，batch_size 对实验精度以及训练速度都会有影响
# batch_size 小，速度 慢，精度 高 
# batch_size 大，速度 块，精度 低
model.fit(x_train, y_train, batch_size=256, epochs=1)
score = model.evaluate(x_test, y_test, batch_size=256)

print("\n",
      "score is: ",score)
'''
将网络结构输出到一个图片中，并保存
'''
from keras.utils import plot_model
plot_model(model, to_file = 'F:\\Gdownloadpaper\\model.png',show_shapes=True)

'''
获取每一层训练好的权重
weights 是一个列表 list 包含每一层的所有权重，每一层的权重是一个  Numpy array 的数据
'''
weights = model.get_weights()

'''
可视化 第一层的 权重
''' 

# 定义一个函数，将每一个滤波器（必须是二维的矩阵）转换为 三维的灰度图像 
import numpy as np
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x = x - x.mean()
    x = x / (x.std() + 1e-5)
    x = x * 0.1

    # clip to [0, 1]
    x = x + 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x = x * 255
    # x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    
    # 利用一个三维数组存放彩色图像，每一层都是相同的灰度值
    x2 = np.zeros((x.shape[0], x.shape[1], 3))
    x2[:,:,0] = x
    x2[:,:,1] = x
    x2[:,:,2] = x
    return x2


layer1_weight = weights[0]
# 第一层的权重是 （5,5,1,4） 类型的数组，将没用的一个维度挤压掉
layer1_weight = np.squeeze(layer1_weight)

# 数组中最后一个元素可以用  arry[-1] 取出
Num_of_filter = layer1_weight.shape[-1]


from scipy.misc import imsave
for i in range(Num_of_filter):
    filter_i = layer1_weight[:,:,i]
    # 将一层滤波变换到 0 ~ 255 的范围，并且生成一个 三层的灰度图，利于显示
    img = deprocess_image(filter_i)
    imsave('F:\\Gdownloadpaper\\layer1_filter_%d.png' % (i), img)


'''
查看原图像经过某一层处理之后的结果
这里是看 conv2_1 之后的结果
'''
'''
需要先利用训练好的网络，对测试数据进行处理
然后，看对第一个图片的处理结果
这里在 conv2_1 之后有 12 张特征图
因此，会出现 12 个结果
'''

from keras.models import Model
# 这里截取原网络的一部分，构造一个新的模型
dense1 = Model(inputs= model.input,
               outputs=model.get_layer('conv2_1').output)
# 利用 训练好的局部网络对测试样本进行处理
dense1_output = dense1.predict(x_test)

# 获取第一个测试样本的处理结果
f1 = dense1_output[0]

for i in range(f1.shape[-1]):
    f1_i = f1[:, :, i]
    img = deprocess_image(f1_i)
    imsave('F:\\Gdownloadpaper\\image1_feature_%d.png' % (i), img)
