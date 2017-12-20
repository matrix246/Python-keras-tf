# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 20:25:02 2017

@author: Administrator
"""
# 声明一个全局变量，用于控制图片窗口，防止后一个被前一个覆盖
WINDOW_ID = 0

#from PIL import image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

cat = mpimg.imread('F:\\Gdownloadpaper\\cat.jpg')
plt.figure(num= 0)
plt.imshow(cat)

# 创建一个一层的模型
from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
# 滤波核的尺寸,以及数目
kernel_size = 3
layer1_filter_num = 3
model.add(Conv2D(layer1_filter_num,
                 (kernel_size , kernel_size),
                 input_shape = cat.shape))

import numpy as np
# 在 cat 的第一个维度之前插入一个维度，因为 Conv2D 需要一个 4 维的输入
cat_batch = np.expand_dims(cat, axis = 0) 
# 利用 model 对 cat_batch 进行处理
conv_cat = model.predict(cat_batch)

def visualze_3_channel_array(cat_batch):
    global WINDOW_ID
    WINDOW_ID +=  1
    cat = np.squeeze(cat_batch)
    print(cat.shape)
    plt.figure(num= WINDOW_ID)
    plt.imshow(cat)
    
def visualze_1_channel_array(cat_batch):
    global WINDOW_ID    
    WINDOW_ID +=  1
    cat = cat_batch[0]
    print(cat.shape)
    plt.figure(num= WINDOW_ID)
    plt.imshow(cat)
    
# 显示滤波之后的 猫
visualze_3_channel_array(conv_cat)

visualze_1_channel_array(conv_cat)


# 获取第一层的三个滤波器（每一个滤波器都是 3*3*3 大小的数组）
layer1_filters = model.get_weights()[0]
# 显示滤波器
for i in range(layer1_filter_num):
    filter_i = layer1_filters[:,:,:,i]
    visualze_3_channel_array(filter_i)

# 显示最后一个滤波器中的元素的数值
for i in range(3):
    filter_i_value = filter_i[:,:,i]
    print('\n 第三个滤波器的 第 %d 层是：\n' %(i) )
    print(filter_i_value)

'''
再添加一个 relu 层，然后看效果
'''
from keras.layers import Activation
model.add(Activation('relu'))
conv_cat2 = model.predict(cat_batch)
# 显示滤波,+ relu 之后的 猫
visualze_3_channel_array(conv_cat2)
visualze_1_channel_array(conv_cat)


