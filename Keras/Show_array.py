# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:32:03 2017

@author: Administrator
"""
# 定义一个函数，可以显示某一层 的输出或者是权重
import numpy as np

import matplotlib.pyplot as plt

def show_img(x):
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
    plt.imshow(x)
    return x2