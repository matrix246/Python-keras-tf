# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:56:42 2017

@author: Administrator
"""

import numpy as np

# util function to convert a tensor into a valid image
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
    return x