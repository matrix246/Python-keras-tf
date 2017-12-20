# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:13:50 2017

@author: Administrator
"""

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights = 'imagenet', include_top=False)

img_path= 'F:\\Gdownloadpaper\\elephant.jpg'
img = image.load_img(img_path, target_size=(224,224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)

features = model.predict(x)
