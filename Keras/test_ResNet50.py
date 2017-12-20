# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:53:54 2017

@author: Administrator
"""

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'F:\\Gdownloadpaper\\2\\1 (11).jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]





from keras.models import Model
# 模型只输出 res4c_branch2b 层的特征
feature = Model(inputs=model.input, outputs = model.get_layer('res4c_branch2b').output)
# 获取 res4c_branch2b 层得到的特征
feat = feature.predict(x)

from Show_array import show_img
feat1 = feat[0,:,:,0] # 获取第一层特征
show_img(feat1)  # 显示第一层特征

