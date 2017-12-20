# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:16:04 2017

@author: Administrator
"""
'''
这个文件是根据 keras 手册中对于 函数式模型 的例子得到的
手册中没有数据，这里，自己随机构造了符合模型的数据，测试模型
'''

import keras
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model


# 解释手册中的这句话
# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
''' 
 输入的句子长度为 100（每个句子里面有100个单词）
 所有的单词都是从一个 10000 大小的词典中取出来的
 因此，一个句子可以用 100 数字表示，每个数字表示单词在字典中的位置
'''
main_input = Input(shape=(100, ), dtype= 'int32', name = 'main_input')

# 解释 Embedding 层的作用
'''
假设我们的词典中共包含 10000 个单词，
则，输入的每个句子中的一个单词可以用一个 1 * 10000 的 one-hot 向量表示
则，输入的每个句子的维度是 100 * 10000 
这样的话，输入数据的维度过大，

因此，我们想找一个 大小为 10000 * 512 的 词向量矩阵 M 
将每个单词对应的 1 * 10000 的 one-hot 向量 与 M 相乘
得到新的 1 * 512 大小的向量，来表示原来的单词

这样，每个句子的维度变成了 100 * 512 
1、减小了输入数据的维度
2、两个单词之间可以用新的 1 * 512 维的向量之间的夹角进行度量
3、提高了运算效率


问题：
    1、矩阵 M 如何获得？
        M 对应 Embedding 层的权重，通过训练获得 M 或者，其他方式
    2、手册中的 eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]] 如何理解
        这里只看  4 -> [0.25, 0.1]
        与前面的解释对应理解
        4
        -> 1 * 10000 的 one-hot 向量 x ( x_4 = 1, 其他为 0 ) 
        -> x * M = [0.25, 0.1] 这里假设 M 是 10000 * 2 大小的词向量
'''

x = Embedding(input_dim= 10000, output_dim= 512,
              input_length= 100)(main_input)

#  LSTM 层的有 32 个 units
# 注意这里 LSTM 的作用
# 将原来大小为 100 * 512 的每个句子 转换为 32 * 1 的特征向量 
lstm_out = LSTM(32)(x)

auxiliary_output = Dense(1,activation= 'sigmoid', name= 'aux_output')(lstm_out)

auxiliary_input = Input(shape=(5,), name= 'aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_output])

x = Dense(64, activation= 'relu')(x)
x = Dense(64, activation= 'relu')(x)
x = Dense(64, activation= 'relu')(x)

main_output = Dense(1, activation='sigmoid', name = 'main_output')(x)

model = Model(inputs=[main_input, auxiliary_input],
              outputs= [main_output, auxiliary_output])

model.compile(optimizer= 'rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'],
              loss_weights=[1., 0.2])

'''
 由于手册中没有给出 对应这个模型的训练集，
 因此，这里我们自己设计数据集合
 构造这个模型需要的输入与输出

 dictionary_size = 10000,  这里的意思是考虑最常见的 10000 个单词，
 与我们最初要求 词典大小为 10000 对应 

 sentence_len=100,  这里与我们最开始要求的每个句子 长为 100 对应

 构造训练样本与测试样本数据
 由于网络最后一层都使用了sigmoid 函数，因此网络输出是 0 到 1 之间的小数
 因此，在构造数据的时候，数据的标签也应该是 属于 [0,1] 之间的小数
'''
import numpy as np
num_train = 30
sentence_len = 100
dictionary_size = 10000
x_train = np.random.randint(1,high = dictionary_size,size=(num_train,sentence_len))
y_train_zan = np.random.random((num_train,1))
y_train_zhuan = np.random.random( (num_train,1))


num_test = 20
x_test = np.random.randint(1,high = dictionary_size,size=(num_test,sentence_len))
y_test_zan = np.random.random( (num_test,1))
y_test_zhuan = np.random.random((num_test,1))

# 额外的输入，既是 日期
date_train = np.random.randint(1,high = 9,size=(num_train,5))
date_test = np.random.randint(1,high = 9,size=(num_test,5))

model.fit([x_train,date_train],
          [y_train_zhuan, y_train_zan])

model.predict([x_test,date_test],batch_size=10)

