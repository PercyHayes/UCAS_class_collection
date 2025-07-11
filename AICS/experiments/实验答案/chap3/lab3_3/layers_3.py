import numpy as np
import struct
import os
import scipy.io
import time

class ContentLossLayer(object):
    def __init__(self):
        print('\tContent loss layer.')
    def forward(self, input_layer, content_layer):
         # TODO： 计算风格迁移图像和目标内容图像的内容损失
        n,c,h,w = input_layer.shape
        loss = np.sum(np.square(input_layer,content_layer))
        loss = loss /(n*c*h*w*2)
        #_____________________________
        return loss
    def backward(self, input_layer, content_layer):
        # TODO： 计算内容损失的反向传播
        n,c,h,w = input_layer.shape
        bottom_diff = (input_layer - content_layer)/(n*c*h*w)
        #_____________________
        return bottom_diff

class StyleLossLayer(object):
    def __init__(self):
        print('\tStyle loss layer.')
    def forward(self, input_layer, style_layer):
        # TODO： 计算风格迁移图像和目标风格图像的Gram 矩阵
        style_layer_reshape = np.reshape(style_layer, [style_layer.shape[0], style_layer.shape[1], -1])
        self.gram_style = np.array([np.matmul(style_layer_reshape[i,:,:], style_layer_reshape[i,:,:].T) for i in range(style_layer.shape[0])])
        #____________________________
        self.input_layer_reshape = np.reshape(input_layer, [input_layer.shape[0], input_layer.shape[1], -1])
        self.gram_input = np.zeros([input_layer.shape[0], input_layer.shape[1], input_layer.shape[1]])
        for idxn in range(input_layer.shape[0]):
            self.gram_input[idxn, :, :] = np.matmul(self.input_layer_reshape[idxn,:,:], self.input_layer_reshape[idxn,:,:].T)
            #___________________        
        M = input_layer.shape[2] * input_layer.shape[3]
        N = input_layer.shape[1]
        self.div = M * M * N * N
        # TODO： 计算风格迁移图像和目标风格图像的风格损失
        style_diff = self.gram_input - self.gram_style
        #______________________
        loss = np.sum(np.square(style_diff))/(4*self.div * style_layer.shape[0])
        #______________________
        return loss
    def backward(self, input_layer, style_layer):
        bottom_diff = np.zeros([input_layer.shape[0], input_layer.shape[1], input_layer.shape[2]*input_layer.shape[3]])
        for idxn in range(input_layer.shape[0]):
            # TODO： 计算风格损失的反向传播
            bottom_diff[idxn, :, :] = np.matmul((self.gram_input[idxn,:,:]-self.gram_style[idxn,:,:]).T,self.input_layer_reshape[idxn,:,:])/(self.div * style_layer.shape[0])
            #__________________________
        bottom_diff = np.reshape(bottom_diff, input_layer.shape)
        return bottom_diff
