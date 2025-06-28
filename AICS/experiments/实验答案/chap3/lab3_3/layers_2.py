import numpy as np
import struct
import os
import time

def show_matrix(mat, name):
    #print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))
    pass

def show_time(time, name):
    #print(name + str(time))
    pass

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride, type=0):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw
        if type == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (self.kernel_size, self.channel_in, self.channel_out))
    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])
        show_matrix(self.weight, 'conv weight ')
        show_matrix(self.bias, 'conv bias ')
    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO: 计算卷积层的前向传播，特征图与卷积核的内积再加偏置
                        h_start = idxh * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = idxw * self.stride
                        w_end = w_start + self.kernel_size
                        self.output[idxn, idxc, idxh, idxw] = np.sum(self.input_pad[idxn, :, h_start:h_end, w_start:w_end] * self.weight[:, :, :, idxc]) + self.bias[idxc]
                       
                        #self.output[idxn, idxc, idxh, idxw] = ____________________________
        self.forward_time = time.time() - start_time
        return self.output
    def forward_speedup(self, input):
        # TODO: 改进forward函数，使得计算加速
        start_time = time.time()
        #_______________________ added

        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1

        mat_w = self.kernel_size * self.kernel_size * self.channel_in
        mat_h = height_out * width_out
        # image2col优化计算速度
        self.img_col = np.zeros([self.input.shape[0],mat_h,mat_w])




        #self.img_col = np.zeros([self.input.shape[0], height_out*width_out, self.input.shape[1]*(self.kernel_size**2)])
        for i in range(height_out):
            for j in range(width_out):
                self.img_col[:,i*width_out+j,:] = self.input_pad[:,:,i*self.stride:i*self.stride+self.kernel_size,j*self.stride:j*self.stride+self.kernel_size].reshape(self.input.shape[0],-1) # 一列

        weight_col = self.weight.reshape(-1, self.channel_out)

        self.output_col = np.matmul(self.img_col.reshape(-1,self.img_col.shape[-1]),weight_col)
        self.output = self.output_col.reshape(self.input.shape[0], height_out, width_out, self.channel_out) + self.bias
        self.output = np.transpose(self.output,[0,3,1,2])
        self.forward_time = time.time() - start_time
        
        return self.output
    def backward_speedup(self, top_diff):
        start_time = time.time()

        self.d_bias = np.sum(top_diff, axis=(0,2,3))
        
        
        #_______________________
        
        height_out = (self.input.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        width_out = (self.input.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1

        # cout, batch, h, w
        top_diff_col = np.transpose(top_diff, [1, 0, 2, 3]).reshape(top_diff.shape[1], -1)
        #  batch, (h * w), (cin * k * k)

        tmp = np.transpose(self.img_col.reshape(-1, self.img_col.shape[-1]), [1, 0])
        self.d_weight = np.matmul(tmp, top_diff_col.T).reshape(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out)
        self.d_bias = top_diff_col.sum(axis=1)
        
        backward_col = np.empty((top_diff.shape[0], self.input.shape[2] * self.input.shape[3], self.kernel_size * self.kernel_size * self.channel_out))
        pad_height = ((self.input.shape[2] - 1) * self.stride + self.kernel_size - height_out) // 2
        pad_width = ((self.input.shape[3] - 1) * self.stride + self.kernel_size - width_out) // 2
        top_diff_pad = np.zeros((top_diff.shape[0], top_diff.shape[1], height_out + 2 * pad_height, width_out + 2 * pad_width))
        top_diff_pad[:, :, pad_height: height_out + pad_height, pad_width: width_out + pad_width] = top_diff
        cur = 0
        for x in range(self.input.shape[2]):
            for y in range(self.input.shape[3]):
                bias_x = x * self.stride
                bias_y = y * self.stride
                backward_col[:, cur, :] = top_diff_pad[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size].reshape(top_diff.shape[0], -1)
                cur = cur + 1

        weight_reshape = np.transpose(self.weight, [3, 1, 2, 0]).reshape(self.channel_out, -1, self.channel_in)[:, ::-1, :].reshape(-1, self.channel_in)
        bottom_diff = np.matmul(backward_col, weight_reshape)
        bottom_diff = np.transpose(bottom_diff.reshape(top_diff.shape[0], self.input.shape[2], self.input.shape[3], self.input.shape[1]), [0, 3, 1, 2])
        

        self.backward_time = time.time() - start_time
        return bottom_diff

    def backward_raw(self, top_diff):
        start_time = time.time()
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO： 计算卷积层的反向传播， 权重、偏置的梯度和本层损失
                        self.d_weight[:, :, :, idxc] += top_diff[idxn,idxc,idxh,idxw] * self.input_pad[idxn,:,idxh*self.stride:idxh*self.stride + self.kernel_size,idxw*self.stride:idxw*self.stride+self.kernel_size]
                        #___________________________
                        self.d_bias[idxc] += top_diff[idxn,idxc,idxh,idxw]
                        #________________________
                        bottom_diff[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size] += top_diff[idxn, idxc, idxh, idxw] * self.weight[:, :, :, idxc]
        bottom_diff = bottom_diff = bottom_diff[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]]
        #____________________________
        self.backward_time = time.time() - start_time
        return bottom_diff
    def get_gradient(self):
        return self.d_weight, self.d_bias
    def update_param(self, lr):
        self.weight += - lr * self.d_weight
        self.bias += - lr * self.d_bias
    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
        show_matrix(self.weight, 'conv weight ')
        show_matrix(self.bias, 'conv bias ')
    def get_forward_time(self):
        return self.forward_time
    def get_backward_time(self):
        return self.backward_time

class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride, type=0):
        self.kernel_size = kernel_size
        self.stride = stride
        ### adding
        self.forward = self.forward_raw
        self.backward = self.backward_raw_book
        if type == 1: # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup

        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))
    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size) // self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO： 计算最大池化层的前向传播， 取池化窗口内的最大值
                        h_start = idxh * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = idxw * self.stride
                        w_end = w_start + self.kernel_size
                        self.output[idxn, idxc, idxh, idxw] = np.max(self.input[idxn, idxc, h_start:h_end, w_start:w_end])
                        
                        #self.output[idxn, idxc, idxh, idxw] = 
                        #___________________________
                        curren_max_index = np.argmax(self.input[idxn, idxc, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size])
                        curren_max_index = np.unravel_index(curren_max_index, [self.kernel_size, self.kernel_size])
                        self.max_index[idxn, idxc, idxh*self.stride+curren_max_index[0], idxw*self.stride+curren_max_index[1]] = 1
        return self.output

    def forward_speedup(self, input):
        # TODO: 改进forward函数，使得计算加速
        start_time = time.time()
        #__________________________ added
        self.input = input
        height_out = (self.input.shape[2] - self.kernel_size) // self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        self.input_col = np.zeros([self.input.shape[0], self.input.shape[1], height_out * width_out, self.kernel_size ** 2])
        for i in range(height_out):
            for j in range(width_out):
                self.input_col[:, :, i * width_out + j, :] = self.input[:, :,i * self.stride:i * self.stride + self.kernel_size,j * self.stride:j * self.stride + self.kernel_size].reshape(self.input.shape[0], self.input.shape[1], -1)
        self.output_col = np.max(self.input_col, axis=3)
        max_index_col = np.zeros([self.input.shape[0] * self.input.shape[1] * height_out * width_out, self.kernel_size ** 2])
        max_index_col[np.arange(self.input.shape[0] * self.input.shape[1] * height_out * width_out),np.argmax(self.input_col, axis=3).reshape(-1)] = 1
        max_index_col = max_index_col.reshape([self.input.shape[0], self.input.shape[1], height_out * width_out,self.kernel_size ** 2])
        self.output = self.output_col.reshape([self.input.shape[0], self.input.shape[1], height_out, width_out])
        self.max_index = np.zeros(self.input.shape)
        for i in range(height_out):
            for j in range(width_out):
                self.max_index[:, :, i * self.stride:i * self.stride + self.kernel_size,j * self.stride:j * self.stride + self.kernel_size] = max_index_col[:, :, i * width_out + j, :].reshape([self.input.shape[0], self.input.shape[1], self.kernel_size, self.kernel_size])




        return self.output
    def backward_speedup(self, top_diff):
        # TODO: 改进backward函数，使得计算加速
        #__________________________
        bottom_diff = np.multiply(self.max_index, top_diff.repeat(self.kernel_size,2).repeat(self.kernel_size,3))
        return bottom_diff


    def backward_raw_book(self, top_diff):
        bottom_diff = np.zeros(self.input.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        max_index = np.argmax(self.input[idxn, idxc, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size])
                        max_index = np.unravel_index(max_index, [self.kernel_size, self.kernel_size])
                        bottom_diff[idxn, idxc, idxh*self.stride+max_index[0], idxw*self.stride+max_index[1]] = top_diff[idxn, idxc, idxh, idxw] 
        show_matrix(top_diff, 'top_diff--------')
        show_matrix(bottom_diff, 'max pooling d_h ')
        return bottom_diff

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))
    def forward(self, input):
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        show_matrix(self.output, 'flatten out ')
        return self.output
    def backward(self, top_diff):
        assert list(top_diff.shape[1:]) == list(self.output_shape)
        top_diff = np.transpose(top_diff, [0, 3, 1, 2])
        bottom_diff = top_diff.reshape([top_diff.shape[0]] + list(self.input_shape))
        show_matrix(bottom_diff, 'flatten d_h ')
        return bottom_diff
