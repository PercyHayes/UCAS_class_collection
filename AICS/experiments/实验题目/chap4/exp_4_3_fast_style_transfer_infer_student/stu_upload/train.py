# coding=utf-8
from torchvision.models import vgg19
from torch import nn
from zipfile import ZipFile
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch
import cv2
import numpy


class COCODataSet(Dataset):

    def __init__(self):
        super(COCODataSet, self).__init__()
        self.zip_files = ZipFile('./data/train2014.zip')
        self.data_set = []
        for file_name in self.zip_files.namelist():
            if file_name.endswith('.jpg'):
                self.data_set.append(file_name)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        file_path = self.data_set[item]
        image = self.zip_files.read(file_path)
        image = numpy.asarray(bytearray(image), dtype='uint8')
        # TODO: 使用cv2.imdecode()函数从指定的内存缓存中读取数据，并把数据转换(解码)成彩色图像格式。
        ______________________________________________ 
        # TODO: 使用cv2.resize()将图像缩放为512*512大小，其中所采用的插值方式为：区域插值
        ______________________________________________ 
        # TODO: 使用cv2.cvtColor将图片从BGR格式转换成RGB格式
        ______________________________________________ 
        # TODO: 将image从numpy形式转换为torch.float32,并将其归一化为[0,1]
        ______________________________________________ 
        # TODO: 用permute函数将tensor从HxWxC转换为CxHxW
        ______________________________________________ 
        return image


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        #TODO: 调用vgg19网络
        ______________________________________
        a = a.features
        #TODO: 定义self.layer1为第2层卷积后对应的特征
        ______________________________________
        #TODO: 定义self.layer2为第4层卷积后对应的特征
        ______________________________________
        #TODO: 定义self.layer3为第8层卷积后对应的特征
        ______________________________________
        #TODO: 定义self.layer4为第12层卷积后对应的特征
        ______________________________________

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out1, out2, out3, out4


class ResBlock(nn.Module):

    def __init__(self, c):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            
            #TODO: 进行卷积，卷积核为3*1*1
            ______________________________________________
            #TODO: 执行实例归一化
            ______________________________________________
            #TODO: 执行ReLU
            ______________________________________________
            #TODO: 进行卷积，卷积核为3*1*1
            ______________________________________________
            #TODO: 执行实例归一化
            ______________________________________________
        )
        
    def forward(self, x):
        #TODO: 返回残差运算的结果
        _________________________________________


class TransNet(nn.Module):

    def __init__(self):
        super(TransNet, self).__init__()
        self.layer = nn.Sequential(
            
            ###################下采样层################
            # TODO：构建图像转换网络，第一层卷积
            _________________________________________
            # TODO：实例归一化
            _________________________________________
            # TODO：创建激活函数ReLU
            _________________________________________
            # TODO：第二层卷积
            _________________________________________
            # TODO：实例归一化
            _________________________________________
            # TODO：创建激活函数ReLU
            _________________________________________
            # TODO：第三层卷积
            _________________________________________
            # TODO：实例归一化
            _________________________________________
            # TODO：创建激活函数ReLU
            _________________________________________

            ##################残差层##################
            _________________________________________
            _________________________________________
            _________________________________________
            _________________________________________
            _________________________________________

            ################上采样层##################
            #TODO: 使用torch.nn.Upsample对特征图进行上采样
            _________________________________________
            #TODO: 执行卷积操作
            _________________________________________
            #TODO: 实例归一化
            _________________________________________
            #TODO: 执行ReLU操作
            _________________________________________

            #TODO: 使用torch.nn.Upsample对特征图进行上采样
            _________________________________________
            #TODO: 执行卷积操作
            _________________________________________
            #TODO: 实例归一化
            _________________________________________
            #TODO: 执行ReLU操作
            _________________________________________
            
            ###############输出层#####################
            #TODO: 执行卷积操作
            _________________________________________
            #TODO： sigmoid激活函数
            _________________________________________
        )

    def forward(self, x):
        return self.layer(x)


def load_image(path):
    # TODO: 使用cv2从路径中读取图片
    ______________________________________
    # TODO: 使用cv2.cvtColor将图片从BGR格式转换成RGB格式
    ______________________________________
    # TODO: 使用cv2.resize()将图像缩放为512*512大小
    ______________________________________
    # TODO: 将image从numpy形式转换为torch.float32,并将其归一化为[0,1]
    ______________________________________
    # TODO: 将tensor从HxWxC转换为CxHxW，并对在0维上增加一个维度
    ______________________________________
    return image


def get_gram_matrix(f_map):
    """
    """
    n, c, h, w = f_map.shape
    if n == 1:
        f_map = f_map.reshape(c, h * w)
        gram_matrix = torch.mm(f_map, f_map.t())
        return gram_matrix
    else:
        f_map = f_map.reshape(n, c, h * w)
        gram_matrix = torch.matmul(f_map, f_map.transpose(1, 2))
        return gram_matrix


if __name__ == '__main__':
    image_style = load_image('./data/udnie.jpg').cpu()
    net = VGG19().cpu()
    g_net = TransNet().cpu()
    print("g_net build PASS!\n")
    #TODO: 使用adam优化器对g_net的参数进行优化，得到optimizer
    _____________________________________________
    #TODO: 在cpu上计算均方误差损失函得到loss_func函数
    _____________________________________________
    print("build loss PASS!\n")
    data_set = COCODataSet()
    print("load COCODataSet PASS!\n")
    batch_size = 1
    data_loader = DataLoader(data_set, batch_size, True, drop_last=True)
    #TODO：输入的风格图像经过特征提取网络生成风格特征s1-s4
    _____________________________________________
    #TODO: 对风格特征s1-s4计算格拉姆矩阵并从当前计算图中分离下来,得到对应的s1-s4
    _____________________________________________
    _____________________________________________
    _____________________________________________
    _____________________________________________
    j = 0
    count = 0
    epochs = 100
    while j <= epochs:
        for i, image in enumerate(data_loader):
            image_c = image.cpu()
            #TODO: 将输入图像经过图像转化网络输出生成图像image_g
            _____________________________________________
            #TODO: 利用特征提取网络提取生成图像的特征out1、out2、out3、out4
            _____________________________________________

            ###############计算风格损失###################
            #TODO: 对生成图像的特征out1-out4计算gram矩阵，并与风格图像的特征s1-s4通过loss_func求损失，分别得到loss_s1-loss_s4
            _____________________________________________
            _____________________________________________
            _____________________________________________
            _____________________________________________
            #TODO：loss_s1-loss_s4相加得到风格损失loss_s
            _____________________________________________

            ###############计算内容损失###################
            #TODO: 将输入图像经过特征提取网络得到内容特图像的特征c1-c4
            _____________________________________________
            #TODO: 将内容图像特征c2从计算图中分离并与内容图像特征out2通过loss_func得到内容损失loss_c2
            _____________________________________________
            loss_c = loss_c2

            ###############计算总损失###################
            loss = loss_c + 0.000000005 * loss_s

            #######清空梯度、计算梯度、更新参数###########
            #TODO: 梯度初始化为零
            _____________________________________________
            #TODO: 反向传播求梯度
            _____________________________________________
            #TODO: 更新所有参数
            _____________________________________________
            print('j:',j, 'i:',i, 'loss:',loss.item(), 'loss_c:',loss_c.item(), 'loss_s:',loss_s.item())
            count += 1
            if i % 10 == 0:
                #TODO: 将图像转换网络的参数fst_train.pth存储在models文件夹下
                _____________________________________________
                #TODO: 利用save_image函数将tensor形式的生成图像image_g以及输入图像image_c以jpg左右拼接的形式保存在/out/train/文件夹下
                _____________________________________________
        j += 1

    print("TRAIN RESULT PASS!\n")
