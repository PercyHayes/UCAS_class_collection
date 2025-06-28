import numpy as np
import matplotlib.pyplot as plt
from model import Network
from data import data


if __name__ == '__main__':

    train_data = data[:,:3]
    train_label = data[:,3:]
    # 不同隐藏节点数对比
    '''
    net1 = Network(3,3,3)
    net2 = Network(6,3,3)
    net3 = Network(9,3,3)
    net4 = Network(12,3,3)
    net5 = Network(15,3,3)
    Loss1 = net1.train(400,0.1,train_data,train_label,mode = 'batch')
    Loss2 = net2.train(400,0.1,train_data,train_label,mode = 'batch')
    Loss3 = net3.train(400,0.1,train_data,train_label,mode = 'batch')
    Loss4 = net4.train(400,0.1,train_data,train_label,mode = 'batch')
    Loss5 = net5.train(400,0.1,train_data,train_label,mode = 'batch')

    plt.plot(Loss1,label = 'h=3')
    plt.plot(Loss2,label = 'h=6')
    plt.plot(Loss3,label = 'h=9')
    plt.plot(Loss4,label = 'h=12')
    plt.plot(Loss5,label = 'h=15')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    '''
    # 不同更新方式对比
    
    hid_num = 6
    net1 = Network(hid_num,3,3)
    train_data = data[:,:3]
    train_label = data[:,3:]
    Loss1 = net1.train(1000,0.1,train_data,train_label,mode = 'batch')
    net2 = Network(hid_num,3,3)
    Loss2 = net2.train(1000,0.1,train_data,train_label,mode = 'single')
    plt.plot(Loss1,label = 'batch')
    plt.plot(Loss2,label = 'single')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    
    # 不同学习率对比
    '''
    net1 = Network(12,3,3)
    net2 = Network(12,3,3)
    net3 = Network(12,3,3)
    net4 = Network(12,3,3)
    net5 = Network(12,3,3)
    Loss1 = net1.train(1200,0.5,train_data,train_label,mode = 'batch')
    Loss2 = net2.train(1200,0.1,train_data,train_label,mode = 'batch')
    Loss3 = net3.train(1200,0.01,train_data,train_label,mode = 'batch')
    Loss4 = net4.train(1200,0.001,train_data,train_label,mode = 'batch')
    Loss5 = net5.train(1200,0.0001,train_data,train_label,mode = 'batch')

    plt.plot(Loss1,label = 'lr=0.5')
    plt.plot(Loss2,label = 'lr=0.1')
    plt.plot(Loss3,label = 'lr=0.01')
    plt.plot(Loss4,label = 'lr=0.001')
    plt.plot(Loss5,label = 'lr=0.0001')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    '''