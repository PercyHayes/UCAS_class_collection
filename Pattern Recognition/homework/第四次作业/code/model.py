import numpy as np

def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



class Network:
    '''
    三层网络
    '''
    def __init__(self, hidden_dim,input_dim,out_dim):
        '''
        hidden_dim: 隐藏层神经元个数
        input_dim: 输入层神经元个数
        out_dim: 输出层神经元个数
        '''
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.out_dim = out_dim

        #权重随机初始化
        self.w_ih = np.random.randn(self.input_dim,self.hidden_dim)
        self.w_ho = np.random.randn(self.hidden_dim,self.out_dim)

    def forward(self,x):
        '''
        前向传播
        x: 输入数据
        '''
        x = np.reshape(x,(1,-1))
        hid = np.matmul(x,self.w_ih)
        hid_act = tanh(hid)

        net = np.matmul(hid_act,self.w_ho)
        o = sigmoid(net)
        return o, hid_act
    
    def backward(self,o,hid_act,target,x):
        '''
        反向传播
        o: 网络输出
        hid_act: 隐藏层激活值
        target: 目标值
        x: 输入数据
        '''
        x = np.reshape(x,(1,-1))
        #计算输出层误差
        error = np.matmul((target - o),(target-o).T)

        #计算输出-隐藏层梯度
        grad_o = (target-o) * o * (1 - o)
        grad_w_ho = np.matmul(hid_act.T,grad_o)
        #计算隐藏-输入层误差
        grad_h = np.matmul(grad_o,self.w_ho.T) * (1-hid_act**2)
        
   
        grad_w_ih = np.matmul(x.T,grad_h)

        return grad_w_ho,grad_w_ih,error
    
    def train(self,epochs,lr,train_data,train_label,mode = 'single'):
        '''
        epochs: 训练次数
        lr: 学习率
        train_data: 训练数据
        train_label: 训练标签
        mode: 训练模式，single为单样本训练，batch为批量训练
        '''

        #单一样本更新
        arr = np.array(range(0, train_data.shape[0],1))
        if mode == 'single':
            Loss = []
            for i in range(epochs):
                err = 0
                index = np.random.permutation(arr)
                for j in index:
                    o, hid_act = self.forward(train_data[j])
                    grad_w_ho,grad_w_ih,error = self.backward(o,hid_act,train_label[j],train_data[j])

                    #权重更新
                    self.w_ih += lr * grad_w_ih
                    self.w_ho += lr * grad_w_ho
                    err += error.item()
                Loss.append(err/train_data.shape[0])
            return Loss
        elif mode == 'batch':
            Loss = []
            for i in range(epochs):
                err = 0
                Delta_w_ih = 0
                Delta_w_ho = 0
                index = np.random.permutation(arr)
                count = 0
                for j in range(0,train_data.shape[0]):
                    o, hid_act = self.forward(train_data[index[j]])
                    grad_w_ho,grad_w_ih,error = self.backward(o,hid_act,train_label[index[j]],train_data[index[j]])
                    # 权重累积
                    Delta_w_ih += grad_w_ih
                    Delta_w_ho += grad_w_ho
                    err += error.item()
                    
                    count += 1
                    if (count+1) % 10==0:
                        # 权重更新
                        self.w_ih += lr * Delta_w_ih
                        self.w_ho += lr * Delta_w_ho

                        #权重清零
                        Delta_w_ih = 0
                        Delta_w_ho = 0
                Loss.append(err/train_data.shape[0])
            return Loss
        else:
            print('mode error')
            return None
