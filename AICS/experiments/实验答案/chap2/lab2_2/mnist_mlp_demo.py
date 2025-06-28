import pycnnl
import time
import numpy as np
import struct
import os

class MNIST_MLP(object):
    def __init__(self):
        # set up net
        
        self.net = pycnnl.CnnlNet()
        self.input_quant_params = []  
        self.filter_quant_params = []
    
    def build_model(self, batch_size=100, input_size=784, 
                    hidden1=100, hidden2=100, out_classes=10): # 建立网络结构
    
        self.batch_size = batch_size
        self.out_classes = out_classes
        

        # creating layers
        # TODO：使用 pycnml 建立三层神经网络结构     
      
        self.net.setInputShape(batch_size, input_size, 1, 1) #设置输入参数
        # fc1
        
        input_shapem1=pycnnl.IntVector(4)  
        input_shapem1[0]=batch_size
        input_shapem1[1]=1
        input_shapem1[2]=1
        input_shapem1[3]=input_size
        weight_shapem1=pycnnl.IntVector(4)  
        weight_shapem1[0]=batch_size
        weight_shapem1[1]=1
        weight_shapem1[2]=input_size
        weight_shapem1[3]=hidden1

        output_shapem1=pycnnl.IntVector(4)  
        output_shapem1[0]=batch_size
        output_shapem1[1]=1
        output_shapem1[2]=1
        output_shapem1[3]=hidden1


        # fc2
        hidden_shape1 = pycnnl.IntVector(4)  
        hidden_shape1 = output_shapem1
        weight_shapem2 = pycnnl.IntVector(4)  
        weight_shapem2[0] = batch_size
        weight_shapem2[1] = 1
        weight_shapem2[2] = hidden1
        weight_shapem2[3] = hidden2
        #output_shapem3 = pycnnl.IntVector(4) 
        
        #fc3
        hidden_shape2 = pycnnl.IntVector(4) 
        hidden_shape2[0] = batch_size
        hidden_shape2[1] = 1
        hidden_shape2[2] = 1
        hidden_shape2[3] = hidden2
        weight_shapem3 = pycnnl.IntVector(4) 
        weight_shapem3[0] = batch_size
        weight_shapem3[1] = 1
        weight_shapem3[2] = hidden2
        weight_shapem3[3] = out_classes
        output_shapem3 = pycnnl.IntVector(4) 
        output_shapem3[0] = batch_size
        output_shapem3[1] = 1
        output_shapem3[2] = 1
        output_shapem3[3] = out_classes

        softmax_shape = pycnnl.IntVector(3)
        softmax_shape[0] = batch_size
        softmax_shape[1] = 1
        softmax_shape[2] = out_classes 

    
        self.net.createMlpLayer('fc1', input_shapem1, weight_shapem1, output_shapem1)
        #__________________ 
        self.net.createReLuLayer('relu1')
        self.net.createMlpLayer('fc2',hidden_shape1,weight_shapem2,hidden_shape2)
        self.net.createReLuLayer('relu2')
        self.net.createMlpLayer('fc3',hidden_shape2,weight_shapem3,output_shapem3)
        self.net.createSoftmaxLayer('softmax',softmax_shape,axis = 2)

    
    def load_mnist(self, file_dir, is_images = True):
        # Read binary data
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()
        # Analysis file header
        if is_images:
            # Read images
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:
            # Read labels
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows, num_cols = 1, 1
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
        print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))
        return mat_data
    
    def load_data(self, data_path, label_path):
        print('Loading MNIST data from files...')
        test_images = self.load_mnist(data_path, is_images = True)#________________________
        test_labels = self.load_mnist(label_path,is_images = False)#________________________
        self.test_data = np.append(test_images, test_labels, axis=1)

    def load_model(self, param_dir):   # 加载参数
        # TODO：使用pycnml接口分别为三层全连接层加载参数
        print('Loading parameters from file ' + param_dir)
        
        params = np.load(param_dir,allow_pickle=True,encoding="latin1").item()
        weigh1 = params['w1'].flatten().astype(np.float64)
        bias1 = params['b1'].flatten().astype(np.float64)
        self.net.loadParams(0, weigh1, bias1)
        print("layer 1 done")
        weigh2 = params['w2'].flatten().astype(np.float64)
        bias2 = params['b2'].flatten().astype(np.float64)
        self.net.loadParams(2,weigh2,bias2)
        #____________________

        weigh3 = params['w3'].flatten().astype(np.float64)
        bias3 = params['b3'].flatten().astype(np.float64)
        self.net.loadParams(4,weigh3,bias3)
        #____________________

           
    def forward(self):
        return self.net.forward()

    def evaluate(self):
        pred_results = np.zeros([self.test_data.shape[0]])
        for idx in range(self.test_data.shape[0]//self.batch_size):
            # print("batch %d"%idx)
            batch_images = self.test_data[idx*self.batch_size:(idx+1)*self.batch_size, :-1]
            data = batch_images.flatten().tolist()
            self.net.setInputData(data)
            start = time.time()
            self.forward()
            end = time.time()
            print('inferencing time: %f'%(end - start))
            prob = self.net.getOutputData()
            prob = np.array(prob).reshape((self.batch_size, self.out_classes))
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx*self.batch_size:(idx+1)*self.batch_size] = pred_labels
        if self.test_data.shape[0] % self.batch_size >0: 
            last_batch = self.test_data.shape[0]//self.batch_size*self.batch_size
            batch_images = self.test_data[-last_batch:, :-1]
            data = batch_images.flatten().tolist()
            self.net.setInputData(data)
            self.forward()
            prob = self.net.getOutputData()
            pred_labels = np.argmax(prob, axis=1)
            pred_results[-last_batch:] = pred_labels
        accuracy = np.mean(pred_results == self.test_data[:,-1])
        print('Accuracy in test set: %f' % accuracy)

HIDDEN1 = 256#32
HIDDEN2 = 128#16
OUT = 10
def run_mnist():
    batch_size = 10000#10000
    h1, h2, c = HIDDEN1, HIDDEN2, OUT
    mlp = MNIST_MLP()
    mlp.build_model(batch_size=batch_size, hidden1=h1, hidden2=h2, out_classes=c)
    model_path = 'weight.npy'
    test_data = '../../mnist_data/t10k-images-idx3-ubyte'
    test_label = '../../mnist_data/t10k-labels-idx1-ubyte'
    mlp.load_data(test_data, test_label)
    
    mlp.load_model(model_path)

    for i in range(10):
        mlp.evaluate()

if __name__ == '__main__':
    run_mnist()
