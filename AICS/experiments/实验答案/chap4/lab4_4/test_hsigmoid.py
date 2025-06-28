import numpy as np
import torch
import torchvision
import numpy as np
#TODO：导入自定义连接库
#______________________________________
import hsigmoid_extension 

def hsigmoid_cpu(rand):
    rand = rand.contiguous()
    #TODO：调用hsigmoid函数对rand进行处理得到输出结果output
    #______________________________________
    ori_shape = rand.shape
    rand_flat = rand.flatten(1)
    output_flat = hsigmoid_extension.hsigmoid_cpu(rand_flat)
    output = output_flat.view(ori_shape)
    return output.contiguous()

def test_hsigmoid():
    torch.manual_seed(12345)
    rand = (torch.randn(3, 512, 512, dtype=torch.float32).abs()+1)
    #TODO：调用hsigmoid_cpu函数对rand进行处理得到输出结果output_cpu
    #______________________________________
    rand = rand.unsqueeze(0)
    output_cpu = hsigmoid_cpu(rand)
    print("------------------hsigmoid test completed----------------------")
    print("input: ", rand)
    print("input_size:", rand.size())
    print("output: ", output_cpu)
    print("output_size:", output_cpu.size())

    print("TEST hsigmoid PASS!\n")
    
test_hsigmoid()
