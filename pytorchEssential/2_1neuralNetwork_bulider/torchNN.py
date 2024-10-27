### 本节讲解pytorch中的神经网络构建
### model is the base class for all neural network modules in pytorch, you can define your own model by subclassing nn.Module.
import torch.nn as nn
import torch

class Mymodel(nn.Module):
    '''method 1 rewrite this function'''
    def __init__(self):
        ## 继承父类的构造函数
        super(Mymodel, self).__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5)
        # self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self,input):
        ## 定义前向传播
        ## ?what is the forward?
        output = input + 1
        return output
## 创建了自己的模型
myModel = Mymodel()
x = torch.tensor(1.0)
myModel(x)
### 返回tensor(2.)
