### 本节主要介绍了一些常用的层，包括：全连接层、激活函数、批标准化层、Dropout层、线性层
### batchNormalization 是用来解决梯度消失和梯度爆炸的问题的，它是通过对每一层的输出进行归一化来解决的
### Dropout 是用来解决过拟合的问题的，它是通过随机的将一些神经元的输出设置为0来解决的
### Linear 是一个简单的全连接层，它的输入和输出都是二维张量，一般用来做特征的线性变换
### embedding layer 在神经网络中，embedding 层（嵌入层）主要用于将离散的输入（如单词或类别）转换为稠密的向量表示。简单来说，它是把机器看不懂的文本或分类数据，变成它能理解的数字向量，这些向量还可以反映词汇或类别之间的关系。
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter



tensorTest = torch.tensor([[1,2,0,3,1], 
                    [0, 1, 2, 3, 1],
                    [1, 2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]] , dtype=torch.float32)

class LL(torch.nn.Module):
    def __init__(self):
        super(LL, self).__init__()
        self.linear = nn.Linear(5,2)

    def forward(self, x):
        # x = self.sigmoid(x)
        x = self.linear(x)
        return x
ll = LL()
ll(tensorTest)

'''
LL 类中的 nn.Linear(5, 2) 会对每一行（有 5 个特征）进行一个线性变换，将它们转换为具有 2 个特征。
例如，ll(tensorTest) 会对 tensorTest 中的每一行（形状为 (5,)）应用线性变换，将输出的形状变为 (5, 2)。
然后，通过对结果转置（output_rows.T）来交换行和列
'''