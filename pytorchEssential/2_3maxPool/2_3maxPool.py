### 本节讲述池化的作用
### 在卷积神经网络中，池化层是一种非常重要的层，一般跟在convolution layer 后面，它的作用是逐渐降低数据的空间维度，从而减少网络的参数和计算量，同时也能够一定程度上控制过拟合。
### 池化层的操作非常简单，它由一个固定大小的滑动窗口在输入数据上滑动，窗口每次滑动的距离称为stride，窗口每次滑动时的取值方式称为pooling方式，常见的pooling方式有max pooling和average pooling。
### ?average pooling是不是另一种形式的convolution?
### ?max pooling是不是另一种形式的convolution?

from math import ceil
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

### 初始化，注意输入数据精度 
testMat = input = torch.tensor([[1,2,0,3,1], 
                    [0, 1, 2, 3, 1],
                    [1, 2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]] , dtype=torch.float32)
print(testMat.shape)
## 现在的tensor不满足pool的要求，需要reshape
testMat = torch.reshape(testMat,(1,1,5,5))
print(testMat.shape)

class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True) 
        # 3x3 kernel, stride = 1, ceil_mode=True表示向上取整, 默认是向下取整(i.e. 不要不完整的取样范围）
    def forward(self, x):
        x = self.maxpool1(x)
        return x 

pool1 =MaxPool()
pool1(testMat)

##! now let's change the ceil mode to False and we will find that the output size is different
### let's try it on the tensorboard 
writer = SummaryWriter('logs')


cifar10 = datasets.CIFAR10('../1_3dataset_transform/data', train=False, download=True, transform=transforms.ToTensor())
cifar10loader = DataLoader(cifar10, batch_size=64, shuffle=True)
step = 0 

for data in cifar10loader:
    img,target = data
    writer.add_images('before', img, step)
    outputImg =  pool1(img)
    writer.add_images('after', outputImg, step)
    step += 1

writer.close()

# class MaxPool(torch.)