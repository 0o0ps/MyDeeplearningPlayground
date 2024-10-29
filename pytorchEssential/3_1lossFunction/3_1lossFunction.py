### 本节主要讲述了pytorch中的损失函数，包括交叉熵损失函数、均方误差损失函数
### !(待补)以及自定义损失函数的方法，最后通过一个简单的线性回归模型来演示损失函数的使用
### Loss有两个作用 1. 对目前的预测打分评分，2. 使用梯度下降的方法，反向传播，优化参数
###  crossEntropyLoss 在分类任务中经常被使用
import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# import os
# os.getcwd()

input = torch.tensor([1,2,3],dtype=torch.float32)
pred = torch.tensor([1,2,5],dtype= torch.float32)
## 体验一下L1Loss，曼哈顿距离
loss = nn.L1Loss()
lossSum = nn.L1Loss(reduction='sum')
loss(input, pred)
lossSum(input, pred)
## 体验一下MSELoss，欧式距离
loss = nn.MSELoss()
lossSum = nn.MSELoss(reduction='sum')
loss(input, pred)
lossSum(input, pred)
## 体验一下交叉熵损失函数
## 交叉熵公式为：H(p,q) = -∑p(x)log(q(x))
pred = torch.tensor([[0.1,0.2,0.3]],dtype=torch.float32)
target = torch.tensor([1])
### 请注意，CrossEntropy的输入要求
pred = torch.reshape(pred,(1,3))
loss = nn.CrossEntropyLoss()
loss(pred,target)
### the output should be 1.1019

### 体验一下cifar10的neural network的操作
cifar10 = datasets.CIFAR10(root='../1_3dataset_transform/data', train=False, download=True, transform=transforms.ToTensor())
cifar10Loader = torch.utils.data.DataLoader(cifar10, batch_size=64, shuffle=True, num_workers=4)
### 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ### 请自行计算 padding 大小
        ###? 在这里32layers明明除不尽3呀，难道最后一层不conv了？如果这里小于3会怎么样？
        self.model = nn.Sequential(
                            nn.Conv2d(3, 32, kernel_size=5, padding=2),
                            nn.MaxPool2d(2),
                            nn.Conv2d(32, 32, kernel_size=5, padding=2),
                            nn.MaxPool2d(2),
                            nn.Conv2d(32, 64, kernel_size=5, padding=2),
                            nn.MaxPool2d(2),
                            nn.Flatten(),
                            nn.Linear(64*4*4, 64),
                            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return x    
### 实例
net = Net()

for data in cifar10Loader:
    imgs, targets = data
    output = net(imgs)
    ls = loss(output,targets)
    print(ls)
    break



