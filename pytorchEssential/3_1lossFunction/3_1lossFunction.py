### 本节主要讲述了pytorch中的损失函数，包括交叉熵损失函数、均方误差损失函数
### !(待补)以及自定义损失函数的方法，最后通过一个简单的线性回归模型来演示损失函数的使用
### 
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
pred = torch.tensor([0.1,0.2,0.7],dtype=torch.float32)
target = torch.tensor([3],dtype=torch.float32)
loss = nn.CrossEntropyLoss()
lossSum = nn.CrossEntropyLoss(reduction='sum')
loss(input,pred)
lossSum(input,pred)