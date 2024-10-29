## 本节玩一下Sequential的用法，用于CIFAR10数据集的分类任务,采用cifar10 quick start的网络结构
from cgi import test
import re
import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

cifar10 = datasets.CIFAR10(root='../1_3dataset_transform/data', train=False, download=True, transform=transforms.ToTensor())
cifar10Loader = torch.utils.data.DataLoader(cifar10, batch_size=64, shuffle=True, num_workers=4)
'''
不用Sequential的写法
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ### 请自行计算 padding 大小
        ###? 在这里32layers明明除不尽3呀，难道最后一层不conv了？如果这里小于3会怎么样？
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64*4*4, 64)
        self.linear2 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

'''
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

net = Net()
### 测试一下网络有没有问题
print(net)
testInput = torch.ones((64,3,32,32))
print(net(testInput))

### 
board = SummaryWriter("logs")
board.add_graph(net, testInput)
board.close()

### 测试下cifar10
for data in cifar10Loader:
    img, target = data
    output = net(img)
    print(output)
    break