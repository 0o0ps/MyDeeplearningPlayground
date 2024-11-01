### 本节主要讲述了optimizer的使用
import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter



### 体验一下cifar10的neural network的操作
cifar10 = datasets.CIFAR10(root='../1_3dataset_transform/data', train=False, download=True, transform=transforms.ToTensor())
cifar10Loader = torch.utils.data.DataLoader(cifar10, batch_size=64, shuffle=True, num_workers=4)
### 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ### 请自行计算 padding 大小
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
### 设置loss
loss_fc = nn.CrossEntropyLoss()
## 设置优化器 lr是学习率
optm = torch.optim.SGD(net.parameters(), lr=0.01)
for epoch in range(10):
    lsSum = 0
    for data in cifar10Loader:
        
        imgs, targets = data
        output = net(imgs)
        ls = loss_fc(output,targets)
        optm.zero_grad() ## 梯度清零
        ls.backward() ## 反向传播，用于计算梯度
        optm.step() ## 更新参数
        lsSum += ls
    print(lsSum)
    
##net - _modules -model - _modules - "0" - "weight"