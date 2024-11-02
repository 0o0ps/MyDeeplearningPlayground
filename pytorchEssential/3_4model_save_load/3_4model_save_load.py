import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
wd = os.getcwd()
print(wd)

vgg = torchvision.models.vgg16_bn(weights=None)
### method1 保存模型网络，参数
torch.save(vgg,"vgg.pth")

### load model
model = torch.load("vgg.pth")
print(model)

len(vgg.state_dict().keys())
vgg.state_dict().keys()
### method 2 保存模型参数
torch.save(vgg.state_dict(),"vgg_state.pth")
print(vgg.state_dict())

model2 = torch.load("vgg_state.pth")
print(model2)
### 变成了参数
### 要想使用，需要重新定义网络，然后加载参数
vgg2 = torchvision.models.vgg16_bn(weights=None)
vgg2.load_state_dict(model2)
print(vgg2)

### 陷阱
### method1 还是需要先定义网络，然后加载，一般用method2