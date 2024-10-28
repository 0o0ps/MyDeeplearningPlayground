### 本小节目的：体验kernel, reshape, conv2d, batchsize, input channel, output channel

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

cifar10 = datasets.CIFAR10(root='../1_3dataset_transform/data', train=False, download=True, transform=transforms.ToTensor())
cifar10Loader = torch.utils.data.DataLoader(cifar10, batch_size=64, shuffle=True, num_workers=4)

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride = 1 ,padding=0 )  # 3 input channel, 6 output channel, 3x3 kernel, padding = 1
        # self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)  # 16 input channel, 8 output channel, 3x3 kernel, padding = 1

    def forward(self, x):
        x = self.conv1(x)
        # x = F.relu(self.conv2(x))
        return x
    

## initialize the neural network
conv = Conv()
print(conv)
# 查看卷积核的权重参数
print(conv.conv1.weight.shape)  # 输出卷积核的形状
print(conv.conv1.weight)  # 输出具体的卷积核数值

board = SummaryWriter("logs")
step = 0 
for data in cifar10Loader:
    imgs, labels = data
    outputs = conv(imgs)
    outputs = torch.reshape(outputs, (-1, 3, 30, 30))
    print(imgs.shape)
    print(outputs.shape)
    board.add_images("images", imgs, step)
    board.add_images("outputs", outputs, step)
    ## the output channel is changed and cannot be displayed cause they don't know why the output channel is 6

    ## -1 会自动计算 batchsize
    step += 1
## imgs should be torch.Size([64, 3, 32, 32]) 64 batchsize, 3 input channel, 32x32 image
## output should be torch.Size([64, 6, 30, 30]) 64 batchsize, 6 output channel, 30x30 image

### tensorboard --logdir=./logs