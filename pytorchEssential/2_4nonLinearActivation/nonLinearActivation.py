### 本节主要讲解非线性激活函数
# 1. Sigmoid
# 2. ReLU

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

class NLA(torch.nn.Module):
    def __init__(self):
        super(NLA, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # x = self.sigmoid(x)
        x = self.relu(x)
        return x

###      
nla = NLA()
nla(tensorTest)
    

    
    