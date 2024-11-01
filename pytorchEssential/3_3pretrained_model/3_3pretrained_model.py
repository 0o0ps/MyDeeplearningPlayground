import torch
# from torchvision import transforms,datasets,models
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision

pretrainedModel = torchvision.models.vgg16_bn(weights='IMAGENET1K_V1')
notPretrainedModel = torchvision.models.vgg16_bn(weights=None)
## 看看有哪些层
print(pretrainedModel)
## 尝试修改最后一层，以适配cifar10数据集
# pretrainedModel.classifier[6] = nn.Linear(4096,10)
### way 1
pretrainedModel.add_module('My_linear',nn.Linear(4096,10))
print(pretrainedModel)
### way 2 加在classifier的最后一层
pretrainedModel.classifier.add_module('My_linear',nn.Linear(4096,10))
print(pretrainedModel)
### way 3 直接替换
pretrainedModel.classifier[7] = nn.Linear(14096,10)
print(pretrainedModel)
### 请仔细观察输出，看看有什么不同，这三种方式有什么区别
###? 如何删除一层？