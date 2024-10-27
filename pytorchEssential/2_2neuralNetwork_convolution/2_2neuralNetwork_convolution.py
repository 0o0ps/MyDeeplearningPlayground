### 本节讲解pytorch中的神经网络中的卷积层
###!Before this tutorial, you should know that the image is made up of pixels, and the pixels are made up of numbers and there are three channels in the image in the RGB format. You should also know that the convolutional layer is used to extract features from the image. The convolutional layer is made up of filters.
# import torch.nn as nn
import torch
import torch.nn.functional as F


input = torch.tensor([[1,2,0,3,1], 
                    [0, 1, 2, 3, 1],
                    [1, 2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]])
### 太有实力了，copilot居然知道我想要卷积核
kernel = torch.tensor([[1,2,1],
                    [0,1,0],
                    [2,1,0]])
print(input.shape)
print(kernel.shape)
### 这两个tensor和 conv2d中要求的不一致
input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))
### batchsize, channel, height, width


output = F.conv2d(input, weight = kernel, stride = 1)
print(output)
'''
the output should be
##
tensor([[[[10, 12, 12],
          [18, 16, 16],
          [13,  9,  3]]]])
##
'''
## stride 步长 can be a tuple, which controls the stride for each dimension of the input(i.e.  the width and height)
output2 = F.conv2d(input, weight = kernel, stride = 2)
print(output2)
'''
the output should be
##
tensor([[[[10, 12],
          [13,  3]]]])
##
'''
### padding是指在输入的每一条边补充0的层数，padding = 1表示在输入的每一条边都补充一圈0, 其目的是为了控制卷积之后的图像大小不变？or 保留边缘信息？
### ANSWER: 保留边缘信息, 填充方式还有对称填充
### Is there any way to fill 1 ,2 or other numbers? why filled with 0?
output3 = F.conv2d(input, weight = kernel, stride = 1, padding = 1)
print(output3)


### dilation是指卷积核的间隔，dilation = 2表示卷积核每隔一个元素进行卷积
