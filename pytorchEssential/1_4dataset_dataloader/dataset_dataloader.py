### 本节主要讲解如何使用pytorch中的dataset和dataloader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from pprint import pprint

init = Compose([
                ToTensor()
                ] 
)

## 测试集
cifar10 = datasets.CIFAR10(root = "./data", train= False ,transform = init ,download = True)

'''batch_size：在训练深度学习模型时，完整的数据集往往很大，所以无法一次性将整个数据集送入模型进行训练。
为此，训练通常采用Mini-Batch的方式，即将数据集划分为多个小批次，每个批次逐个送入模型进行训练。
每处理完一个批次的数据，模型会计算一次损失并更新参数。'''
###? 什么时候需要drop last?
testLoader =  DataLoader(cifar10, batch_size = 32, shuffle = True,num_workers=0,drop_last=False)
board = SummaryWriter("log")
step = 0


### 多少epoch
for epoch in range(2):
    # 每个 epoch 里怎么抓牌
    step = 0
    # 打印每个 data 返回的内容的 shape
    # pprint([item.shape for item in data])
    ##? 有没有什么办法可以怎么用testLoader这个对象
    ###? 我怎么知道这个data return的是什么？
    for data in testLoader:
        img, target = data  # 假设 data 是一个元组 (img, target)
        board.add_images(f"cifar10_{epoch}", img, step)
        step += 1

'''
### 多少epoch
for epoch in range(2):
    ### 每个epoch里怎么抓牌
    step = 0
    for data in testLoader:
        ## pprint([item.shape for item in data])

        img, target = data
        board.add_images(f"cifar10{epoch}", img,step)
        step += 1
        
        # break
'''    
