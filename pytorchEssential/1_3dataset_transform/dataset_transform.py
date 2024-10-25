### 本节讲述 torch中如何下载数据集，以及如何对数据集进行transform预处理
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Normalize

init = Compose([
                ToTensor()
                ] 
)
### 下载CIFAR10数据集，root为数据集存放路径，train为True表示下载训练集，False表示下载测试集
###? 暂时不懂什么叫train?in
cifar10 = datasets.CIFAR10(root = "./data", train= True ,transform = init ,download = True)
# cifar10 = datasets.CIFAR10(root = "./data", train= False, download = True)
# print(cifar10.__init__)
dir(cifar10)
print(cifar10)
print(cifar10[0])
### ?how can I know the structure of cifar10?
cifar10.classes
img , target = cifar10[0]

# inspect.getmembers(cifar10)
### ?remote ssh的图像怎么打开
# img.show()

board = SummaryWriter("log")
for i in range(10):
    image, _= cifar10[i]
    board.add_image("cifar10", image,i)
    
