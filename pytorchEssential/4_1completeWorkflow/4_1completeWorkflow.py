
from cgi import test
import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision




### 数据集
cifar10_train = datasets.CIFAR10(root='../1_3dataset_transform/data', train=True, download=True, transform=transforms.ToTensor())
cifar10_test = datasets.CIFAR10(root='../1_3dataset_transform/data', train=False, download=True, transform=transforms.ToTensor())
cifar10Loader_train  = torch.utils.data.DataLoader(cifar10_train, batch_size=64, shuffle=True, num_workers=4)
cifar10Loader_test = torch.utils.data.DataLoader(cifar10_test, batch_size=64, shuffle=True)

### 看一下数据集的长度
train_data_length = len(cifar10_train)
test_data_length = len(cifar10_test)
print(train_data_length, test_data_length)

### 构建网络,这里最好用import的方法

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

if __name__ == "__main__":
    net = Net() 
    
### 分类问题
loss_fn = nn.CrossEntropyLoss()
### optimizer
learningRate = 1e-2
optimizer = torch.optim.SGD(net.parameters(), lr=learningRate)
### steps
trainSteps = 0
testSteps = 0
epoch = 20

board = SummaryWriter(log_dir='logs')
######################## train part #######################
## 每一个epoch
for i in range(epoch):
    net.train() ## 训练模式
    print(f"-------------------epoch {i+1} start--------------------")
    ### each batchsize
    for data in cifar10Loader_train:
        img, target = data
        output = net(img)
        loss = loss_fn(output, target)
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播
        optimizer.step()   ## 会更新net中的参数
        trainSteps += 1
        if trainSteps % 100 == 0:
            print(f"No {trainSteps} train : loss {loss.item()}")
            board.add_scalar('train_loss', loss.item(), trainSteps)


    ######################## test part #######################
    net.eval() ## 测试模式
    total_loss = 0
    ## 不算梯度
    total_train_steps = 0
    t_correct_count = 0
    with torch.no_grad():
        for data in cifar10Loader_test:
            imgs , target = data
            # print(len(data[1]))
            out = net(imgs)
            loss = loss_fn(out,target)
            total_loss += loss
            # print(f"测试集上的loss :{loss}")
            total_train_steps += 1
            testSteps += 1
            if total_train_steps % 100 == 0:
                print(f'test round no{total_train_steps}: {total_loss.item()}')
                board.add_scalar('test_loss', loss.item(), testSteps)
                # 计算准确率
            pred = out.argmax(dim=1) #1表示行
                # print(pred == target)
            correct_count = (pred == target).sum()
            t_correct_count += correct_count ## 正确数
    
    print(f"total_loss: {total_loss}")  W
    print(f"accuracy: {t_correct_count / test_data_length}")     
            
# torch.save(net.state_dict(), 'model.pth')