### transform的应用
### ? 为什么要用transform
### 怎么用transform
### 常用指令 totensor是什么？
### tensor是什么？为什么要transform成tensor？ 后续操作都是基于tensor的 

from torchvision import transforms
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter


imgPath = '/mnt/public/wangxinkang/projects/pytorchPlayground/MyDeeplearningPlayground/pytorchEssential/data/hymenoptera_data/train/ants/0013035.jpg'
img = Image.open(imgPath)
### Totensor的对象
tensor_trans = transforms.ToTensor()
### 传入图片
tensor_img = tensor_trans(img)
print(tensor_img)

## 转置成tensor

###复习上节课上的tensorboard
img_cv = cv2.imread(imgPath)
tensor_trans_cv = transforms.ToTensor()
tensor_cv = tensor_trans_cv(img_cv)
board = SummaryWriter("log")
board.add_image("cv", tensor_cv)
board.close()

###* 常用的transform 模块
### Compose
### totensor
### Normalize
### ToPILImage
### 上面已经讲过了totensor,接下来我们讲一下Normalize
################接下来是操作####################
### 同样的，我们先加载一张图片
imgPath = '/mnt/public/wangxinkang/projects/pytorchPlayground/MyDeeplearningPlayground/pytorchEssential/data/hymenoptera_data/train/ants/0013035.jpg'
img_cv = cv2.imread(imgPath)
type(img_cv)
## transform it into tensor
## object of totensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img_cv)
board = SummaryWriter("log")
board.add_image("Normal",tensor_img,0)
## Normalize
print(tensor_img[0][0][0])
norm1 = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
## normalization
norm1_img = norm1(tensor_img)
print(norm1_img[0][0][0])
board.add_image("Normal",norm1_img,1)
##change the mean and std
norm1 = transforms.Normalize(mean=[0.3,0.3,0.1],std=[0.7,0.1,0.4])
###?为什么图片变成全黑了？
### ?有人说要把文件尾缀改成.png 为什么？
###?如何调成紫色？
## normalization
norm1_img = norm1(tensor_img)
print(norm1_img[0][0][0])
board.add_image("Normal",norm1_img,2)

######Resize的用法######
tranResize = transforms.Resize((512,512))
img_resize = tranResize(tensor_img)
print(type(img_resize))
board.add_image("resize",img_resize,0)


##### Compose的用法######
trans_resize2 = transforms.Resize(512)
### 组合transform tran to tensor->resize->normalize
trans_compose = transforms.Compose([tensor_trans,trans_resize2,norm1])
img_compose = trans_compose(img_cv)
board.add_image("compose",img_compose,1)



