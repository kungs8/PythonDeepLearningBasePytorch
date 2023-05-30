# -*- encoding: utf-8 -*-
'''
@Time    :   2020/6/28:下午9:38
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''
import os

from torchvision import models, datasets, transforms, utils
from torch.utils import data

# torchvision有4个功能模块:
# 1. models: 模型
# 2. datasets: 下载一些经典数据，3.2节中有实例
# 3. transforms: 提供了对PIL Image对象和Tensor对象对常用操作
# 4. utils:

# 本节中主要使用 datasets 的ImageFolder处理自定义数据集，以及如何使用transform对源数据进行预处理、增强等
# ----------------------------------------------------------------------------------------------------------------------
# 3—1. transforms:
# transforms提供了对PIL Image对象和Tensor对象对常用操作
# 1) 对PIL Image对常见操作
#     Scale／Resize: 调整尺寸，长宽比保持不变
#     CenterCrop、RandomCrop、RandomSizedCrop: 裁剪图片，CenterCrop和RandomCrop在crop时是固定size，RandomResizedCrop则是random size的crop
#     Pad: 填充
#     ToTensor: 把一个取值范围是[0, 255] 的PIL.Image转换成Tensor。
#             形状为(H, W, C)的Numpy.ndarray转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.Floattensor
#     RandomHorizontaFlip: 图像随机水平翻转，翻转概率为0.5
#     RandomVerticalFlip: 图像随机垂直翻转
#     ColorJitter: 修改亮度、对比度和饱和度

# 2) 对Tensor的常见操作
#     Normalize: 标准化，即 减均值，除以标准差
#     ToPILImage: 将Tensor转为PIL.Image

# 如果对数据集进行多个操作，可以通过Compose将这些操作像管道一样拼接起来，类似于nn.Sequential
transforms.Compose([
    # 将给定对PIL.Image进行中心切割，得到给定对size
    # size也可以是一个Integer，在这种情况下，切出来对图片形状是正方形
    transforms.CenterCrop(10),
    # 切割中心点的位置随机选取
    transforms.RandomCrop(20, padding=10),
    # 把一个取值范围是[0, 255]的PIL.Image或者shape为(H, W, C)的numpy.ndarray,
    # 转换为形状为(C, H, W),取值范围是[0, 1]的torch.FloatTensor
    transforms.ToTensor(),
    # 规范化到[-1, 1]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 还可以自己定义一个Python Lambda表达式，如将每个像素值加10，可表示为：
# transforms.Lambda(lambda x: x.add(10))
#
# 更多内容参考官网: https://PyTorch.org/docs/stable/torchvision/transforms.html

# ----------------------------------------------------------------------------------------------------------------------
# 3-2. ImageFolder
# 当文件依据标签处于不同文件下时，可以利用torchvision.datasets.ImageFolder来直接构造出dataset
# loader = datasets.ImageFolder(path)
# loader = data.DataLoader(dataset)

# ImageFolder会将目录中的文件夹名自动转化成序列，当DataLoader载入时，标签自动就是整数序列了
# 下面利用ImageFolder读取不同目录下的图片数据，然后使用transforms进行图像预处理，预处理有多个，用compose把这些操作拼接在一起。
# 然后使用DataLoader加载
# 对处理后的数据用torchvision.utils中的save_image保存为一个png格式文件，然后用Image.open打开该png文件
from torchvision import transforms, utils
from torchvision import datasets
import torch
import matplotlib.pyplot as plt

my_trans = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
train_data = datasets.ImageFolder("./data/torchvision_data",
                                  transform=my_trans)
train_loader = data.DataLoader(dataset=train_data,
                               batch_size=8,
                               shuffle=True)
for i_batch, img in enumerate(train_loader):
    if i_batch == 0:
        print(img[1])

        fig = plt.figure()
        grid = utils.make_grid(img[0])
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.show()

        map_path = os.getcwd() + "/res_data"
        print(map_path)
        if not os.path.exists(map_path):
            os.makedirs(map_path)
            print("map_path is new, makedir!")
        
        map_path = map_path + "/test01.png"
        utils.save_image(grid, map_path)
    break

# 打开test01.png文件
from PIL import Image
img = Image.open("./res_data/test01.png")
img.show()