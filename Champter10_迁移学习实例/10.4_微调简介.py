# -*- encoding: utf-8 -*-
"""
@File       : 10.4_微调实例.py
@Time       : 2023/8/18 16:22
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
# 微调允许修改预先训练好的网络参数来学习目标任务，所以，虽然训练时间要比特征抽取方法长，但精度更高。
# 微调但大致过程是在预先训练过的网络上添加新的随机初始化层，此外，预先训练的网络参数也会被更新，但会使用较小的学习率以防止预先训练好的参数发生较大的改变。
#
# 常用的方法：固定底层的参数，调整一些顶层或具体层的参数。
# 好处：减少训练参数的数量，同时也有助于克服过拟合现象的发生。
# (尤其是当目标任务的数据量不足够大的时候，该方法实践起来很有效果)
# 微调要优于特征提取，因为它能对迁移过来的预训练网络参数进行优化，使其更加适合新的任务。

from torchvision import transforms
# 1.数据预处理
# 这里对数据集添加了几种数据增强方法：图像裁剪、旋转、颜色改变等。
#  测试数据与特征提取一样，没有变化
trans_train = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2.加载预训练模型
from torchvision import models
net = models.resnet18(pretrained=True)

# 3.修改分类器
# 修改最后全连接层，把类别数由原来的1000改为10
import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.fc = nn.Linear(512, 10)
net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])  # 如果是多GPU，则需要这句
net.to(device)

# 4.选择损失函数及优化器
# 这里学习率为`1e-3`, 使用微调训练模型时，会选择一个稍大一点学习率，如果选择太小，效果要差一些
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9)

# 5.训练及验证模型
train(net, trainloader, testloader, 20, optimizer, criterion)

# 完整的详见 `10.4_微调实例.py`