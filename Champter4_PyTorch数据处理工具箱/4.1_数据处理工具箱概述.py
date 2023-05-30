# -*- encoding: utf-8 -*-
'''
@Time    :   2020/6/26:下午11:08
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''

# ---------------------------------------------------------------------------------------------------------------------
# 1. torch.utils.data工具包
#   1) Dataset
#     是一个抽象类，其他数据集需要继承这个类，并且复写其中的两个方法(__getitem__、__len__)
#   2) DataLoader
#     定义一个新的迭代器，实现批量(batch)读取， 打乱数据(shuffle)并提供并行加速等功能
#   3) random_split
#     把数据集随机拆分为给定长度的非重叠的新数据集
#   4) *sample
#     多种采样函数

# ---------------------------------------------------------------------------------------------------------------------
# 2. PyTorch可视化处理工具(Torchvision) 视觉处理工具包。独立于Pytorch，需要另外安装，使用 pip 或 conda 安装。
#     pip install torchvision 或 conda install torchvision
# 包含4个类：
#     1) datasets
#         提供常用的数据集加载，设计上都是继承自torch.utils.data.Dataset，主要包括MNIST、CIFAR10／100、ImageNet和COCO等
#     2) models
#         提供深度学习中各种经典等网络结构以及训练好的模型(如果选择 pretrained=True)，包括alexNet、VGG系列、ResNet系列、Inception系列等
#     3) transforms
#         常用等数据预处理操作，主要包括对Tensor及PIL Image 对象对操作
#     4) utils
#         含两个函数：
#           -  make_grid 它能将多张图拼接在一个网络中
#           - save_img 它能将Tensor 保存成图片
