# -*- encoding: utf-8 -*-
"""
@File       : 10.2_特征提取.py
@Time       : 2023/8/17 09:45
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
在特征提取中，可以在预先训练好的网络结构后，修改或添加一个简单的分类器，将源任务上的预先训练好的网络作为另一个目标任务的特征提取器，
只对最后增加的分类器参数进行重新学习，而预先训练好的网络参数不会被修改或冻结。

一、PyTorch提供的预处理模型
迁移学习，需要使用对应的预训练模型。PyTorch提供来很多现成的预训练模块，直接拿来用就可以。
torchvision.models 模块中有很多模型，这些模型可以只有随机值参数的架构或已在大数据集训练过的模型。
预训练模型可以通过传递参数PRETRAINED=True构造，它将从torch.utils.model_zoo中提取相关的预训练模型。
1. models模块中包括以下模型
    - Alexnet
    - VGG
    - ResNet
    - Squeezenet
    - DenseNet
    - Inception v3
    - GoogleNet
    - ShuffleNet v2

2. 调用随机权重的模型
import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()

3. 获取预训练模型
在torch.utils.momdel_zoo 中提供来预训练模型，通过传递参数 pretrained=True 来构造。如果pretrained=False，表示只需要网络结构，不需要用预训练模型的参数来初始化。
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

4. 注意不同模式
有些模型在训练和测试阶段用到了不同的模块，例如标准化(BatchNormalization)、Dropout层等。使用 model.train() 或 model.eval() 可以切换到相应的模式。

5. 规范化数据
所有的预训练模型都要求输入图片以相同的方式进行标准化，即：小批(Mini-Batch)3通道RGB格式(3*h*w)，其中h和w应小于224。
图片加载时像素值的范围应在[0, 1]内，然后通过指定 mean=[0.485, 0.456, 0.406] 和 std=[0.229, 0.224, 0.225] 进行标准化, eg:
from torchvision import transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

6. 如何冻结某些层
如果需要冻结排除最后一层之外的所有网络，可设置requires_grad=False，这样便可冻结参数，在backward()中不计算梯度，eg:
from torchvision import models
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False