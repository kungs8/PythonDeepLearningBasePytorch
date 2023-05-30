# -*- encoding: utf-8 -*-
'''
@Time    :   2020/7/20:下午3:15
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''

# ----------------------------------------------------------------------------------------------------------------------
# 损失函数(Loss Function)在机器学习中非常重要，因为训练模型的过程实际就是优化损失函数的过程。
# 损失函数对每个参数的偏导数就是梯度下降中提到的梯度，防止过拟合时添加的正则化项也是在损失函数后面。
# 损失函数用来衡量模型的好坏，损失函数越小说明模型和参数越符合训练样本。
# 任何能够衡量模型预测值与真实值之间的差异的函数都可以叫作损失函数。
# 在机器学习中常用的损失函数有两种，即交叉熵(Cross Entroy)和均方误差(Mean squared error, MSE)，分别对机器学习中的分类问题和回归问题。

# 对分类问题的损失函数一般采用交叉熵，交叉熵反应的两个概率分布的距离(不是欧氏距离)。
# 分类问题进一步又可分为多目标分类，如一次要判断100张图是否包含10种动物，或单目标分类。

# 回归问题预测的不是类别，而是一个任意实数。
# 在神经网络中一般只有一个输出节点，该输出值就是预测值。
# 反应的预测值与实际之间的距离可以用欧式距离来表示，所以对这类问题通常使用均方差作为损失函数，均方差的定义如下：
# $ MSE = \frac{\sum_{i=1}^{n}(y_i - (y_i)^{'})**2}{n}$

# PyTorch 中已集成多种损失函数，这里介绍两个经典的损失函数，其它的损失函数基本上是在它们的基础上的变种或延伸。
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
# 1. torch.nn.MSELoss 均方误差损失函数
#   nn.MSELoss(size_average=None, reduce=None, reduction="mean")
# 计算公式：
# $l(x, y) = L = {l1, l2, ..., lN}^T, li = (xi - yi)^2, N是批量大小$
# 如果参数 reduction 为 非None(缺省值为 'mean'),则：
# l(x, y) = mean(L), if reduction='mean'; sum(L), if reduction='sum'
# x 和 y 是任意形状的张量，每个张量都有n个元素，如果reduction取'None'， l(x, y) 将不是标量；如果取'sum'， 则l(x, y)只是差平方的和，但不会除以n。
# 参数说明：
#     - size_average, reduce在以后版本将移除，主要看参数 reduction，reduction 可以取 None、mean、sum,缺省值为mean。
#     - 如果size_average, reduce取值，将覆盖reduction的取值。

torch.manual_seed(10)  # 设置随机种子，为了后续重跑数据一样
loss = nn.MSELoss(reduction="mean")  # reduction 默认为"mean"，reduction可取值none、mean、sum，缺省值为mean
input = torch.randn(1, 2, requires_grad=True)
print("MSELoss input:", input)

target = torch.randn(1, 2)
print("MSELoss target:", target)

output = loss(input, target)
print("MSELoss output:", output)

output.backward()

# ----------------------------------------------------------------------------------------------------------------------
# 2. torch.nn.CrossEntropyLoss 交叉熵损失，又称对数似然损失、对数损失；二分类时还可称之为逻辑回归损失
# PyTorch中，这里不是严格意义上的交叉熵损失函数，而是先将input经过softmax激活函数，将向量"归一化"成概率形式。然后再与target计算严格意义上的交叉熵损失。
# 在多分类任务中，经常采用softmax激活函数+交叉熵损失函数，因为交叉熵描述了两个概率分布的差异，然而神经网络输出的是向量，并不是概率分布的形式。
# 所以需要softmax激活函数将一个向量进行"归一化"成概率分布的形式，再采用交叉损失函数计算loss
# 一般格式：
#   torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
# 计算公式：
#   $loss(x, class) = -log(\frac{exp(x[class])}{\sum_{j}exp(x[j])}) = -x[class] + log(\sum_{j}exp(x[j]))$
# 如果带上权重参数weight, 则：
#   $loss(x, class) = weight[class](-x[class] + log(\sum_{j}exp(x[j])))$
# weight(Tensor): 为每个类别的loss设置权值，常用于类别不均衡问题。weight必须是float类型的tensor，其长度要与类别C一致，即每一个类别都要设置weight。

import torch
from torch import nn
torch.manual_seed(10)  # 设置随机种子，为了后续重跑数据一样

loss = torch.nn.CrossEntropyLoss()
# 假设类别数是5
input = torch.randn(3, 5, requires_grad=True)
print(f"CrossEntropyLoss input:{input}")
# 每个样本对应的类别索引，其值范围为[0, 4]
target = torch.empty(3, dtype=torch.long).random_(5)
print(f"CrossEntropyLoss target:{target}")
output = loss(input, target)
print(f"CrossEntropyLoss output:{output}")
output.backward()