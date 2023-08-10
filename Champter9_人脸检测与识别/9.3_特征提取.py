# -*- encoding: utf-8 -*-
"""
@File       : 9.3_特征提取.py
@Time       : 2023/8/10 16:15
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import math

# import some packages
# --------------------
通过人脸检测和对齐后，就获得了包括人脸的区域图像，然后通过深度卷积网络，把输入的人脸图像转换为一个向量，这个过程就是特征提取。
特征提取是一项重要内容，传统机器学习这部分往往要占据大部分时间和精力，有时虽然花去了时间，但效果却不一定理想。深度学习却可以自动获取特征。
    - 传统机器学习算法: 输入 -> 人工特征提取 -> 权重学习 -> 预测结果
    - 深度学习算法: 输入 -> 基础特征提取 -> 多层复杂特征提取 -> 权重学习 -> 预测结果

人脸识别的一个关键问题就是如何衡量人脸的相似或不同。
对分类问题，通过在最后一层添加softmax函数，把输出转换为一个概率分布，然后使用信息熵进行类别的区分。

而对于普通的分类任务，网络最后一层全连接层输出的特征主要可分就行，并不要求类内紧凑和类间分离，这一点非常不适合人脸识别任务。
如果人脸之间的相似度用最后softmax输出的向量之间的欧氏距离，效果往往不很理想。
例如使用CNN对MNIST进行分类，设计一个卷积神经网络，让最后一层输出一个2维向量(便于可视化)，可以看出同一类的点之间距离可能较大，不同类之间的距离(如在靠近中心位置)可能很小。
如果通过欧氏度量方法哎衡量两个向量(或两个人脸)的相似度效果就不理想。
因此，需要设计一个有效的Loss Function 使得学习到的深度特征具有比较强的可区分性。

几种损失函数及其优缺点：
    1) softmax
        softmax损失函数是最初的人脸识别函数，其原理是去电最后的分类层，作为解特征网络导出特征向量用于人脸识别。
        softmax训练的时候收敛地很快，但是精确度一般达到0.9左右就不会再上升了。
        一方面是作为分类网络，softmax不能像度量学习(Metric Learning)一样显式的优化类间和类内距离，所以性能不会特别好。
        softmax 损失函数：
            L_{S} = -\frac{1}{N}\Sigma_{i=1}^{N}log{\frac{e^{w_{yi}^{T}x_{i}+b_{y_{i}}}}{\Sigma_{j}^{n} e^{w_{j}^{T}x_{i}+b_{j}}}}
        其中N批量大小(Batch-Size)，n是类别数目。
    2) 三元组损失(Triplet Loss)
        Triplet Loss属于度量学习(Metric Learning), 通过计算两张图像之间的相似度，使得输入图像被归入到相似度大的图像类别中去，
        使同类样本之间的距离尽可能缩小，不同类样本之间的距离尽可能放大。其损失函数为：
            L_{t} = \Sigma_{i=1}^{N}[\| f(x_{i}^{a}) -  f(x_{i}^{p})\|_{2}^{2} - \| f(x_{i}^{a}) -  f(x_{i}^{n})\|_{2}^{2} + \alpha]_{+}
        其中N是批量大小(Batch-Size)，$x_{i}^{a}$、$x_{i}^{p}$、$x_{i}^{n}$为每次从训练数据中取出的3张人脸图像，前两个表示同一个人，$x_{i}^{n}$为一个不同人的图像。
        `\|`表示欧氏距离，最后的`+`表示`[]`内的值大于0时，取`[]`内的值，否则取0。
        三元损失直接使用度量学习，因此可以解决人脸的特征表示问题。
        缺点：在训练过程中，元组的选择要求的技巧比较高，而且要求数据集比较大。
    3) 中心损失(Center Loss)
        类内距离有的时候甚至是比内间距离要大的，这也是使用softmax损失函数效果不好的原因之一，它具备分类能力但是不具备度量学习(Metric Learning)的特性，没法压缩同一类别。
        Center Loss，用于压缩同一类别。
        核心：为每类别提供一个类别中心，最小化每个样本与该中心的距离，其损失函数：
            L_{C} = \Sigma_{i=1}^{N}\|x_{i} - c_{y_{i}}\|_{2}^{2}
        其中$x_{i}$为一个样本，$y_{i}$是该样本对应的类别，$C_{y_{i}}$为该类别的中心。$L_{C}$比较好的解决同类间的内聚性，利用中心损失时，一般还会加上softmax损失以保证类间的可分性。
        最终的损失函数由两部分组成：
            L = L_{S} + \lambda L_{C}
        其中 $\lambda$用于平衡两个损失函数，通过Center Loss 方法处理后，为每个类别学习一个中心，并将每个类别的所有特征向量拉向对应类别中心，
        当中心损失的权重$\lambda$越大，生成的特征就会越具有内聚性。
        $L_{t}$、$L_{C}$都基于欧氏距离的度量学习，在实际应用中也取得了不错的效果，
        但Center Loss为每个类别需要保留一个类别中心，当类别数量很多(>10000)时，这个内存消耗非常可观，对GPU的内存要求较高
    4) ArcFace
        在softmax损失函数中，把 $W_{Y_{I}^{T}x_{i}$ 可以等价表示为：$|W_{y_{i}}|\cdot|x_{i}|cos(\theta)$,
        其中 `||` 表示模，$\theta$ 为权重 $W_{y_{i}}$ 与特征$x_{i}$的夹角。
        对权重及特征进行归一化，原来的表达式可简化为：
            L_{arc} = -\frac{1}{N}\Sigma_{i=1}^{N}log\frac{e^{S\cdot(cos(\theta_{y_{i}} + m))}}{e^{S\cdot(cos(\theta_{y_{i}} + m))} + \Sigma_{j=1,j\ne y_{i}}^{n}e^{S\cdot cos\theta _{j}}}

        ArcFace损失函数不仅对权重进行了正则化，还对特征进行了正则化。
        另乘上一个scale参数(简写为S)，使分类映射到一个更大到超球面上，使分类更方便。

总的来说，ArcFace优于其他几种Loss，著名的Megaface赛事，在很长的一段时间都停留在91%左右，在洞见实验室使用ArcFace提交后，准确率迅速提到了98%。
ArcFace伪代码步骤：
    1) 对x进行归一化
    2) 对W进行归一化
    3) 计算$W_{x}$得到预测向量y
    4) 从y中挑出与Ground Truth 对应的值
    5) 计算其反余弦得到角度
    6) 角度加上m
    7) 得到挑出从y中挑出与Ground Truth 对应的值所在位置的独热码
    8) 将$cos(\theta + m)$ 通过独热码放回原来的位置
    9) 对所有值乘上固定值

# demo：============================
import math
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

class ArcMarginProduct(nn.Module):
    """
    Implement of large margin arc distance
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin cos(theda + m)
    """
    def __init__(self, input_feature, out_feature, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct).__init__()
        self.input_feature = input_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        # 初始化权重
        self.weight = Parameter(torch.FloatTensor(out_feature, input_feature))
        nn.init.xavier_normal(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # y = x * W^T + b
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # 将cos(theta+m)更新到tensor相应的位置中
        one_hot = torch.zeros(cosine.size(), device="cuda")
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = self.s
        return output