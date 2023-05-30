# -*- encoding: utf-8 -*-",
"""
@File      : 6.3_池化层.py
@Time      : 2023/5/23 16:32
@Author    : yanpenggong
@Version   : 1.0
@Email     : yanpenggong@163.com
@Copyright : 侵权必究
@Project : 003.pytorchLearning
@Software: PyCharm
"""
# here put the import lib
import torch
from torch import nn

池化（Pooling）又称下采样，通过卷积层获得图像的特征后，理论上可以直接使用这些特征训练分类器（如Softmax）。
但是，这样做将面临巨大的计算量挑战，而且容易产生过拟合的现象。为了进一步降低网络训练参数及模型的过拟合程度，就要对卷积层进行池化(Pooling)处理。
常用的池化方式通常有3种。
·最大池化（Max Pooling）：选择Pooling窗口中的最大值作为采样值。
·均值池化（Mean Pooling）：将Pooling窗口中的所有值相加取平均，以平均值作为采样值。
·全局最大（或均值）池化：与平常最大或最小池化相对而言，全局池化是对整个特征图的池化而不是在移动窗口范围内的池化。

池化层在CNN中可用来减小尺寸，提高运算速度及减小噪声影响，让各特征更具有健壮性。池化层比卷积层更简单，它没有卷积运算，只是在滤波器算子滑动区域内取最大值或平均值。
而池化的作用则体现在降采样：保留显著特征、降低特征维度，增大感受野。深度网络越往后面越能捕捉到物体的语义信息，这种语义信息是建立在较大的感受野基础上。

1. 局部池化
我们通常使用的最大或平均池化，是在特征图（Feature Map）上以窗口的形式进行滑动（类似卷积的窗口滑动），操作为取窗口内的平均值作为结果，经过操作后，特征图降采样，减少了过拟合现象。
其中在移动窗口内的池化被称为局部池化。
在PyTorch中，最大池化常使用nn.MaxPool2d，平均池化使用nn.AvgPool2d。在实际应用中，最大池化比其他池化方法更常用。它们的具体格式如下：
torch.nn.MaxPool2d(kernel_size, stride, padding, dilation=1,return_indices=False, ceil_mode=False)
参数说明：
    - kekernel_size: 池化窗口的大小，取一个4维向量，一般是[height，width]，如果两者相等，可以是一个数字，如kernel_size=3。
    - stride: 窗口在每一个维度上滑动的步长，一般也是[stride_h，stride_w]，如果两者相等，可以是一个数字，如stride=1。
    - padding: 和卷积类似
    - dilation: 卷积对输入数据的空间间隔
    - return_indices: 是否返回最大值对应的下标
    - ceil_mode: 使用一些方块代替层结构

实例：
# 池化窗口为正方形 size=3, stride=2
m1 = nn.MaxPool2d(kekernel_size=3, stride=2)
# 池化窗口为非正方形
m2 = nn.MaxPool2d(kernel_size=(3, 2), stride=(2, 1))
input = torch.randn(20, 16, 50, 32)
output = m2(input)
print(output.shape)


2.  全局池化
与局部池化相对的就是全局池化，全局池化也分最大或平均池化。所谓的全局就是针对常用的平均池化而言，平均池化会有它的filter size，比如2×2，而全局平均池化就没有size，它针对的是整张Feature Map。
下面以全局平均池化为例。
全局平均池化（Global Average Pooling，GAP），不以窗口的形式取均值，而是以特征图为单位进行均值化，即一个特征图输出一个值。

使用全局平均池化代替CNN中传统的全连接层。在使用卷积层的识别任务中，全局平均池化能够为每一个特定的类别生成一个特征图（Feature Map）。
GAP的优势在于：各个类别与Feature Map之间的联系更加直观（相比与全连接层的黑箱来说），Feature Map被转化为分类概率也更加容易，因为在GAP中没有参数需要调，所以避免了过拟合问题。GAP汇总了空间信息，因此对输入的空间转换鲁棒性更强。
所以目前卷积网络中最后几个全连接层，大都用GAP替换。
全局池化层在Keras中有对应的层，如全局最大池化层（GlobalMaxPooling2D）。
PyTorch虽然没有对应名称的池化层，但可以使用PyTorch中的自适应池化层(AdaptiveMaxPool2d(1)或nn.AdaptiveAvgPool2d(1))来实现，如何实现后续有实例介绍，这里先简单介绍自适应池化层，其一般格式为：
nn.AdaptiveMaxPool2d(output_size, return_indices=False)
实例：
# 输出大小为5*7
m = nn.AdaptiveMaxPool2d(output_size=(5, 7))
input = torch.randn(1, 64, 8, 9)
output = m(input)
# t输出大小为正方形 7*7
m = nn.AdaptiveMaxPool2d(output_size=7)
input = torch.randn(1, 64, 10, 9)
output = m(input)
# t输出大小为正方形 10*7
m = nn.AdaptiveMaxPool2d(output_size=(None, 7))
input = torch.randn(1, 64, 10, 9)
output = m(input)
# t输出大小为正方形 1*1
m = nn.AdaptiveMaxPool2d(output_size=(1))
input = torch.randn(1, 64, 10, 9)
output = m(input)
print(output.size()

Adaptive Pooling输出张量的大小都是给定的output_size。
例如输入张量大小为（1，64，8，9），设定输出大小为（5，7），通过Adaptive Pooling层，可以得到大小为（1，64，5，7）的张量。