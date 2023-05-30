# -*- encoding: utf-8 -*-",
"""
@File      : 6.2_卷积层.py
@Time      : 2023/5/23 13:40
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

6.2.2 步幅(stride)
小窗口（实际上就是卷积核或过滤器）在左边窗口中每次移动的格数（无论是自左向右移动，或自上向下移动）称为步幅（strides），在图像中就是跳过的像素个数。上面小窗口每次只移动一格，故参数strides=1。这个参数也可以是2或3等数。如果是2，每次移动时就跳2格或2个像素。

6.2.3 填充(padding)
根据是否扩展Padding又分为Same、Valid。
采用Same方式时，对图片扩展并补0；
采用Valid方式时，不对图片进行扩展。
那如何选择呢？在实际训练过程中，一般选择Same方式，使用Same不会丢失信息。
设补0的圈数为p，输入数据大小为n，过滤器大小为f，步幅大小为s，则有：p=(f-1)/2
卷积后的大小为：(n+2p-f)/s + 1

6.2.4 多通道上的卷积
由于输入数据、卷积核都是单个，因此在图形的角度来说都是灰色的，并没有考虑彩色图片情况。
但在实际应用中，输入数据往往是多通道的，如彩色图片就3通道，即R、G、B通道。
对于3通道的情况应如何卷积呢？3通道图片的卷积运算与单通道图片的卷积运算基本一致，对于3通道的RGB图片，其对应的滤波器算子同样也是3通道的。
例如一个图片是6×6×3，分别表示图片的高度（Height）、宽度（Weight）和通道（Channel）。
过程是将每个单通道（R，G，B）与对应的filter进行卷积运算求和，然后再将3通道的和相加，得到输出图片的一个像素值。

6.2.5 激活函数
卷积神经网络与标准的神经网络类似，为保证其非线性，也需要使用激活函数，即在卷积运算后，把输出值另加偏移量，输入到激活函数，然后作为下一层的输入。
常用的激活函数有：nn.Sigmoid、nn.ReLU、nnLeakyReLU、nn.Tanh等。

6.2.6 卷积函数
卷积函数是构建神经网络的重要支架，通常PyTorch的卷积运算是通过nn.Conv2d来完成的。下面先介绍nn.Conv2d的参数，以及如何计算输出的形状（Shape）。
1. nn.Conv2d函数
torch.nn.Conv2d( in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=T)
主要参数说明：
    - in_channels(int): 输入信号的通道
    - out_channels(int): 卷积产生的通道
    - kernel_size(int or tuple): 卷积核的尺寸
    - stride(int or tuple, optional): 卷积步长
    - padding(int or tuple, optional): 输入的每一条边补充0的层数
    - dilation(int or tuple, optional): 卷积核元素之间的间距
    - groups(int, optional): 控制输入和输出之间的连接。
        - group=1, 输出是所有的输入的卷积
        - group=2, 此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出的一半，随后将这两个输出连接起来。
    - bias(bool, optional): 如果bias=True，添加偏置。其中参数kernel_size、stride、padding、dilation也可以是一个int的数据，
        此时卷积height和width值相同；也可以是一个tuple数组，tuple的第一维度表示height的数值，tuple的第二维度表示width的数值。
2. 输出形状
- Input(N, C_in, H_in, W_in)
- Output(N, C_out, H_out, W_out)
    H_out = (H_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1)-1)/stride[0] + 1
    W_out = (W_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1)-1)/stride[1] + 1
- weight:(out_channels, in_channels/groups, kernel_size[0], kernel_size[1])
    - 当groups=1:
        conv = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=1, groups=1)
        conv.weight.data.size()  # torch.size([12, 6, 1, 1])
    - 当groups=2:
        conv = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=1, groups=2)
        conv.weight.data.size()  # torch.size([12, 3, 1, 1])
    - 当groups=3:
        conv = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=1, groups=3)
        conv.weight.data.size()  # torch.size([12, 2, 1, 1])
    in_channels/groups 必须是整数，否则报错。

6.2.7 转置卷积
转置卷积（Transposed Convolution）在一些文献中也称为反卷积（Deconvolution）或部分跨越卷积（Fractionally-Strided Convolution）。何为转置卷积，它与卷积又有哪些不同？
通过卷积的正向传播的图像一般越来越小，记为下采样（Downsampled）。卷积的方向传播实际上就是一种转置卷积，它是上采样（Up-Sampling）。
我们先简单回顾卷积的正向传播是如何运算的，假设卷积操作的相关参数为：输入大小为4，卷积核大小为3，步幅为2，填充为0，即（n=4,f=3,s=1,p=0），根据式（6-2）可知，输出o=2。
卷积的反向传播算法：假设损失函数为L，则反向传播时，对L关系的求导，利用链式法则得到。
二维转置卷积的格式：
torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, out_channels=0, groups=1)