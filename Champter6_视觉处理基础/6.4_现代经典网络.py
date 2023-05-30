# -*- encoding: utf-8 -*-",
"""
@File      : 6.4_现代经典网络.py
@Time      : 2023/5/23 17:16
@Author    : yanpenggong
@Version   : 1.0
@Email     : yanpenggong@163.com
@Copyright : 侵权必究
@Project : 003.pytorchLearning
@Software: PyCharm
"""
# here put the import lib
卷积神经网络发展非常迅速，应用非常广阔，所以近几年的卷积神经网络得到了长足的发展，卷积神经网络近几年发展的大致轨迹。
LeNet(卷积、池化、全连接)
    - Alexnet(ReLU、Dropout、GPU、数据增强)
        - 网络加深
            VGG16 -> VGG19                 -|
        - 网络加宽                            }两者结合(ResNet) -> Inception ResNet
            GoogleNet -> InceptionV3、V4   -|
        - 从分类到检测
            RCNN -> Fast R-CNN -> Faster R-CNN
        - 增加新功能
            FCN、GCN、SENet、CapsuleNetwork、ENAS
1998年LeCun提出了LeNet，可谓是开山鼻祖，系统地提出了卷积层、池化层、全连接层等概念。
时隔多年后，2012年Alex等提出AlexNet，提出一些训练深度网络的重要方法或技巧，如Dropout、ReLu、GPU、数据增强方法等。
此后，卷积神经网络迎来了爆炸式的发展。接下来我们将就一些经典网络架构进行说明。




1. LeNet-5模型
LeNet是卷积神经网络的大师LeCun在1998年提出的，用于解决手写数字识别的视觉任务。
自那时起，CNN最基本的架构就定下来了，即卷积层、池化层、全连接层。
（1）模型架构
    LeNet-5模型结构为输入层-卷积层-池化层-卷积层-池化层-全连接层-全连接层-输出，为串联模式，如图6-19所示。
（2）模型特点
    ·每个卷积层包含3个部分：卷积、池化和非线性激活函数。
    ·使用卷积提取空间特征。
    ·采用降采样（Subsample）的平均池化层（Average Pooling）。
    ·使用双曲正切（Tanh）的激活函数。
    ·最后用MLP作为分类器。
input(32*32) -> 6*28*28 -> 6*14*14 -> 16*10*10 -> 16*5*5 -> 120 -> 84 -> output(10)

2. AlexNet模型
AlexNet在2012年ImageNet竞赛中以超过第2名10.9个百分点的绝对优势一举夺冠，从此，深度学习和卷积神经网络如雨后春笋般得到迅速发展。
（1）模型架构
    AlexNet为8层深度网络，其中5层卷积层和3层全连接层，不计LRN层和池化层。
（2）模型特点
    ·由5层卷积和3层全连接组成，输入图像为3通道224×224大小，网络规模远大于LeNet。
    ·使用ReLU激活函数。
    ·使用Dropout，可以作为正则项防止过拟合，提升模型鲁棒性。
    ·具备一些很好的训练技巧，包括数据增广、学习率策略、Weight Decay等。

3. VGG模型
在AlexNet之后，另一个提升很大的网络是VGG，ImageNet上将Top5错误率减小到7.3%。
VGG-Nets是由牛津大学VGG（Visual Geometry Group）提出，是2014年ImageNet竞赛定位任务的第一名和分类任务的第二名。
VGG可以看成是加深版本的AlexNet.都是Conv Layer+FC layer，在当时看来这是一个非常深的网络了，层数高达16或19层。
（1）模型结构
（2）模型特点
    ·更深的网络结构：网络层数由AlexNet的8层增至16和19层，更深的网络意味着更强大的网络能力，也意味着需要更强大的计算力，不过后来硬件发展也很快，显卡运算力也在快速增长，以此助推深度学习的快速发展。
    ·使用较小的3×3的卷积核：模型中使用3×3的卷积核，因为两个3×3的感受野相当于一个5×5，同时参数量更少，之后的网络都基本遵循这个范式。

4. GoogleNet模型
VGG是增加网络的深度，但深度达到一个程度时，可能就成为瓶颈。
GoogLeNet则从另一个维度来增加网络能力，每单元有许多层并行计算，让网络更宽了。
（1）模型结构
    网络总体结构，包含多个Inception模块，为便于训练添加了两个辅助分类分支补充梯度。
（2）模型特点
    1）引入Inception结构，这是一种网中网（Network In Network）的结构。
        通过网络的水平排布，可以用较浅的网络得到较好的模型能力，并进行多特征融合，同时更容易训练。
        另外，为了减少计算量，使用了1×1卷积来先对特征通道进行降维。堆叠Inception模块就叫作Inception网络，而GoogLeNet就是一个精心设计的性能良好的Inception网络（Inception v1）的实例，即GoogLeNet是Inception v1网络的一种。
    2）采用全局平均池化层。
    将后面的全连接层全部替换为简单的全局平均池化，在最后参数会变得更少。
    而在AlexNet中最后3层的全连接层参数差不多占总参数的90%，使用大网络在宽度和深度上允许GoogleNet移除全连接层，但并不会影响到结果的精度，在ImageNet中实现93.3%的精度，而且要比VGG还快。不过，网络太深无法很好训练的问题还是没有得到解决，直到ResNet提出了Residual Connection。

5. ResNet模型
2015年，何恺明推出的ResNet在ISLVRC和COCO上超越所有选手，获得冠军。
ResNet在网络结构上做了一大创新，即采用残差网络结构，而不再是简单地堆积层数，
ResNet在卷积神经网络中提供了一个新思路。
残差网络的核心思想即：输出的是两个连续的卷积层，并且输入时绕到下一层去。
（1）模型结构
    通过引入残差，Identity恒等映射，相当于一个梯度高速通道，可以更容易地训练避免梯度消失的问题。所以，可以得到很深的网络，网络层数由GoogLeNet的22层到了ResNet的152层。
（2）模型特点
    ·层数非常深，已经超过百层。
    ·引入残差单元来解决退化问题。

6. 胶囊网络简介
2017年底，Hinton和他的团队在论文中介绍了一种全新的神经网络，即胶囊网络（CapsNet）[1]。
与当前的卷积神经网络（CNN）相比，胶囊网络具有许多优点。
目前，对胶囊网络的研究还处于起步阶段，但可能会挑战当前最先进的图像识别方法。
胶囊网络克服了卷积神经网络的一些不足：
    1）训练卷积神经网络一般需要较大数据量，而胶囊网络使用较少数据就能泛化。
    2）卷积神经网络因池化层、全连接层等丢失大量的信息，从而降低了空间位置的分辨率，而胶囊网络对很多细节的姿态信息（如对象的准确位置、旋转、厚度、倾斜度、尺寸等）能在网络里被保存。这就有效地避免嘴巴和眼睛倒挂也认为是人脸的错误。
    3）卷积神经网络不能很好地应对模糊性，但胶囊网络可以。所以，它能在非常拥挤的场景中也表现得很好。
胶囊网络是如何实现这些优点的呢？当然这主要归功于胶囊网络的一些独特算法，因为这些算法比较复杂，这里就不展开来说，我们先从其架构来说，希望通过对架构的了解，对胶囊网络有个直观的认识，胶囊网络的结构如图6-26所示。
（1）模型结构
    该架构由两个卷积层和一个全连接层组成，其中第一个为一般的卷积层，第二个卷积相当于为Capsule层做准备，并且该层的输出为向量，所以，它的维度要比一般的卷积层再高一个维度。
    最后就是通过向量的输入与路由（Routing）过程等构建出10个向量，每一个向量的长度都直接表示某个类别的概率。
（2）模型特点
    1）神经元输出为向量：
        每个胶囊给出的是输出是一组向量，不是如同传统的人工神经元是一个单独的数值（权重）。
    2）采用动态路由机制：
        为了解决这组向量向更高层的神经元传输的问题，就需要动态路由（Dynamic Routing）机制，而这是胶囊神经网络的一大创新点。
        Dynamic Routing使得胶囊神经网络可以识别图形中的多个图形，这一点也是CNN所不具备的功能。
    虽然，CapsNet在简单的数据集MNIST上表现出了很好的性能，但是在更复杂的数据集如ImageNet、CIFAR-10上，却没有这种表现。
    这是因为在图像中发现的信息过多会使胶囊脱落。
    由于胶囊网络仍然处于研究和开发阶段，并且不够可靠，现在还没有很成熟的任务。
    但是，这个概念是合理的，这个领域将会取得更多的进展，使胶囊网络标准化，以更好地完成任务。
    如果读者想进一步了解，可参考原论文（https://arxiv.org/pdf/1710.09829.pdf）