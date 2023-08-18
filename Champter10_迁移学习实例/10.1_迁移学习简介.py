# -*- encoding: utf-8 -*-
"""
@File       : 10.1_迁移学习简介.py
@Time       : 2023/8/16 18:14
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
深度学习一般需要大数据、深网络，但有时很难同时获取这些条件。还是想获取一个高性能的模型：迁移学习(Transfer Learning)
迁移学习在计算机视觉任务和自然语言处理任务中经常使用，这些模型往往需要大数据、复杂的网络结构。
如果使用迁移学习，可将预训练的模型作为新模型的起点，这些预训练的模型在开发神经网络的时候已经在大数据集上训练好、模型设计也比较好，这样的模型通用性也比较好。
如果要解决的问题与这些模型相关性较强，那么使用这些预训练模型，将大大地提升模型的性能和泛化能力。
这里主要是使用迁移学习来加速训练过程，提升深度模型的性能。

1. 迁移学习简介
    考虑到训练词向量模型一般需要大量的数据，而且耗时比较长。为节省时间、提高效率，本实例采用迁移学习方法，即直接利用训练好的词向量模型作为输入数据。
    迁移学习：
        是一种机器学习，简单来说，就是把任务A开发的模型作为初始点，重新使用在任务B中。比如，A任务是识别图像中车辆，B任务是可以识别卡车、识别轿车、识别公交车等。
        合理使用迁移学习，可以避免针对每个目标任务单独训练模型，从而极大地节约计算资源。

    在计算机视觉任务和自然语言处理任务中，将预训练好的模型作为新模型的起点是一种常用方法，通常预训练这些模型，往往需要消耗大量时间和巨大的计算资源。
    迁移学习就是把预训练好的模型迁移到新的任务上。

    在神经网络迁移中，主要有两个应用场景：
        - 特征提取(Feature Extraction)
            冻结除最终完全连接层之外的所有网络的权重。最后一个全连接层被替换为具有随机权重的新层，并且仅训练该层。
        - 微调(Fine Tuning)
            使用预训练网络初始化网络，而不是随机初始化，用新数据训练部分或整个网络。