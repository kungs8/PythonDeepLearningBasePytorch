# -*- encoding: utf-8 -*-
'''
@Time    :   2020/6/26:下午11:23
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''

# ---------------------------------------------------------------------------------------------------------------------
# utils.data 包括 Dataset 和 DataLoader。torch.utils.data.Dataset 为抽象类。自定义数据集需要继承这个类，并实现两个函数:
#     - __len__: 提供数据的大小(size)
#     - __getitem__: 通过给的索引获取数据和标签。
# __getitem__ 一次只能获取一个数据，索引需要通过 torch.utils.data.DataLoader 来定义一个新的迭代器，实现 batch 获取。
# 下面定义一个简单的数据集，然后通过具体使用 Dataset 及 DataLoader。

# ---------------------------------------------------------------------------------------------------------------------
# 1) 导入需要的模块
import torch
from torch.utils import data
import numpy as np

# 2) 定义获取数据集的类
# 该类继承基类Dataset, 自定义一个数据集及对应标签
class TestDataset(data.Dataset):  # 继承Dataset
    def __init__(self):
        self.Data = np.asarray([[1, 2], [3, 4], [2, 1], [3, 4], [4, 5]])  # 一些由2维向量表示的数据集
        self.Label = np.asarray([0, 1, 0, 1, 2])  # 这是数据集对应的标签

    def __getitem__(self, index):
        # 把numpy转换为Tensor
        txt = torch.from_numpy(self.Data[index])
        label = torch.tensor(self.Label[index])
        return txt, label

    def __len__(self):
        return len(self.Data)

# 3) 获取数据集中数据
Test = TestDataset()
print(Test[2])  # 相当于调用__getitem__(2)
print(Test.__len__())

# 以上数据以tuple返回，每次只返回一个样本。实际上，Dataset只负责数据的抽取，调用一次__getitem__只返回一个样本。
# 如果希望批量处理(batch)，还要同时进行shuffle和并行加速等操作，可选择DataLoader。
# DataLoader等格式为：
# data.DataLoader(dataset=dataset,  # 加载的数据集
#                 batch_size=1,  # 批大小
#                 shuffle=False,  # 是否将数据打乱
#                 sampler=None,  # 样本抽样
#                 batch_sampler=None,
#                 num_workers=0,  # 使用多进程加载的进程数，0代表不使用多进程
#                 collate_fn=<function default_collate at 0x7f108ee01620>,  # 如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可
#                 pin_memory=False,  # 是否将数据保存在pin_memory中的数据转到GPU会快一些
#                 drop_last=False,  # dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
#                 timeout=0,
#                 worker_init_fn=None)

test_loader = data.DataLoader(dataset=Test,
                              batch_size=2,
                              shuffle=False,
                              num_workers=2)
for i, traindata in enumerate(test_loader):
    Data, Label = traindata
    print("""i:{} data:{} Label:{}""".format(i,
                               Data,
                               Label))

# 这个结果可以看出，这是批量读取。
# 可以像使用迭代器一样使用它，比如对它进行循环操作。
# 不过由于它不是迭代器，可以通过iter命令将其转换为迭代器
dataiter = iter(test_loader)
imgs, labels = next(dataiter)

# 一般用data.Dataset处理同一个目录下的数据。
# 如果数据在不同目录下，因为不同的目录代表不同类别(这种情况比较普遍)，使用data.Dataset来处理就很不方便。
# 不过，使用PyTorch另一种可视化数据处理工具(即torchvision)就非常方便，不但可以自动获取标签，还提供很多数据预处理、数据增强等转换函数。