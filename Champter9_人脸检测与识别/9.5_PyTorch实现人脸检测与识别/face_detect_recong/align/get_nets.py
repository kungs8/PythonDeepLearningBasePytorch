# -*- encoding: utf-8 -*-
"""
@File       : get_nets.py
@Time       : 2023/8/11 10:52
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""

# import some packages
# --------------------
import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from torch.nn import functional as F
from pathlib import Path

FILE = Path(__file__).resolve()  # 获取本文件的全路径
ROOT = FILE.parents[0]  # 文件根路径

class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 10, 3, 1)),
            ("prelu1", nn.PReLU(10)),
            ("pool1", nn.MaxPool2d(2, 2, ceil_mode=True)),

            ("conv2", nn.Conv2d(10, 16, 3, 1)),
            ("prelu2", nn.PReLU(16)),

            ("conv3", nn.Conv2d(16, 32, 3, 1)),
            ("prelu3", nn.PReLU(32))
        ]))

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

        weights = np.load(f"{ROOT}/weights/pnet.npy", allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        :param x: a float tensor with shape [batch_size, 3, h, w]
        :return:
            b: a float tensor with shape [batch_size, 4, h', w']
            a: a float tensor with shape [batch_size, 2, h', w']
        """
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        a = F.softmax(a, dim=1)
        return b, a


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        :param x: a float tensor with shape [batch_size, c, h, w]
        :return:
            a float tensor with shape [batch_size, c*h*w]
        """
        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()
        return x.view(x.size(0), -1)


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 28, 3, 1)),
            ("prelu1", nn.PReLU(28)),
            ("pool1", nn.MaxPool2d(3, 2, ceil_mode=True)),

            ("conv2", nn.Conv2d(28, 48, 3, 1)),
            ("prelu2", nn.PReLU(48)),
            ("pool2", nn.MaxPool2d(3, 2, ceil_mode=True)),

            ("conv3", nn.Conv2d(48, 64, 2, 1)),
            ("prelu3", nn.PReLU(64)),

            ("flatten", Flatten()),
            ("conv4", nn.Linear(576, 128)),
            ("prelu4", nn.PReLU(128))
        ]))

        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

        weights = np.load(f"{ROOT}/weights/rnet.npy", allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        :param x: a float tensor with shape [batch_size, 3, h, w]
        :return:
            b: a float tensor with shape [batch_size, 4]
            a: a float tensor with shape [batch_size, 2]
        """
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)
        a = F.softmax(a, dim=1)
        return b, a

class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 232, 3, 1)),
            ("prelu1", nn.PReLU(32)),
            ("pool1", nn.MaxPool2d(3, 2, ceil_mode=True)),

            ("conv2", nn.Conv2d(32, 64, 3, 1)),
            ("prelu2", nn.PReLU(64)),
            ("pool2", nn.MaxPool2d(3, 2, ceil_mode=True)),

            ("conv3", nn.Conv2d(64, 64, 3, 1)),
            ("prelu3", nn.PReLU(64)),
            ("pool3", nn.MaxPool2d(2, 2, ceil_mode=True)),

            ("conv4", nn.Conv2d(64, 128, 2, 1)),
            ("prelu4", nn.PReLU(128)),

            ("flatten", Flatten()),
            ("conv5", nn.Linear(1152, 256)),
            ("drop5", nn.Dropout(0.25)),
            ("prelu5", nn.PReLU(256))
        ]))

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

        weights = np.load(f"{ROOT}/weights/onet.npy", allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        :param x: a float tensor with shape [batch_size, 3, h, w]
        :return:
            c: a float tensor with shape [batch_size, 10]
            b: a float tensor with shape [batch_size, 4]
            a: a float tensor with shape [batch_size, 2]
        """
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        a = F.softmax(a, dim=1)
        return c, b, a


if __name__ == '__main__':
    PNet()