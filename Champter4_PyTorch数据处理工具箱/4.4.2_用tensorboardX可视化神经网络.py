# -*- encoding: utf-8 -*-
'''
@Time    :   2020/7/1:上午9:49
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''


# ----------------------------------------------------------------------------------------------------------------------
# 4.4.2 用tensorboardX可视化神经网络
# (1) 导入需要的模块
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter


# (2) 构建神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn = nn.BatchNorm2d(20)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x) + F.relu(-x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.bn(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

# (3) 把模型保存为graph
#  定义输入
input = torch.rand(32, 1, 28, 28)  # 假设输入32张1*28*28的图片
print(input)
# 实例化神经网络
model = Net()
# 将model保存为graph
with SummaryWriter(log_dir="./res_data/logs", comment="Net") as w:
    w.add_graph(model, (input, ))
