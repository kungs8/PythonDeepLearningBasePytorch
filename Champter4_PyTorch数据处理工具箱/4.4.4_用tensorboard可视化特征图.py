# -*- encoding: utf-8 -*-
'''
@Time    :   2020/7/9:下午3:34
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''

# 利用tensorboardX对特征图进行可视化，不同卷基层的特征图是抽取程度是不一样的。
# x 从 cifair10数据集获取
import torch
from torch.utils import data

import torchvision
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.nn import functional as F
from torchvision.transforms import transforms

def main(data):
    # 把一些转换函数组合起来
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 对张量进行归一化，3个通道的平均值和方差(eg:Normalize([均值1, 均值2, 均值3], [方差1, 方差2, 方差3]))
    )
    # 下载数据
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
    testloader = data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2)

    # 图像数据的类型
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    # 检测是否有可用的GPU, 有则使用，否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # 定义CNNNet网络
    class CNNNet(nn.Module):
        def __init__(self):
            super(CNNNet, self).__init__()
            # in_chanels:输入数据的通道数,例RGB图片通道数为3; out_channel:输出数据的通道数,这个根据模型调整; kernel_size:卷积核尺寸; stride:步长
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(1296, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = x.view(-1, 36*6*6)
            x = F.relu(self.fc2(F.relu(self.fc1(x))))
            return x


    net = CNNNet()
    net = net.to(device)

    # 学习率
    lr = 0.0001
    momentum = 0.9  # 动量参数，值域(0, 1)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    # 初始化参数
    for m in net.modules():
        # 判断一个对象是否是一个已知的类型
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight)  # 正态分布
            nn.init.xavier_normal_(m.weight)  # xavier正态分布,xavier一般用于激活函数S型(eg:sigmoid、tanh)的权重初始化
            nn.init.kaiming_normal_(m.weight)  # 卷积层参数初始化，kaiming更适合激活函数为ReLU类的权重初始化
            nn.init.constant_(tensor=m.bias, val=0)  # 用值val填充向量
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight)  # 全连接层参数初始化

    # 训练模型
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(iterable=trainloader, start=0):  # iterable: 要遍历的数据; start:编号开始的值
            # 获取训练数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 权重参数梯度清零
            optimizer.zero_grad()

            # 正向及反向传播
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 显示损失值
            running_loss += loss.item()
            if i % 2000 == 1999:
                print("epoch:{}, pre_batches:{}, loss:{:.3f}".format(epoch+1, i+1, running_loss/2000))
                running_loss = 0.0
    print("Fished Training!")

    # 增加视图
    writer = SummaryWriter(log_dir="./res_data/logs", comment="feature map")

    for i, data in enumerate(trainloader):
        # 获取训练数据
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        x = inputs[0].unsqueeze(0)  # unsqueeze(0):在最前面增加一维(eg:(2,3) --> (1,2,3))
        break

    img_grid = vutils.make_grid(x, normalize=True, scale_each=True, nrow=2)

    net.eval()

    for name, layer in net._modules.items():
        # 为fc层处理x
        x = x.view(x.size(0), -1) if "fc" in name else x
        print(x.size())

        x = layer(x)
        print(f'{name}')

        # 查看卷基层的特征图
        if "layer" in name or "conv" in name:
            x1 = x.transpose(0, 1)  # C, B, H, W --> B, C, H, W
            img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=4)

            # normalize进行归一化处理
            writer.add_image(f"{name}_feature_maps", img_grid, global_step=0)
    writer.close()

if __name__ == '__main__':
    main(data)