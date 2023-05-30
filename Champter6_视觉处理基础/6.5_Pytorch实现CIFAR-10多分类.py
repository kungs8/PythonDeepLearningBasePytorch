# -*- encoding: utf-8 -*-",
"""
@File      : 6.5_Pytorch实现CIFAR-10多分类.py
@Time      : 2023/5/24 10:33
@Author    : yanpenggong
@Version   : 1.0
@Email     : yanpenggong@163.com
@Copyright : 侵权必究
@Project : 003.pytorchLearning
@Software: PyCharm
"""
# 基于数据集CIFAR-10，利用卷积神经网络进行分类。
# 1. 数据集说明
# CIFAR-10数据集由10个类的60000个32×32彩色图像组成，每个类有6000个图像。有50000个训练图像和10000个测试图像。
# 数据集分为5个训练批次和1个测试批次，每个批次有10000个图像。测试批次包含来自每个类别的恰好1000个随机选择的图像。
# 训练批次以随机顺序包含剩余图像，但由于一些训练批次可能包含来自一个类别的图像比另一个更多，因此总体来说，5个训练集之和包含来自每个类的正好5000张图像。
# 这10类都是彼此独立的，不会出现重叠，即这是多分类单标签问题。

# here put the import lib
# 2.1. 导入库及下载数据
import torch
import torchvision
from torchvision import transforms
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2)
classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# 2.2 随机查看部分数据
import matplotlib.pyplot as plt
import numpy as np
# 显示图像
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# 随机获取部分训练数据
dataiter = iter(trainloader)
images, labels = next(dataiter)
# 显示图像
imshow(torchvision.utils.make_grid(images))
# 打印标签
print(" ".join([f"{classes[labels[j]]}" for j in range(4)]))

# 3.1. 构建网络
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
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

# 3.2 查看网络结构
# 显示网络中定义了哪些层
print(net)

# 3.3 查看网络中的前几层
# 取模型中的前4层
layer4 = nn.Sequential(*list(net.children())[:4])
print(f"<---layer4--->:\n{layer4}")

# 3.4 初始化参数
for m in net.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight)
        nn.init.xavier_normal_(m.weight)
        nn.init.kaiming_normal_(m.weight)  # 卷积层参数初始化
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight)  # 全连接层参数初始化

# 4. 训练模型
# 4.1 选择优化器
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 4.2 训练模型
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
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
            print(f"[{epoch+1}, {i+1:5d}] loss: {running_loss/2000:.3f}")
            running_loss = 0.0
print("Finished Training")

# 5. 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images. labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy of the network on the 10000 test images: {100*correct/total}%")

# 6. 采用全剧平均池化
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(16*5*5, 120)
        # 使用全局平均池化层
        self.app = nn.AdaptiveAvgPool2d(1)
        self.fc3 = nn.Linear(36, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.app(x)
        x = x.view(x.shape[0], -1)
        x = self.fc3(x)
        return x
net = CNNNet()
net = net.to(device)
# 循环同样的次数，其精度达到63%左右，但其使用的参数比没使用全局池化层的网络少很多。
# 前者只用了16022个参数，而后者用了173742个参数，是前者的10倍多。
# 这个网络比较简单，如果遇到复杂的网络，这个差距将更大。
# 具体查看参数总量的语句如下。由此可见，使用全局平均池化层确实能减少很多参数，而且在减少参数的同时，其泛化能力也比较好。
# 不过，它收敛速度比较慢，这或许是它的一个不足。不过这个不足可以通过增加循环次数来弥补。
# 使用带全局平均池化层的网络，使用的参数总量为：
print(f"net_gvp have {sum(x.numel() for x in net.parameters())} parameters in total")
# et_gvp have 16022 paramerters in total
# 不使用全局平均池化层的网络，使用的参数总量为：
# net have 173742 paramerters in total

# 7. 像Keras一样显示各层参数
# 用Keras显示一个模型参数及其结构非常方便，结果详细且规整。
# 当然，PyTorch也可以显示模型参数，但结果不是很理想。这里介绍一种显示各层参数的方法，其结果类似Keras的展示结果。
# 7.1 先定义汇总各层网络参数的函数
import collections
import torch
def paras_summary(input_size, model):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = collections.OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = -1
            summary[m_key]["output_shape"] = list(output[0].size())
            summary[m_key]["output_shape"][0] = -1
            params = 0
            if hasattr(module, "weight"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                if module.weight.requires_grad:
                    summary[m_key]["trainable"] = True
                else:
                    summary[m_key]["trainable"] = False
        if (not isinstance(module, nn.Sequential)) and \
           (not isinstance(module, nn.ModuleList)) and \
           (not (module == model)):
            hooks.append(module.register_forward_hook(hook))
    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [torch.randn(1, *in_size) for in_size in input_size]
    else:
        x = torch.randn(1, *input_size)
    # create properties
    summary = collections.OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()
    return summary

# 7.2 确定输入及实例化模型
net = CNNNet()
input_size = [3, 32, 32]
summary_infos = paras_summary(input_size, net)
print(f"<---summary_infos--->:\n{summary_infos}")