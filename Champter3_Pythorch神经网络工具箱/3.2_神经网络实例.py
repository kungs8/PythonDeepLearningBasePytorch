# -*- encoding: utf-8 -*-
'''
@File    :   3.2_神经网络实例.py
@Time    :   2020/06/13 11:05:59
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 主要步骤
# 1. 利用Pytorch内置函数mnist下载数据
# 2. 利用torchvision对数据进行预处理，调用torch.utils建立壹给数据迭代器
# 3. 可视化源数据
# 4. 利用nn工具箱构建神经网络模型(使用两个隐含层，每层激活函数为ReLU，最后使用torch.max(out, 1)找出张量out最大值对应索引作为预测值)
# 5. 实例化模型，并定义损失函数及优化器
# 6. 训练模型
# 7. 可视化结果

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 3.2.2 准备数据
# 1) 导入必要的模块
import numpy as np
import torch
# 导入PyTorch内置的mnist数据
from torchvision.datasets import mnist
# 导入预处理模块
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# 导入nn及优化器
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt

# 2) 定义一i写超参数
train_batch_size = 64
test_batch_size = 128
learning_rate = 0.01
num_epochs = 20
lr = 0.01
momentum = 0.5  # 动量参数，值域(0, 1)

# 3) 下载数据并对数据及可能性预处理
# 定义预处理函数，这些预处理依次放在Compose函数中
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
# 下载数据
train_dataset = mnist.MNIST("./data", train=True, transform=transform, download=True)
test_dataset = mnist.MNIST("./data", train=False, transform=transform)

# dataloader是一个可迭代对象，可以像迭代器一样使用
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# 说明
# ① tansforms.Compose可以把一些转换函数组合在一起
# ② Normalize([0.5], [0.5])对张量进行归一化，这里两个0.5分别表示对张量进行归一化的全局平均值和方差。
#     因为图像是灰色的只有一个通道，如果有多个通道，需要有多个数字(eg:3通道，Normalize([m1, m2, m3], [n1, n2, n3]))
# ③ download参数控制是否需要下载，如果./data 目录下已有MNIST，可选择False
# ④ 用DataLoader得到生成器，这可节省内存
# ⑤ torchvision及data的使用见第4章

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 3.2.3 可视化源数据
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
for i in range(6):
    plt.scatter(2, 3, i+1)
    plt.tight_layout()  # 自动调整子图参数
    plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
    plt.title("Ground Truth:{}".format(example_targets[i]))
    plt.xticks()
    plt.yticks()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 3.2.4 构建模型
# 1) 构建网络
class Net(nn.Module):
    """
    使用sequential构建网络，Sequential()函数的功能是将网络的层组合到一起
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# 2) 实例化网络
# 检测是否有可用的GPU, 有则使用，否则使用CPU
device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
# 实例化网络
model = Net(in_dim=28 * 28, n_hidden_1=300, n_hidden_2=100, out_dim=10)
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 3.2.5 训练模型
# 1) 训练模型
# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    model.train()
    
    # 动态修改参数学习率
    if epoch%5 == 0:
        optimizer.param_groups[0]['lr']*=0.1
    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)
        img = img.view(img.size(0), -1)
        # 前向传播
        out = model(img)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct/img.shape[0]
        train_acc += acc
    losses.append(train_loss/len(train_loader))
    acces.append(train_acc/len(train_loader))
    # 在测试集上检验结果
    eval_loss = 0
    eval_acc = 0
    # 将模型改为预测模型
    model.eval()
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        img = img.view(img.size(0), -1)
        out = model(img)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred== label).sum().item()
        acc = num_correct/img.shape[0]
        eval_acc +=acc
    eval_losses.append(eval_loss/len(test_loader))
    eval_acces.append(eval_acc/len(test_loader))
    print("""epoch:{}, Train loss:{:.4f}, Train Acc:{:.4f},
        Test loss:{:.4f}, Test Acc:{:.4f},
        """.format(
            epoch, train_loss/len(train_loader), train_acc/len(train_loader),
            eval_loss/len(test_loader), eval_acc/len(test_loader)
            ))

# 2) 可视化训练及测试损失值
plt.title("trainloss")
plt.plot(np.arange(len(losses)), losses)
plt.legend(["Train Loss"], loc="upper right")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------------------------------------