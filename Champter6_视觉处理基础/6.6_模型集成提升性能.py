# -*- encoding: utf-8 -*-",
"""
@File      : 6.6_模型集成提升性能.py
@Time      : 2023/5/29 15:48
@Author    : yanpenggong
@Version   : 1.0
@Email     : yanpenggong@163.com
@Copyright : 侵权必究
@Project : 003.pytorchLearning
@Software: PyCharm
"""
# here put the import lib
# 为改善一项机器学习或深度学习的任务，首先想到的是从模型、数据、优化器等方面进行优化，使用方法比较方便。
# 不过有时尽管如此，但效果还不是很理想，此时，我们可尝试一下其他方法，如模型集成、迁移学习、数据增强等优化方法。
# 本节我们将介绍利用模型集成来提升任务的性能，后续章节将介绍利用迁移学习、数据增强等方法来提升任务的效果和质量。
# 模型集成是提升分类器或预测系统效果的重要方法，目前在机器学习、深度学习国际比赛中时常能看到利用模型集成取得佳绩的事例。
# 其在生产环境也是人们经常使用的方法。模型集成的原理比较简单，有点像多个盲人摸象，每个盲人只能摸到大象的一部分，但综合每人摸到的部分，就能形成一个比较完整、符合实际的图像。
# 每个盲人就像单个模型，那如果集成这些模型犹如综合这些盲人各自摸到的部分，就能得到一个强于单个模型的模型。
# 实际上模型集成也和我们通常说的集思广益、投票选举领导人等原理差不多，是1+1>2的有效方法。
# 当然，要是模型集成发挥效应，模型的多样性也是非常重要的，使用不同架构、甚至不同的学习方法是模型多样性的重要体现。如果只是改一下初始条件或调整几个参数，有时效果可能还不如单个模型。
# 具体使用时，除了要考虑各模型的差异性，还要考虑模型的性能。如果各模型性能差不多，可以取各模型预测结果的平均值；
# 如果模型性能相差较大，模型集成后的性能可能还不及单个模型，相差较大时，可以采用加权平均的方法，其中权重可以采用SLSQP、Nelder-Mead、Powell、CG、BFGS等优化算法获取。
# 接下来，通过使用PyTorch来具体实现一个模型集成的实例，希望通过这个实例，使读者对模型集成有更进一步的理解。

# 1. 使用模型
# 使用6.5节的两个模型（即CNNNet，Net）及经典模型LeNet。前面两个模型比较简单，在数据集CIFAR-10上的正确率在68%左右，这个精度是比较低的。
# 而采用模型集成的方法，可以提高到74%左右，这个提升还是比较明显的。CNNNet、Net的模型结构请参考6.5节，下列代码生成了LeNet模型。
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np
import collections
from torchvision import transforms

#定义一些超参数
BATCHSIZE = 100
DOWNLOAD_MNIST = False
EPOCHES = 20
LR = 0.001
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1296, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        #print(x.shape)
        x=x.view(-1, 36*6*6)
        x=F.relu(self.fc2(F.relu(self.fc1(x))))
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 36, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.aap=nn.AdaptiveAvgPool2d(1)
        #self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(36, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #print(x.shape)
        #x = x.view(-1, 16 * 5 * 5)
        x = self.aap(x)
        #print(x.shape)
        #x = F.relu(self.fc2(x))
        x = x.view(x.shape[0], -1)
        #print(x.shape)
        x = self.fc3(x)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net1 = CNNNet()
net2 = Net()
net3 = LeNet()

# 把3个网络模型放在一个列表里
mlps = [net1.to(device), net2.to(device), net3.to(device)]

optimizer = torch.optim.Adam([{"params": mlp.parameters()} for mlp in mlps], lr=LR)

loss_function = nn.CrossEntropyLoss()

for ep in range(EPOCHES):
    for img, label in trainloader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()  # 10个网络清除梯度
        for mlp in mlps:
            mlp.train()
            out = mlp(img)
            loss = loss_function(out, label)
            loss.backward()  # 网络们获得梯度
        optimizer.step()

    pre = []
    vote_correct = 0
    mlps_correct = [0 for i in range(len(mlps))]
    for img, label in testloader:
        img, label = img.to(device), label.to(device)
        for i, mlp in enumerate(mlps):
            mlp.eval()
            out = mlp(img)

            _, prediction = torch.max(out, 1)  # 按行取最大值
            pre_num = prediction.cpu().numpy()
            mlps_correct[i] += (pre_num == label.cpu().numpy()).sum()

            pre.append(pre_num)
        arr = np.array(pre)
        pre.clear()
        result = [collections.Counter(arr[:, i]).most_common(1)[0][0] for i in range(BATCHSIZE)]
        vote_correct += (result == label.cpu().numpy()).sum()
    print("epoch:" + str(ep) + "集成模型的正确率" + str(vote_correct / len(testloader)))

    for idx, coreect in enumerate(mlps_correct):
        print("模型" + str(idx) + "的正确率为：" + str(coreect / len(testloader)))