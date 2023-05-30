# -*- encoding: utf-8 -*-",
"""
@File      : 6.7_使用现代经典模型提升性能.py
@Time      : 2023/5/30 09:12
@Author    : yanpenggong
@Version   : 1.0
@Email     : yanpenggong@163.com
@Copyright : 侵权必究
@Project : 003.pytorchLearning
@Software: PyCharm
"""
# here put the import lib
# 前面通过使用一些比较简单的模型对数据集CIFAR-10进行分类，精度在68%左右，然后使用模型集成的方法，同样是这些模型，但精度却提升到74%左右。
# 虽有一定提升，但结果还是不够理想。
# 精度不够很大程度与模型有关，前面我们介绍的一些现代经典网络，在大赛中都取得了不俗的成绩，说明其模型结构有很多突出的优点，所以，人们经常直接使用这些经典模型作为数据的分类器。
# 这里我们就用VGG16这个模型，来对数据集IFAR10进行分类，直接效果非常不错，精度一下子就提高到90%左右，效果非常显著。
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
cfg = {
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512,512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512,512, 512, "M", 512, 512, 512, 512, "M"],
}
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, out_channels=x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
net4 = VGG("VGG16")
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

print('==> Building model..')
mlps = [net4.to(device)]
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
    print("epoch:" + str(ep)+"集成模型的正确率"+str(vote_correct/len(testloader)))

    for idx, coreect in enumerate(mlps_correct):
        print("VGG16模型迭代" + str(ep) + "次的正确率为：" + str(coreect / len(testloader)))