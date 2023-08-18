# -*- encoding: utf-8 -*-
"""
@File       : 10.2_1_特征提取实例.py
@Time       : 2023/8/17 11:49
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import numpy as np
import torch
from torch import nn
import torch.utils.data
from torch.nn import functional as F
import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from datetime import datetime
import matplotlib.pyplot as plt


def load_data():
    """加载数据"""
    trans_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trans_valid = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=trans_train)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=trans_valid)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return trainloader, testloader, classes


def display_images(trainloader, classes):
    """展示下载的图像"""
    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    # 随机获取部分训练数据
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    # 显示图像
    imshow(torchvision.utils.make_grid(images[:4]))
    print(" ".join([classes[labels[j]] for j in range(4)]))


def download_pretained_model():
    """下载预训练模型"""
    # 这里将自动下载预训练模型，该模型网络架构为ResNet18，已经在ImageNet大数据集上训练好了，该数据集有1000类别。
    net = models.resnet18(pretrained=True)
    return net


def freezing_parametric_models(model):
    """冻结参数模型，在反向传播时，将不会更新"""
    for param in model.parameters():
        param.requires_grad = False


def update_out_class_n(model):
    """修改最后一层的输出类别数"""
    # 原来输出为 512*1000，现在把输出改为512*10，因为新的数据集只有10个类别
    model.fc = nn.Linear(512, 10)
    return model


def print_freeze_param_states(model):
    """查看冻结前后的参数情况"""
    total_params = sum(p.numel() for p in model.parameters())  # p.numel() 返回每个参数的元素数量
    print(f"原总参数个数: {total_params}")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"需训练参数个数: {total_trainable_params}")
    # 如果不冻结的话，需要更新的参数非常大，冻结后，只需要更新全连接层的相关参数。


def get_acc(output, label):
    """获取准确度"""
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def train(train_data, valid_data, num_epochs):
    """训练及验证模型"""
    # 下载预训练模型
    model = download_pretained_model()
    # 冻结参数模型
    freezing_parametric_models(model)
    # 获取device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 修改最后一层的输出类别数。原来输出为 512*1000，现在把输出改为512*10，因为新的数据集只有10个类别
    model.fc = nn.Linear(512, 10)
    # 查看冻结前后的参数情况
    print_freeze_param_states(model)
    model.to(device)

    # 定义损害函数及优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9)

    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        model = model.train()

        for i, (im, label) in enumerate(train_data):
            im = im.to(device)  # (bs, 3, h, w)
            label = label.to(device)  # (bs, h, w)
            # forward
            output = model(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 获取损失函数值
            train_loss += loss.item()
            # 获取准确度
            train_acc += get_acc(output, label)
            print(f"Epoch: {epoch+1}, data_i: {i}/{len(train_data)}")

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)  # divmod 函数会将时间差 time_difference 分解为小时和分钟部分，并计算商和余数
        m, s = divmod(remainder, 60)
        time_str = f"Time {h:2d}:{m:2d}:{s:2d}"

        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            model = model.eval()
            for im, label in valid_data:
                im = im.to(device)
                label = label.to(device)
                output = model(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = f"Epoch {epoch}. Train Loss: {train_loss/len(train_data)}, Train Acc: {train_acc/len(train_data)},\t" \
                        f"Valid Loss: {valid_loss/len(valid_data)}, Valid Acc: {valid_acc/len(valid_data)}, "
        else:
            epoch_str = f"Epoch {epoch}. Train Loss: {train_loss / len(train_data)}, Train Acc: {train_acc / len(train_data)}, "

        prev_time = cur_time
        print(epoch_str + time_str)


def main():
    # 训练的步数
    num_epochs = 20
    # 加载数据
    trainloader, testloader, classes = load_data()
    # 显示图像
    display_images(trainloader, classes)
    # 训练及验证模型
    train(trainloader, testloader, num_epochs)




if __name__ == '__main__':
    # 使用迁移学习中特征提取方法来实现，预训练模型采用 ResNet18网络，精度由之前(几层卷积层和全连接层)的68%提升到75%。
    main()