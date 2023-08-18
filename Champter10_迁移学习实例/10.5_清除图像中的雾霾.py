# -*- encoding: utf-8 -*-
"""
@File       : 10.5_清除图像中的雾霾.py
@Time       : 2023/8/18 17:08
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import glob

import numpy as np
import torch
import torchvision.utils
from torch import nn
from matplotlib.image import imread
from matplotlib import pyplot as plt
from PIL import Image

class Model(nn.Module):
    """自定义一个模型"""
    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        source = []
        source.append(x)

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        # torch.cat(tensors, dim=1)
        # - tensors：tuple| list, 待拼接的两个张量。
        # - dim：指定沿哪个维度进行拼接。在你的例子中，dim=1 表示沿着第一个维度（即列）进行拼接。
        # eg:
        #   - x2 & x3:(batch_size, num_channels, height, width) -> dim=1 -> (batch_size, x2_num_channels+x3_num_channels, height, width)
        #   - x2 & x3:(batch_size, num_channels, height, width) -> dim=2 -> (batch_size, num_channels, x2_height+x3_height, width)
        concat1 = torch.cat((x1, x2), dim=1)
        x3 = self.relu(self.e_conv3(concat1))

        concat2 = torch.cat((x2, x3), dim=1)
        x4 = self.relu(self.e_conv4(concat2))
        concat3 = torch.cat((x1, x2, x3, x4), dim=1)
        x5 = self.relu(self.e_conv5(concat3))
        clean_image = self.relu((x5 * x) - x5 + 1)
        return clean_image


def cl_img(device, net, img_path):
    """训练处理图像"""
    img = Image.open(img_path)
    img_np = (np.asarray(img) / 255.0)
    img_tensor = torch.from_numpy(img_np).float()
    img_tensor = img_tensor.permute(2, 0, 1)  # 维度重排。PIL.Image :(height, width, channels)，PyTorch :(channels, height, width)
    img_tensor = img_tensor.to(device).unsqueeze(0)  # (height, width, channels) -> (1, height, width, channels)

    # 装载预训练模型
    net.load_state_dict(torch.load("./model/clean_photo/dehazer.pth", map_location=device))

    clean_img = net.forward(img_tensor)
    torchvision.utils.save_image(torch.cat((img_tensor, clean_img), 0), f"./tmp/clean_photo/results/{img_path.split('/')[-1]}")

def main():
    # 1. 查看原来的图像
    img = imread("./data/clean_photo/test_images/shanghai01.jpg")
    plt.imshow(img)
    # 加载数据
    img_paths = glob.glob("./data/clean_photo/test_images/*")
    img_paths = [img_path for img_path in img_paths if img_path.split(".")[-1] == "jpg"]

    # 2. 定义一个神经网络
    model = Model()

    # 3. 训练模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model.to(device)
    for img_path in img_paths:
        cl_img(device, net, img_path)
        print(f"{img_path} done.")

    # 4. 查看清除雾霾后的效果
    img = imread("./tmp/clean_photo/results/shanghai01.jpg")
    plt.imshow(img)
    plt.show()



if __name__ == '__main__':
    """实例：利用一个预训练模型清除图像中的雾霾，使图像更清晰"""
    main()