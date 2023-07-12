# -*- encoding: utf-8 -*-
"""
@File       : 8.5_1_读取图像并展示.py.py
@Time       : 2023/7/12 16:31
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import torch
# import some packages
# --------------------
import torch
from torch import nn
from torchvision.utils import make_grid
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


class Generator(nn.Module):
    """定义生成器(Generator)和前向传播函数

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)


def main():
    # 展示real_images
    resconsPath = "./temp/cgan_sample/real_images.png"
    img = mpimg.imread(resconsPath)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    # 展示fake_images
    resconsPath = "./temp/cgan_sample/fake_images_20.png"
    img = mpimg.imread(resconsPath)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    # 5. CGAN可视化
    # 利用网格(10 * 10)的形式显示指定条件下生成的图像
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.randn(100, 100).to(device)
    labels = torch.LongTensor([i for i in range(10) for _ in range(10)]).to(device)
    G = Generator().to(device)
    images = G(z, labels).unsqueeze(1)
    grid = make_grid(images, nrow=10, normalize=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid.permute(1, 2, 0).detach().cpu().numpy(), cmap="binary")
    ax.axis("off")
    plt.show()

    # 6. 查看指定标签的数据
    def generate_digit(generate, digit):
        """可视化指定单个数字条件下生成的数字"""
        z = torch.randn(1, 100).to(device)
        label = torch.LongTensor([digit]).to(device)
        img = generate(z, label).detach().cpu()
        img = 0.5 * img + 0.5
        return transforms.ToPILImage()(img)

    img = generate_digit(G, 8)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
