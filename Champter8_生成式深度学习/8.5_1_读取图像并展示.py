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

    # 7. 可视化损失值
    # 记录判别器、生成器的损失值代码如下:
    # 配置模型存储信息
    # writer = SummaryWriter(log_dir="logs")
    # writer.add_scalars("scalars", {"g_loss": g_loss, "d_loss": d_loss}, step)
    # 使用TensorBoard查看SummaryWriter保存的日志文件的步骤:
    # 1) 安装TensorBoard：在命令行中输入pipinstalltensorboard，安装TensorBoard工具。
    # 2) 启动 TensorBoard：在命令行中输入`tensorboard --logdir=/path/to/logs`，
    #     其中 /path/to/logs 是您保存日志文件的路径。
    #     执行该命令后，TensorBoard 将在默认端口（6006）上启动一个 Web 服务器。
    # 3) 打开 TensorBoard：在浏览器中输入 http://localhost:6006，打开 TensorBoard 界面。
    # 4) 查看日志文件：在 TensorBoard 界面中，您可以选择要查看的日志文件，例如训练损失、准确率、梯度等。
    #     可以使用各种可视化工具，例如曲线图、直方图、散点图等，来分析这些数据。



if __name__ == '__main__':
    main()
