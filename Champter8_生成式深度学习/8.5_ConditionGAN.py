# -*- encoding: utf-8 -*-
'''
@File       : 8.5_ConditionGAN.py
@Time       : 2023/07/03 18:07:53
@Version    : 1.0
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Copyright  : 侵权必究
'''
import os.path

# import some packages
# --------------------
import torch
import torchvision.datasets
from torch import nn
import torch.utils.data
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

# VAE 和GAN都能基于潜在空间的随机向量z生成新图片，GAN生成的图像比VAE的更清晰，质量更好些。
# 不过生成的都是随机的，无法预先控制要生成的哪类或哪个数。
# 如果在生成新图像的同时，能加上一个目标控制那就太好了，如果希望生成某个数字，生成某个主题或类别的图像，实现按需生成的目的，这样的应用非常广泛。
# 需要就是最大的生产力，经过研究人员的不懈努力，提出一个基于条件的GAN，即：Condition GAN(CGAN)。

# 1. CGAN架构
# 在GAN这种完全无监督的方式加上一个标签或一点监督信息，使整个网络就可看成半监督模型。其基本架构与GAN类似，只要添加一个条件y即可，y就是加入的监督信息，
# 比如说MNIST数据集可以提供某个数字的标签信息，人脸生成可以提供性别、是否微笑、年龄等信息，带某个主题的图像等标签信息。
# 对生成器输入一个从潜在空间随机采样的一个向量z及一个条件y，生成一个符合该条件的图像G(z/y)。对判别器来说，输入一张图像x和条件y，输出该图像在该条件下的概率D(x/y)。
# 这只是CGAN的一个蓝图。

# 2. CGAN生成器
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


# 3. CGAN判别器
class Discriminator(nn.Module):
    """定义判别器(Discriminator) 及前向传播函数

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()


def denorm(x):
    """
    Clamp函数x限制在区间[min, max]内
    :param x:
    :return:
    """
    out = (x + 1) / 2
    return out.clamp(0, 1)


def reset_grad(d_optimizer, g_optimizer):
    """
    重置优化器函数
    :param d_optimizer:
    :param g_optimizer:
    :return:
    """
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
def main():
    # 设备配置
    # torch.cuda.set_device(1)  # 这句用来设置pytorch在哪块GPU上运行，这里假设使用序号为1的这块GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义一些超参数
    num_epochs = 2
    batch_size = 100
    sample_dir = "./temp/cgan_sample"
    ckps_dir = "./ckpts"
    if not os.path.exists(ckps_dir):
        os.makedirs(ckps_dir)
    # 配置模型存储信息
    writer = SummaryWriter(log_dir="logs")

    # 在当前目录，创建不存在的目录"temp/cgan_samples"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    # Image processing
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    # MNIST dataset
    mnist = torchvision.datasets.MNIST(root="./data", train=True, transform=trans, download=False)
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

    G = Generator().to(device)
    D = Discriminator().to(device)
    # 定义判别器的损失函数、交叉熵及优化器
    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)

    # 开始训练
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data_loader):
            step = epoch * len(data_loader) + i + 1
            images = images.to(device)
            labels = labels.to(device)
            # 定义图像是真或假的标签
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
            # ============================================================= #
            #                          训练判别器                             #
            # ============================================================= #
            # 4. CGAN损失函数
            # 定义判别器对真图像的损失函数
            real_validity = D(images, labels)
            d_loss_real = criterion(real_validity, real_labels)
            real_score = real_validity
            # 定义判别器对假图像(即由潜在空间点生成的图像)的损失函数
            z = torch.randn(batch_size, 100).to(device)
            fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
            fake_images = G(z, fake_labels)
            fake_validity = D(fake_images, fake_labels)
            d_loss_fake = criterion(fake_validity, torch.zeros(batch_size).to(device))
            fake_score = fake_images
            # CGAN总的损失值
            d_loss = d_loss_real + d_loss_fake

            # 对生成器和判别器的梯度清零
            reset_grad(d_optimizer, g_optimizer)
            d_loss.backward()
            d_optimizer.step()

            # ============================================================= #
            #                          训练生成器                             #
            # ============================================================= #
            # 定义生成器对假图像对损失函数，这里我们要求判别器生成的图像越来越像真图像，
            # 故损失函数中的标签改为真图像的标签，即希望生成的假图像，越来越靠近真图像
            z = torch.randn(batch_size, 100).to(device)
            fake_images = G(z, fake_labels)
            validity = D(fake_images, fake_labels)
            g_loss = criterion(validity, torch.ones(batch_size).to(device))
            # 对生成器、判别器的梯度清零，进行反向传播及运行生成器的优化器
            reset_grad(d_optimizer, g_optimizer)
            g_loss.backward()
            g_optimizer.step()

            writer.add_scalars("scalars", {"g_loss": g_loss, "d_loss": d_loss}, step)

            if (i + 1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], d_loss:{d_loss.item():.4f}, "
                      f"g_loss:{g_loss.item():.4f}, D(x):{real_score.mean().item():.2f}, "
                      f"D(G(z)):{fake_score.mean().item()*(-1):.2f}")

        # 保存真图像
        if (epoch + 1) == 1:
            images = images.reshape(images.size(0), 1, 28, 28)
            save_image(denorm(images), os.path.join(sample_dir, "real_images.png"), nrow=10)
        # 保存假图像
        fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        save_image(denorm(fake_images), os.path.join(sample_dir, f"fake_images_{epoch+1}.png"), nrow=10)

    # 保存模型
    torch.save(G.state_dict(), f"{ckps_dir}/G.ckpt")
    torch.save(D.state_dict(), f"{ckps_dir}/D.ckpt")


if __name__ == '__main__':
    main()