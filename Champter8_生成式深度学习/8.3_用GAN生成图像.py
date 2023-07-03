# -*- encoding: utf-8 -*-
'''
@File       : 8.3_用GAN生成图像.py
@Time       : 2023/06/27 16:10:23
@Version    : 1.0
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Copyright  : 侵权必究
'''

# import some packages
# --------------------
# 这里弱化了网络和数据集的复杂度。数据集为 MNIST、网络用全连接层。
import os
import torch
import torchvision
from torch import nn
from torchvision import transforms
import torch.utils.data
from torchvision.utils import save_image
from matplotlib import image as mping
import matplotlib.pyplot as plt

# 设备配置
# torch.cuda.set_device(1)  # 这句话用来设置pytorch在哪块GPU上运行，这里假设使用序号为1到这块GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义一些超参数
latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 200
batch_size = 100
sample_dir = "./temp/gan_samples"

# 在当前目录，创建不存在的目录 sample_dir
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Image processing
trans = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)
# MNIST dataset
mnist = torchvision.datasets.MNIST(root="./data", train=True, transform=trans, download=False)
# data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)


# 1. 判别器
# 定义判别器网络结构，这里使用LeakyReLU为激活函数，输出一个节点并经过Sigmoid后输出，用于真假二分类。
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
)

# 2. 生成器
# 生成器与AVE的生成器类似，不同的地方是输出为nn.tanh,使用nn.tanh将使数据分布在[-1, 1]之间。其输入是潜在空间的向量z，输出维度与真图像相同。
# 构建生成器，这个相当于AVE中的解码器
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
)

# 把判别器和生成器迁移到GPU上
D = D.to(device)
G = G.to(device)

# 定义判别器的损失函数交叉熵及优化器
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)

# Clamp 函数x限制在区间[min, max]内
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad():
    """梯度清零"""
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

# 3. 开始训练
total_step = len(data_loader)

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device)

        # 定义图像是真或假的标签
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.ones(batch_size, 1).to(device)

        # ================================================
        #                    训练判别器                    #
        # ================================================
        # 定义判别器对真图像的损失函数
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # 定义判别器对假图像的损失函数
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # 得到判别器的总损失函数
        d_loss = d_loss_real + d_loss_fake

        # 对生成器、判别器对梯度清零
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================================
        #                    训练生成器                    #
        # ================================================
        # 定义生成器对假图像的损失函数。
        # 这里要求判别器生成的图像越来越像真图像，故损失函数中的标签改为真图像的标签，即希望生成的假图像，越来越靠近真图像。
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)
        # 对生成器、判别器对梯度清零
        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 200 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Step [{i+1}/{total_step}]",
            f"d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}",
            f"D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}")

    # 保存真图像
    if (epoch + 1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, f"real_images.png"))
    # 保存假图像
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(images), os.path.join(sample_dir, f"fake_image_{epoch+1}.png"))
print("<---train is over--->")

# 保存模型
torch.save(G.state_dict(), "G.ckpt")
torch.save(D.state_dict(), "D.ckpt")
print("<---save model is over--->")


# 4. 可视化结果
recons_path = "./temp/gan_samples/fake_images_200.png"
image = mping.imread(recons_path)
plt.imshow(image)
plt.axis("off")
plt.show()