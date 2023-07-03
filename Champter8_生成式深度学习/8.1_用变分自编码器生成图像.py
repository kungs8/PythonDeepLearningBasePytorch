# -*- encoding: utf-8 -*-",
"""
@File      : 8.1_用变分自编码器生成图像.py
@Time      : 2023/5/30 10:57
@Author    : yanpenggong
@Version   : 1.0
@Email     : yanpenggong@163.com
@Copyright : 侵权必究
@Project : 003.pytorchLearning
@Software: PyCharm
"""
# here put the import lib
# 深度学习不仅在于其强大的学习能力，更在于它的创新能力。我们通过构建判别模型来提升模型的学习能力，通过构建生成模型来发挥其创新能力。
# 判别模型通常利用训练样本训练模型，然后利用该模型，对新样本x，进行判别或预测。而生成模型正好反过来，根据一些规则y，来生成新样本x。
# 生成式模型很多，本章主要介绍常用的两种：变分自动编码器(VAE)和生成式对抗网络（GAN）及其变种。
# 虽然两者都是生成模型，并且通过各自的生成能力展现其强大的创新能力，但他们在具体实现上有所不同。
# GAN是基于博弈论，目的是找到达到纳什均衡的判别器网络和生成器网络。
# 而VAE基本根植贝叶斯推理，其目标是潜在地建模，从模型中采样新的数据。
#
# 变分自编码器是自编码器的改进版本，自编码器是一种无监督学习，但它无法产生新的内容，变分自编码器对其潜在空间进行拓展，使其满足正态分布，情况就大不一样了。

# 1. 自编码器
# 自编码器是通过对输入X进行编码后得到一个低维的向量z，然后根据这个向量还原出输入X。
# 通过对比X与 的误差，再利用神经网络去训练使得误差逐渐减小，从而达到非监督学习的目的。图8-1为自编码器的架构图。
# 自编码器因不能随意产生合理的潜在变量，从而导致它无法产生新的内容。
# 因为潜在变量Z都是编码器从原始图片中产生的。为解决这一问题，研究人员对潜在空间Z（潜在变量对应的空间）增加一些约束，使Z满足正态分布，由此就出现了VAE模型，VAE对编码器添加约束，就是强迫它产生服从单位正态分布的潜在变量。
# 正是这种约束，把VAE和自编码器区分开来。

# 2. 变分自编码器
# 变分自编码器关键一点就是增加一个对潜在空间Z的正态分布约束，如何确定这个正态分布就成主要目标，我们知道要确定正态分布，只要确定其两个参数均值u和标准差σ。
# 那么如何确定u、σ？
# 用一般的方法或估计比较麻烦效果也不好，研究人员发现用神经网络去拟合，简单效果也不错。
#  $$Z=mu+exp(log_var)*sigma$$
# Z是从潜在空间抽取的一个向量，Z通过解码器生产一个样本$\bar{X}$),这是某模块的功能。
# 这里ε是随机采样的，这就可保证潜在空间的连续性、良好的结构性。而这些特性使得潜在空间的每个方向都表示数据中有意义的变化方向。
# 以上这些步骤构成整个网络的前向传播过程，那反向传播应如何进行？要确定反向传播就会涉及损失函数，损失函数是衡量模型优劣的主要指标。这里我们需要从以下两个方面进行衡量。
#     1）生成的新图像与原图像的相似度；
#     2）隐含空间的分布与正态分布的相似度。
# 度量图像的相似度一般采用交叉熵（如nn.BCELoss），度量两个分布的相似度一般采用KL散度（Kullback-Leibler divergence）。这两个度量的和构成了整个模型的损失函数。
# 以下是损失函数的具体代码，AVE损失函数的推导过程，有兴趣的读者可参考原论文：https://arxiv.org/pdf/1606.05908.pdf。
# # 定义重构损失函数及KL散度
# reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
# kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
# # 两者相加得总损失
# loss = reconst_loss + kl_div

# =============================================================================
# 3. 用变分自编码器生成图像
# 用PyTorch实现AVE。数据集采用MNIST。
# 3.1 导入一些包
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.utils.data
from torchvision.utils import save_image

# 3.2 定义一些超参数
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 30
batch_size = 128
learning_rate = 0.001
# 在当前目录创建不存在的目录`temp/ave_samples`
sample_dir = "./temp/ave_samples"
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# 3.3 对数据集进行预处理(eg: 转换Tensor，把数据集转化为循环、可批量加载的数据集)
dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
# 数据加载
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# 3.4 构建AVE模型，主要由Encode和Decode两部分组成
# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, img_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(img_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, img_size)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, log_var):
        """
        用mu, log_var 生成一个潜在空间点z, mu, log_var 为两个统计参数。
        假设这个分布能生成图像
        Args:
            mu (_type_): _description_
            log_var (_type_): _description_

        Returns:
            _type_: _description_
        """
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

# 3.5 选择GPU及优化器
# 设置Pytorch在哪块GPU上运行，这里开设使用序号为1的这块GPU
# torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 3.6 训练模型，同时保持原图像与随机生成的图像
for epoch in range(num_epochs):
    model.train()
    for i, (x, _) in enumerate(data_loader):
        # 前向传播
        model.zero_grad()
        x = x.to(device).view(-1, image_size)
        x_reconst, mu, log_var = model(x)

        # 计算重构损失和KL散度
        # 基于KL散度，见VAE文章中的Appendix B 或 http://yunjey47.tistory.com/43
        reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # 反向传播及优化器
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Step [{i+1}/{len(data_loader)}], Reconst Loss: {reconst_loss.item():.4f}, KL Div: {kl_div.item():.4f}")
    
    with torch.no_grad():
        # 保存采样图像，即潜在向量Z通过解码器生成的新图像
        z = torch.randn(batch_size, z_dim).to(device)
        out = model.decode(z).view(-1, 1, 28, 28)
        save_image(out, os.path.join(sample_dir, f"sampled-{epoch+1}.png"))
        # 保存重构图像，即原图像通过解码器生成的图像
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, f"reconst-{epoch+1}.png"))