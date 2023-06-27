# -*- encoding: utf-8 -*-
'''
@File       : 8.2_GAN.py
@Time       : 2023/06/27 11:56:23
@Version    : 1.0
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Copyright  : 侵权必究
'''

# import some packages
# --------------------
import torch

# 1. GAN 简介
# 生成式对抗网络(Generative Adversarial Nets, GAN):基于博弈论的，在2014年由 Ian Goodfellow提出，
# 主要解决的问题是如何从训练样本中学习出新样本，训练样本就是图像生成新图像，训练样本是文章生成新文章等。

# GAN既不依赖标签来优化，也不是根据对结果奖惩来调整参数。它是依据生成器和判别器之间的博弈来不断优化。
# 打个不太恰当的比方，就像一台验钞机和一台制造假币的机器之间的博弈，两者不断博弈，博弈的结果假币月来越像真币，直到验钞机无法识别一张货币是假币还是真币为止。

# GAN基本原理：
#     - 一个是伪造者
#     - 一个是技术鉴赏者
#     他们训练的目的都是打败对方。
# 从网络的角度来看，由两部分组成：
#     - 生成器网络：它一个潜在空间的随机向量作为输入，并将其解码为一张合成图像。
#     - 判别器网络：以一张图像(真实的或合成的均可)作为输入，并预测该图像来自训练集还是来自生成器网络。
# 如何不断提升判别器辨别是非的能力？如何使生成的图像越来越像真图像？这些都是通过控制它们各自的损失函数来控制。
# 训练结束后，生成器能够将输入空间中的任何点转换为一张可信图像。与VAE不同的是，这个潜空间无法保证带连续性或有特殊含义的结构。
# GAN的优化过程不像通常的求损失函数的最小值，而是保持生成与判别两股力量的动态平衡。因此，其训练过程要比一般神经网络难很多。

# 2. GAN的损失函数
从GAN的架构图可知，控制生成器和判别器的关键是损失函数，而如何定义损失函数就成为整个GAN的关键。目标很明确：
既要不断提升判别器辨别是非或真假的能力，又要不断提升生成器不断提升图像的质量，使判别器越来越难判别。通过使用损失函数充分说明。

为了达到判别器的目标，其损失函数既要考虑识别真图像能力，又要考虑识别假图像能力，而不能只考虑一方面，故判别器的损失函数为两者的和。
下面的代码中：
    - D: 判别器
    - G: 生成器
    - real_labels: 真图像标签
    - fake_labels: 假图像标签
    - images: 真图像
    - z: 是从潜在空间随机采样的向量，通过生成器得到的假图像
# 定义判别器对真实图像的损失函数
outputs = D(images)
d_loss_real = criterion(outputs, real_labels)
real_cores = outputs
# 定义判别器对假图像(即:由潜在空间点生成的图像)的损失函数
z = torch.randn(btch_size, latent_size).to(device)
fake_images = G(z)
outputs = D(fake_images)
d_loss_fake = criterion(outputs, fake_labels)
fake_score = outputs
# 得到判别器总的损失函数
d_loss = d_loss_real + d_loss_fake

生成器的损失函数如何定义才能使其越来越向真图像靠近？以真图像为标杆或标签即可。
z = torch.randn(batcg_size, latent_size).to(device)
fake_images = G(z)
outputs = D(fake_images)
g_loss = criterion(outputs, real_labels)