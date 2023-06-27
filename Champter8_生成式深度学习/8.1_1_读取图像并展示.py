# -*- encoding: utf-8 -*-
'''
@File       : 8.1_1_读取图像并展示.py
@Time       : 2023/06/26 11:49:31
@Version    : 1.0
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Copyright  : 侵权必究
'''

# import some packages
# --------------------
from matplotlib import image as mping
import matplotlib.pyplot as plt

# 3.7 展示原图像及重构图像
reconsPath = "./temp/ave_samples/reconst-20.png"
recons_image = mping.imread(reconsPath)
plt.imshow(recons_image)
plt.axis("off")
plt.show()
# 其中奇数列尾原图像，偶数列为原图像重构的图像。

# 3.8 显示由潜在空间点Z生成的新图像
gen_path = "./temp/ave_samples/sampled-20.png"
gen_image = mping.imread(gen_path)
plt.imshow(gen_image)
plt.axis("off")
plt.show()
# 这里构建网络主要是用：全连接层。也可以换成卷积层，如果编码层使用卷积层(eg: nn.Conv2d)，解码器需要使用反卷积层(eg: nn.ConvTranspose2d)