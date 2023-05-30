# -*- encoding: utf-8 -*-
'''
@File    :   2.6_使用Numpy实现机器学习.py
@Time    :   2020/06/08 15:17:57
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''

# here put the import lib

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 需求：
#     使用最原始的Numpy实现有关回归的一个机器学习任务，不用Pytorch中的类或包。这种方法每一步都是透明的，有利于理解每步的工作原理
# 步骤：
# 1. 给出一个数组x，基于表达式y=3x**2+2，加上一些噪声数据到达另一组数据y。
# 2. 构建一个机器学习模型，学习表达式y=wx**2+b的两个参数w、b。利用数组x, y的数据为训练数据
# 3. 采用梯度下降法，通过多次迭代，学习到w, b的值
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 1) 导入需要的库
import numpy as np
from matplotlib import pyplot as plt

# 2) 生成输入数据x及目标数据y
# 设置随机种子，生成同一份数据，以便用多种方法进行比较
np.random.seed(100)
x = np.linspace(-1, 1, 100).reshape(100, 1)
y = 3*np.power(x, 2)+2+0.2*np.random.rand(x.size).reshape(100, 1)

# 3) 查看x, y数据分布
# 画图
plt.scatter(x, y)
plt.show()

# 4) 初始化权重参数
# 随机初始化参数
wl= np.random.rand(1, 1)
bl = np.random.rand(1, 1)

# 5) 训练模型
    # 定义损失函数，假设批量大小为100:
    # Loss = 1/2*\Sigma_{i=1}^{100}(w*x_{i}**2+b-y_{i})**2
    # 对损失函数求导：\partial{Loss}/\partial{w} = \Sigma_{i=1}^{100}(w*x_{i}**2+b-y_{i})*x_{i}**2
                    # \partial{Loss}/\partial{b} = \Sigma_{i=1}^{100}(w*x_{i}**2+b-y_{i})
    # 利用梯度下降法学习参数，学习率为lr: w_{l} = lr*\partial{Loss}/\partial{w}
                                        # b_{l} = lr*\partial{Loss}/\partial{b}
lr = 0.001  # 学习率
for i in range(800):
    # 前向传播
    y_pred = np.power(x, 2)*wl + bl
    # 定义损失函数
    loss = 0.5 * (y_pred -y) ** 2
    loss = loss.sum()
    # 计算梯度
    grad_w = np.sum((y_pred-y)*np.power(x, 2))
    grad_b = np.sum((y_pred-y))
    # 使用梯度下降法，是loss最小
    wl -= lr * grad_w
    bl -= lr * grad_b

# 6) 可视化结果
plt.plot(x, y_pred, "r-", label="predict")
plt.scatter(x, y, color="blue", marker="o", label="true")  # true data
plt.title("predict \n wl={:.2f}, bl={}".format(wl[0][0], bl[0][0]))
plt.xlim(-1, 1)
plt.ylim(2, 6)
plt.legend()
plt.show()