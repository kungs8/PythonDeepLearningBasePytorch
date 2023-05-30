# -*- encoding: utf-8 -*-
'''
@File    :   2.7_使用Tensor及Antograd实现机器学习.py
@Time    :   2020/06/08 17:09:28
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''

# here put the import lib
# 本节使用Pytorch的一个自动求导包(antograd)，利用自动反向传播来求梯度，无需手动计算梯度
# 1) 导入需要的库
import torch
from matplotlib import pyplot as plt

# 2) 生成训练数据，并可视化数据分布情况
torch.manual_seed(100)
dtype = torch.float
# 生成x坐标数据，x为tensor, 需要把x的形状转换为100*1, 有两种方法
# x = torch.linspace(-1, 1, 100).reshape(100, 1) 
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# 生成y坐标数据，y为tensor，形状为100*1，令假设一些噪声
y = 3*x.pow(2)+2+0.2*torch.rand(x.size())
# 画图，把tensor数据转换为numpy数据
plt.scatter(x.numpy(), y.numpy())
plt.show()

# 3) 初始化权重参数
#随机初始化参数，参数w、b为需要学习的，故需requires_grad=True
w = torch.randn(1, 1, dtype=dtype, requires_grad=True)
b = torch.zeros(1, 1, dtype=dtype, requires_grad=True)

# 4) 训练模型
lr = 0.001  # 学习率
for ii in range(800):
    # 前向传播，并定义损失函数loss
    y_pred = x.pow(2).mm(w) + b
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()
    # 自动计算梯度，梯度存放在grad属性中
    loss.backward()
    # 手动更新参数，需要用torch.no_grad(), 使上下文环境中切断自动求导的计算
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    # 梯度清零
    w.grad.zero_()
    b.grad.zero_()

# 5) 可视化训练结果
plt.plot(x.numpy(), y_pred.detach().numpy(), "r-", label="predict")  # predict
plt.scatter(x.numpy(), y.numpy(), color="blue", marker="o", label="true")  # True data
plt.title("predict \n w={:.2f}, b={}".format(w[0][0], b[0][0]))
plt.xlim(-1, 1)
plt.ylim(2, 6)
plt.legend()
plt.show()