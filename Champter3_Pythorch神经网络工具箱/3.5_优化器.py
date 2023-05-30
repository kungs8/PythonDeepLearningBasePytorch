# -*- encoding: utf-8 -*-
'''
@Time    :   2020/6/26:上午9:07
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''

# -------------------------------------------------------------------------------------------------------------------------------------------------------
# Pytorch常用的优化方法都封装在 torch.optim 里面，其设计很灵活，可扩展为自定义的优化方法。
# 所有都优化方法都继承了基类 optim.Optimizer，并实现了自己的优化步骤。
# 最常用的优化算法是梯度下降法及其各种变种，后续章节会介绍各种算法的原理，这类优化算法通过使用参数的梯度值更新参数。

# 3.2中随机梯度下降法(SGD)是最普通的优化器，一般SGD并没有加速效果。
# 3.2中使用的SGD包含动量参数Momentum，它是SGD的改良版。

# 优化器使用的一般步骤：
# 1. 建立优化器实例
from torch import optim, nn
optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
# 下面步骤在训练模型的for循环中。

# 2. 向前传播
# 把输入数据传入神经网络Net实例化对象model中，自动执行forward函数，得到out输出值，然后用out与标记label计算损失值loss
model = Net(in_dim=28 * 28, n_hidden_1=300, n_hidden_2=100, out_dim=10)
criterion = nn.CrossEntropyLoss()

out = model(img)
loss = criterion(out, label)

# 3. 清空梯度
# 缺省情况梯度是累加的，在梯度反向传播前，先需把梯度归零
optimizer.zero_grad()

# 4. 反向传播
# 基于损失值，把梯度进行反向传播
loss.backward()

# 5. 更新参数
# 基于当前梯度(存储在参数的.grad属性中)更新参数
optimizer.step()