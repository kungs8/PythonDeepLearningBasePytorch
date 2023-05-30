# -*- encoding: utf-8 -*-
'''
@Time    :   2020/7/1:上午9:52
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''

# ----------------------------------------------------------------------------------------------------------------------
# 4.4.3 用tensorboardX可视化损失值
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter


dtype = torch.FloatTensor
writer = SummaryWriter(log_dir="./res_data/logs", comment="Linear")
np.random.seed(100)
x_train = np.linspace(-1, 1, 100).reshape(100, 1)
y_train = 3 * np.power(x_train, 2) + 2 + 0.2 * np.random.rand(x_train.size).reshape(100, 1)


# 参数设置
input_size = 1
output_size = 1
num_epoches = 6
learning_rate = 0.01


model = nn.Linear(in_features=input_size, out_features=output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    inputs = torch.from_numpy(x_train).type(dtype=dtype)
    targets = torch.from_numpy(y_train).type(dtype=dtype)

    output = model(inputs)
    loss = criterion(input=output, target=targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 保存loss的数据与epoch数值
    writer.add_scalar("训练损失值", loss, epoch)
writer.close()