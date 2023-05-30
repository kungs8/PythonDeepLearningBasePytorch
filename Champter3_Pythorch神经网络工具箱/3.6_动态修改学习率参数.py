# -*- encoding: utf-8 -*-
'''
@Time    :   2020/6/26:上午10:52
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''

# -------------------------------------------------------------------------------------------------------------------------------------------------------
# 修改参数的方式可以通过修改参数 opetimizer.params_groups 或新建 optimizer
# 新建 optimizer 比较简单，optimizer 十分轻量级，所有开销很小。
# 新的优化器会初始化动量等状态信息，这对于使用动量的优化器(momentum 参数的 sgd)可能会造成收敛中的震荡。
# 这里直接采用修改参数 optimizer.params_groups
# optimizer.param_groups: 长度1的list
# optimizer.param_groups[0]: 长度为6的字典，包括权重参数、lr、momentum等参数
len(optimizer.param_groups[0]) # 结果为6

# 以下是3.2节中动态修改学习率参数代码:
for epoch in range(num_epoches):
    # 动态修改参数学习率
    if epoch % 5 ==0:
        optimizer.param_groups[0]["lr"] *= 0.1
        print("new lr value:", optimizer.param_groups[0]["lr"])
    for img, label in train_loader:
        # 需要执行的内容
        pass