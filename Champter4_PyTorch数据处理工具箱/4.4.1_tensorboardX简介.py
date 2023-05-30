# -*- encoding: utf-8 -*-
'''
@Time    :   2020/6/28:下午10:43
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''

# Tensorboard是Google TensorFlow的可视化工具，它可以记录训练数据、评估数据、网络结构、图像等，
# 并且可以在web上显示，对观测神经网络训练过程非常有帮助
# Pytorch可以采用tensorboard_logger、visdom等可视化工具，但这些方法比较复杂或不够友好，
# 为解决这问题，推出了可用于PyTorch可视化但心啊但更强大但工具tensorboardX

# ----------------------------------------------------------------------------------------------------------------------
# 4.4.1 tensorboardX简介
# 使用tensorboardX的一般步骤：
# 1) 导入tensorboardX，实例化SummaryWriter类，指明记录日志路径等信息
from tensorboardX import SummaryWriter
# 实例化SummaryWriter，并指明日志存放路径，在当前目录没有logs目录将自动创建
#     writer = SummaryWriter(log_dir="logs")
# 调用实例 # writer.add_xxx()
#     writer.add_graph()
# 关闭writer
#     writer.close()
# 说明：
# (1) 如果是Windows环境，log_dir注意路径解析
#     writer = SummaryWriter(log_dir=r"C:\myboard\test\logs")
# (2) SummaryWriter的格式
# SummaryWriter(log_dir=None, comment="", **kwargs)
# 其中comment在文件名加上comment后缀
# (3) 如果不写log_dir，系统将在当前目录创建一个runs的目录
#
# 2) 调用相应的API接口，接口一般格式为:
# add_xxx(tag-name, object, iteration-number)
# 即add_xxx(标签， 记录的对象， 迭代次数)

# 3) 启动tensorboard服务
#     cd 到 logs目录所在的同级目录，在命令行输入如下命令，logdir等式右边可以是相对路径或绝对路径
#     tensorboard --logdir --port 6006

# 4) web展示
#     在浏览器输入：
#         http://服务器IP或名称:6006  # 如果是本机， 服务器名称可以使用localhost
# 更多内容参见：https://github.com/lanpa/tensorboardX