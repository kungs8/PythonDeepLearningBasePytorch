# -*- encoding: utf-8 -*-
'''
@File    :   3.4_神经网络工具箱nn.py
@Time    :   2020/06/17 14:25:33
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# 3.4 神经网络工具箱 nn
# 前面使用Autograd及Tensor实现机器学习实例时，需要做不少设置，如对叶子节点的参数requires_grad设置为True，然后调用backward，再从grad属性中提取梯度。
# 对于大规模的网络，Autograd太过于底层和烦琐。为了简单、有效解决这个问题，nn是一个有效工具。
# nn工具箱中有两个重要模块：nn.Module, nn.functional

# -------------------------------------------------------------------------------------------------------------------------------------------------------
# 3.4.1 nn.Module
# nn.Module是nn的一个核心数据结构,它可以是神经网络的某个层(Layer), 也可以是包含多层的神经网络
# 最常见的做法:继承nn.Module, 生成自己的网络/层,如3.2 小节实例中,定义的Net类就采用这种方法(classNet(torch.nn.Module))
# 其中,nn已经实现了绝大多数层,包括全连接层、损失层、激活层、卷积层、循环层等,
# 上述层都是nn.Module的子类,能够自动检测到自己的Parameter, 并将其作为学习参数,且针对GPU运行进行了cuDNN优化

# -------------------------------------------------------------------------------------------------------------------------------------------------------
# 3.4.2 nn.functional
# nn层分两类:
# 1. 继承了nn.Module,其命名一般为nn.Xxx(第一个是大写), eg: nn.Linear, nn.Conv2d, nn.CrossEntropyLoss等
# 2. nn.functional中的函数,其名称一般为nn.functional.xxx, eg: nn.functional.linear, nn.functional.conv2d, nn.functional.cross_entropy等

# 功能上两者相当,基于nn.Module能实现的层,使用nn.functional也能实现,反之亦然，且性能方面两者没有太大差异。
# 区别:
#     1) nn.Xxx继承于nn.Module, nn.Module需要先实例化并传入参数,然后以函数调用的方式调用实例化的对象并传入输入数据,能够与nn.Sequential结合使用
#        nn.functional.xxx无法与nn.Sequential结合使用
#     2) nn.Xxx不需要自己定义和管理weight/bias参数
#        nn.functional.xxx需要自己定义weight/bias参数,每次调用的时候都需要手动传入weight/bias等参数,不利于代码复用
#     3) Dropout操作在训练和测试阶段是有区别的,使用nn.Xxx方式定义Dropout,在调用model.eval()之后,自动实现状态的转换
#        nn.functional.xxx无此功能

# 两者功能都是相同,PyTorch官方推荐:
#     1) 具有学习参数的(eg: conv2d,linear,batch_norm) 采用nn.Xxx
#     2) 没有学习参数的(eg:maxpool,loss func, activation func)等根据个人选择使用nn.functional.xxx或nn.Xxx方式

# 3.2 小节中使用激活层，我们使用F.relu 来实现，即 nn.functional.xxx 方式。
# -------------------------------------------------------------------------------------------------------------------------------------------------------
