# -*- encoding: utf-8 -*-
'''
@File    :   test_gpu.py
@Time    :   2022/02/21 15:46:49
@Author  :   yanpenggong
@Version :   1.0
@Email   :   yanpenggong@163.com
@Copyright : 侵权必究
'''

# here put the import lib
import torch
from torch.backends import cudnn


if __name__ == '__main__':
    # 测试CUDA
    print("Support CUDA ?: ", torch.cuda.is_available())
    
    x = torch.tensor([10.0])
    y = torch.randn(2, 3)
    
    # 判断是否支持cuda，是的则进行cuda运算
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    
    print("x:", x)
    print("y:", y)

    z = x + y
    print("z:", z)

    # 测试CUDNN
    print("Support cudnn ?: ", cudnn.is_acceptable(x))
    