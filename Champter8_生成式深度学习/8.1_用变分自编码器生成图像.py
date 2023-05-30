# -*- encoding: utf-8 -*-",
"""
@File      : 8.1_用变分自编码器生成图像.py
@Time      : 2023/5/30 10:57
@Author    : yanpenggong
@Version   : 1.0
@Email     : yanpenggong@163.com
@Copyright : 侵权必究
@Project : 003.pytorchLearning
@Software: PyCharm
"""
# here put the import lib
变分自编码器是自编码器的改进版本，自编码器是一种无监督学习，但它无法产生新的内容，变分自编码器对其潜在空间进行拓展，使其满足正态分布，情况就大不一样了。