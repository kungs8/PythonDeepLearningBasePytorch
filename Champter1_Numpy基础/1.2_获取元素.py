import numpy as np

np.random.seed(2020)
nd11 = np.random.random([10])
print("nd11:", nd11)

# 获取指定位置的数据，获取第4个元素
print("获取第4个元素：", nd11[3])

# 截取一段数据
print("截取一段数据：", nd11[3:6])

# 截取固定间隔数据
print("截取固定间隔数据：", nd11[1:6:2])

# 倒叙取数
print("倒叙取数：", nd11[::-2])

# 截取一个多维数组的一个区域内数据
nd12 = np.arange(25).reshape([5, 5])
print("nd12:", nd12)
print("截取一个多维数组的一个区域内数据:", nd12[1:3, 1:3])

# 截取一个多维数组中，数值在一个值域之内的数据
print("数值在一个值域之内的数据:", nd12[(nd12>3)&(nd12<10)])

# 截取多维数组中，指定的行，如读取第2，3行
print("读取第2，3行:", nd12[[1,2]])
print("读取第2，3行:", nd12[1:3, :])

# 截取多维数组中，指定列，如读取第2，3列
print("读取第2，3列:", nd12[:, 1:3])

# -------------------------------------------------------------------------------------
# 通过random。choice函数从指定的样本中随机抽取数据
from numpy import random as nr

a = np.arange(1, 25, dtype=float)
print("a:", a)
c1 = nr.choice(a, size=(3, 4))  # size指定输出数组形状
c2 = nr.choice(a, size=(3, 4), replace=False)  # replace缺省为True，即可重复抽取
# 下式中参数p指定每个元素对应的抽取概率，缺省为每个元素被抽取的概率相同。
c3 = nr.choice(a, size=(3, 4), p=a/np.sum(a))
print("随机可重复抽取:", c1)
print("随机不可重复抽取:", c2)
print("随机但按制度概率抽取:", c3)
