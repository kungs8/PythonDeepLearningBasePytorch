import torch
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2.4.1 tensor概述
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
# add 不修改自身数据
z = x.add(y)
print("z: ", z)
print("x: ", x)
# add_修改自身数据
x.add_(y)
print("x add after:", x)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2.4.2 创建Tensor
# 根据list数据生成Tensor
print("根据list数据生成Tensor:\n", torch.Tensor([1, 2, 3, 4, 5, 6]))
# 根据指定形状生成Tensor
print("根据指定形状生成Tensor:\n", torch.Tensor(2, 3))
# 根据给定的Tensor的形状
t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
# 查看Tensor的形状
print("Tensor的形状:", t.size())
print("Tensor的形状:", t.shape)
# 根据已有形状创建Tensor
print("根据已有形状创建Tensor:\n", torch.Tensor(t.size()))

# torch.Tensor与torch.tensor区别
# torch.Tensor是torc.empty和torch.tensor之间的一种混合，但当传入数据时，torch.Tensor使用全局默认dtype(FloatTensor)，而torch.tensor是从数据中推断数据类型
# torch.tensor(1) 返回一个固定的值，而torch.Tensor(1)返回一个大小为1的张量，它是随机初始化的值。
t1 = torch.Tensor(1)
t2 = torch.tensor(1)
print("t1的值:{}, t1的数据类型:{}".format(t1, t1.type()))
print("t2的值:{}, t2的数据类型:{}".format(t2, t2.type()))

# 生成一个单位矩阵
print("生成一个单位矩阵:\n", torch.eye(2, 2))
# 自动生成全是0的矩阵
print("自动生成全是0的矩阵:\n", torch.zeros(2, 3))
# 根据规则生成数据
print("根据规则生成数据:\n", torch.linspace(1, 10, 4))
# 生成满足均匀分布随机数
print("生成满足均匀分布随机数:\n", torch.rand(2, 3))
# 生成满足标准分布随机数
print("生成满足标准分布随机数:\n", torch.randn(2, 3))
# 返回所给数据形状相同，值全为0的张量
print("返回所给数据形状相同，值全为0的张量:\n", torch.zeros_like(torch.rand(2, 3)))

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2.4.3 修改Tensor形状
# 生成一个形状为2*3矩阵
x = torch.randn(2, 3)
# 查看矩阵的形状
print("x.size(): ", x.size())
# 查看x的维度
print("x的维度:", x.dim())

# 把x变为3*2的矩阵
print("3*2的矩阵:\n", x.view(3, 2))
# 把x展平为1维问题
y = x.view(-1)
print("y维度:", y.shape)
# 添加一个维度
z = torch.unsqueeze(y, 0)
print("添加一个维度:", z)
print("查看z的形状:", z.size())
print("计算z的元素个数:", z.numel())

# torch.view与torch.reshape的异同
# 1) reshape()可以由torch.reshape()， 也可由torch.Tensor.reshape()调用。但view()只可由torch.Tensor.view()来调用
# 2) 对于一个将要被view的Tensor，新的size必须与原来的size与stride兼容。否则，在view之前必须调用contiguous()方法
# 3) 同样也是返回与input数量相同，但形状不同的Tensor。若满足view的条件，则不会copy，若不满足，则会copy
# 4) 如果只想重塑张量，请使用torch.reshape。如果还关注内存使用情况并希望确保两个张量共享相同的数据，请使用torch.view

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2.4.4 索引操作
# 设置一个随机种子
torch.manual_seed(100)

# 生成一个形状为2*3的矩阵
x = torch.randn(2, 3)
print("2*3的矩阵:\n", x)

# 根据索引获取第1行所有数据
print("第1行所有数据:", x[0, :])

# 获取最后1列数据
print("最后1列数据:", x[:, -1])

# 生成是否大于0的Byter张量
mask = x>0
print("是否大于0的Byter张量:\n", mask)

# 获取非0下标，即行、列索引
print("非0下标:\n", torch.nonzero(mask))

# 获取指定索引对应的值，输出根据以下规则得到
# out[i][j] = input[index[i][j]][j]  # if dim == 0
# out[i][j] = input[i][index[i][j]]  # if dim == 1
index = torch.LongTensor([[0, 1, 1]])
torch.gather(x, 0, index)
print("torch.gather(x, 0, index):", torch.gather(x, 0, index))
index = torch.LongTensor([[0, 1, 1], [1, 1, 1]])
print("x pre gather: \n", x)
a = torch.gather(x, 1, index)
print("a:", a)

# 把a的值返回到一个2*3的0矩阵中
z = torch.zeros(2, 3)
z.scatter_(1,index, a)
print("z:", z)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2.4.5 广播机制
import numpy as np
A = np.arange(0, 40, 10).reshape(4, 1)
B = np.arange(0, 3)
# 把ndarray转换为Tensor
A1 = torch.from_numpy(A)  # 形状为4*1
B1 = torch.from_numpy(B)  # 形状为3
# 自动实现广播
C = A1 + B1
print("C:", C)
# 我们可以根据广播机制，手工进行配置
# 根据规则1，B1需要向A1看齐，把B变为(1, 3)
B2 = B1.unsqueeze(0)  # B2的形状为(1, 3)
# 使用expand函数重复数组，分别变为4*3矩阵
A2 = A1.expand(4, 3)
B3 = B2.expand(4, 3)
# 然后进行相加，C1与C结果一致
C1 = A2 + B3
print("C1:", C1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2.4.6 逐元素操作
t = torch.randn(1, 3)
t1 = torch.randn(3, 1)
t2 = torch.randn(1, 3)
print("t:", t)
print("t1:", t1)
print("t2:", t2)
# t+0.1*(t1/t2)
res = torch.addcdiv(t, 0.1, t1, t2)
print("res:\n", res)
# 计算sigmoid(t)
res_sigmoid = torch.sigmoid(t)
print("res_sigmoid:", res_sigmoid)
# 将t限制在[0, 1]之间
res_limit= torch.clamp(t, 0, 1)
print("res_limit:", res_limit)
# t+2进行就地运算
print("t add pre:", t)
t.add_(2)
print("t:", t)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2.4.7归并操作
# 生成一个含6个数的向量
a = torch.linspace(0, 10, 6)
print("linspace a:", a)
# 使用view方法，把a变为2*3矩阵
a = a.view((2, 3))
print("view a:\n", a)
# 沿y轴方向累加，即dim=0
b = a.sum(dim=0)  # b的形状为[3]
print("沿y轴方向累加 b:", b)
# 沿y轴方向累加，即dim=0，并保留含1的维度
b = a.sum(dim=0, keepdim=True)  # b的形状为[1, 3]
print("沿y轴方向累加，并保留含1的维度 b:", b)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2.4.8 比较操作
x = torch.linspace(0, 10, 6).view(2, 3)
print("x value:\n", x)

# 求所有元素的最大值
max_value = torch.max(x, dim=0)  # 结果为[6, 8, 10]
print("max_value:", max_value)

# 求最大的2个元素
max_2 = torch.topk(x, 1, dim=0)  # 结果为[[6, 8, 10]], 对应索引为 tensor([[1, 1, 1]])
print("max_2:\n", max_2)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2.4.9 矩阵操作
# 1) Torch的dot与Numpy的dot有点不同，Torch中的dot是对两个为ID张量进行点积运算，Numpy中的dot无此限制
# 2) mm是对2D的矩阵进行点积，bmm对含batch的3D进行点积运算
# 3) 转置运算会导致存储空间不连续，需要调用contiguous方法转为连续
a = torch.tensor([2, 3])
b = torch.tensor([3, 4])
dot_value = torch.dot(a, b)  # 运算结果为18
print("dot_value:", dot_value)

x = torch.randint(10, (2, 3))
y = torch.randint(6, (3, 4))
print("x:", x)
print("y:", y)
mm_value = torch.mm(x, y)
print("2D mm_value:", mm_value)

x = torch.randint(10, (2, 2, 3))
y = torch.randint(6, (2, 3, 4))
print("x:", x)
print("y:", y)
mm_value = torch.bmm(x, y)
print("3D mm_value:", mm_value)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2.4.10 Pytorch与Numpy比较
# 一、Numpy:
# 1) 数据类型
    # np.ndarray
    # np.float32
    # np.float64
    # np.int64
# 2) 从已有数据构建
x = np.array([3.2, 4.3], dtype=np.float16)
b = x.copy()
c = np.concatenate((x, b), axis=0)  # 按轴axis连接array组成一个新的array
print("c:",c)  # c: [3.2 4.3 3.2 4.3]
# 3) 线性代数
dot_value = np.dot(x, b)
# 4) 属性
print("ndim:", x.ndim)
print("size:", x.size)
# 5) 形状操作
print("reshape:\n", x.reshape(2,1))
print("flatten:", x.flatten())  # 展平
# 6) 类型转换
print("floor:", np.floor(x))
# 7) 比较
print("less:", np.less(x, b))
print("less_equal:", np.less_equal(x, b))
print("greater:", np.greater(x, b))
print("greater:", np.greater(x, b))
print("greater_equal:", np.greater_equal(x, b))
print("equal:", np.equal(x, b))
print("not_equal:", np.not_equal(x, b))
# 8) 随机种子
np.random.seed(100)

# Pytorch:
# 1) 数据类型
    # torch.Tensor
    # torch.float32; torch.float
    # torch.float64; torch.double
    # torch.int64; torch.long
# 2) 从已有数据构建
x = torch.tensor([3.2, 4.3])
x = x.reshape(2,1)
x = [x for i in range(3)]
print("x:", x)
c = torch.cat((x), 1)  # 把list中的tensor拼接起来
print("c:",c)
# 3) 线性代数
x = torch.tensor([3.2, 4.3])
x = x.reshape(2,1)
b = x.clone()
b = b.reshape(1, 2)
mm_value = torch.mm(x,b)
print("mm_value:", mm_value)
# 4) 属性
print("dim:", x.dim())  # 维度
print("nelement:", x.nelement())  # 维度
# 5) 形状操作
print("reshape:", x.reshape(2, 1))
print("view:", x.view(1, 2))
# 6) 类型转换
print("torch.floor:", torch.floor(x))
print("x.floor:", x.floor())
# 7) 比较
    # torch.lt(input, other, out=None) 逐元素比较input和other，即是否input < other
    # input.lt(other) 逐元素比较input和other，即是否input < other
print("torch.lt:", torch.lt(input=x, other=3.5))
print("x.lt:", x.lt(3.5))
    # torch.le(input, other, out=None)  逐元素比较input和other，即是否input <= other.
    # input.le(other)  逐元素比较input和other，即是否input <= other.
print("torch.le:", torch.le(input=x, other=3.2))
print("x.le:", x.le(3.2))
    # torch.gt(input, other, out=None) 逐元素比较input和other，即是否input > other
    # input.gt(other) 逐元素比较input和other，即是否input > other
print("torch.gt:", torch.gt(input=x, other=3.2))
print("x.gt:", x.gt(3.2))
    #  torch.ge(input, other, out=None) 逐元素比较input和other，即是否input >= other
    # input.ge(other) 逐元素比较input和other，即是否input >= other
print("torch.ge:", torch.ge(input=x, other=3.2))
print("x.ge:", x.ge(3.2))
    # torch.eq(input, other, out=None) 比较元素是否相等，第二个参数可以是一个数，或者是第一个参数同类型形状的张量
    # input.eq( other) 比较元素是否相等，第二个参数可以是一个数，或者是第一个参数同类型形状的张量
print("torch.eq:", torch.eq(input=x, other=3.2))
print("x.eq:", x.eq(3.2))
    # torch.equal(tensor1, tensor2, out=None) 如果两个张量有相同的形状和元素值，则返回true，否则False
    # input.equal(other, out=None) 如果两个张量有相同的形状和元素值，则返回true，否则False
a = torch.Tensor([1, 2])
b = torch.Tensor([1, 2])
print("torch.equal:", torch.equal(input=a, other=b))
print("x.equal:", a.equal(b))
    # torch.ne(input, other, out=None) 逐元素比较input和other，即是否input 不等于 other。第二个参数可以为一个数或与第一个参数相同形状和类型的张量
print("torch.ne:", torch.ne(input=x, other=3.2))

    # torch.kthvalue(input, k, dim=None, out=None)
    # 取输入张量input指定维度上第k个最小值。如果不指定dim。默认为最后一维。
    # 返回一个元组(value, indices), 其中indices是原始输入张量中沿dim维的第k个最小值下标。
    # 参数：
    # input(Tensor) ---- 要对比的张量
    # k(int) ---- 第k个最小值
    # dim(int, 可选的) ---- 沿着此维度进行排序
    # out(tuple,可选的) ---- 输出元组
a = torch.arange(1, 6)
b = torch.kthvalue(a, 4)
print("torch.kthvalue 4:\n", b)
b = torch.kthvalue(a, 1)
print("torch.kthvalue 1:\n", b)

# 8) 随机种子
torch.manual_seed(100)