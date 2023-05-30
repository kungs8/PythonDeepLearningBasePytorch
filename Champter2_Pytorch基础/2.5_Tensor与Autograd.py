# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 2.5.3 标量反向传播
# 1) 定义叶子节点及算子节点
import torch
# 定义输入张量
x = torch.Tensor([2])

# 初始化权重参数w, 偏移量b, 并设置require_grad属性为True， 为自动求导
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# 实现前向传播
y = torch.mul(w, x)  # 等价于w*x
z = torch.add(y, b)  # 等价于y+b

# 查看x, w, b页子节点的require_grad属性
print("x, w, b的require_grad属性分别为:{x}, {w}, {b}".format(x=x.requires_grad, w=w.requires_grad, b=b.requires_grad))
# x, w, b的require_grad属性分别为:False， True， True

# 2) 查看叶子节点、非叶子节点的其它属性
# 查看非叶子节点的requires_grad属性
print("y, z的require_grad属性分别为:{y}, {z}".format(y=y.requires_grad, z=z.requires_grad))

# 查看各节点是否为叶子节点
print("x, w, b, y, z的is_leaf属性分别为:{x}, {w}, {b}, {y}, {z}".format(x=x.is_leaf, w=w.is_leaf, b=b.is_leaf, y=y.is_leaf, z=z.is_leaf))
# x, w, b, y, z的is_leaf属性分别为:True, True, True, False, False

# 查看叶子节点的grad_fn属性
print("x, w, b的grad_fn属性分别为:{x}, {w}, {b}".format(x=x.grad_fn, w=w.grad_fn, b=b.grad_fn))
# 查看非叶子节点的grad_fn属性
print("y, z的grad_fn属性分别为:{y}, {z}".format(y=y.grad_fn, z=z.grad_fn))

# 3) 自动求导，实现梯度方向传播，即梯度的反向传播
# 基于z张量进行梯度反向传播，执行backward之后计算图会自动清空
# z.backward()
# 如果需要多次使用backward，需要修改参数retain_graph为True，此时梯度是累加的
z.backward(retain_graph=True)

# 查看叶子节点的梯度，x是叶子节点但它无需求导，故其梯度为None
print("参数w, b, x的梯度分别为：{w}, {b}, {x}".format(w=w.grad, b=b.grad, x=x.grad))
# 参数w, b, x的梯度分别为：tensor([2.]), tensor([1.]), None

# 非叶子节点的梯度，执行backward之后，会自动清空
print("非叶子节点y, z 的梯度分别为: {y}, {z}".format(y=y.grad, z=z.grad))
# 非叶子节点y, z 的梯度分别为: None, None

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 2.5.4 非标量反向传播
# backward函数的格式为:
# torch.backward(gradient=None, retain_graph=None, create_graph=False)
# 实例：
# 1) 定义叶子节点及计算节点
# 定义叶子节点张量x， 形状为1*2
x = torch.tensor([[2, 3]], dtype=torch.float, requires_grad=True)
print("x:", x)
# 初始化Jacobian矩阵
J = torch.zeros(2, 2)
# 初始化目标张量，形状为1*2
y = torch.zeros(1, 2)
# 定义y与x之间的映射关系
# y1=x1**2+3*x2, y2=x2**2+3*x1
y[0, 0] = x[0, 0] ** 2 + 3 * x[0, 1]
y[0, 1] = x[0, 1] ** 2 + 3 * x[0, 0]

# 2) 手工计算y对x的梯度
# 3) 调用backward来获取y对x的梯度
# 生成y1对x的梯度
y.backward(torch.Tensor([[1, 0]]), retain_graph=True)
J[0] = x.grad
# 梯度是累加的，故需要对x的梯度清零
x.grad = torch.zeros_like(x.grad)
# 生成y2对x的梯度
y.backward(torch.Tensor([[0, 1]]))
J[1] = x.grad
# 显示jacobian矩阵的值
print("J:", J)

# ----------------------------------------------------------------------------------------------------------------------------------------------------