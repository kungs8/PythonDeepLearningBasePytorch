import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1.3.1 对应元素相乘
A = np.array([[1, 2], [-1, 4]])
B = np.array([[2, 0], [3, 4]])
print("A*B:", A*B)
print("np.multiply:", np.multiply(A, B))

# Numpy数组与单一数值(标量)运算
print("A*2.0:", A*2.0)
print("A/2.0:", A/2.0)

# 推广：数组通过一些激活函数后，输出与输入形状一致
X = np.random.rand(2, 3)
def softmoid(x):
    return 1/ (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))
print("输入参数X:", X)
print("输入参数X的形状", X.shape)
print("激活函数softmoid:", softmoid(X))
print("激活函数softmoid输出形状:", softmoid(X).shape)
print("激活函数relu:", relu(X))
print("激活函数relu输出形状:", relu(X).shape)
print("激活函数softmax:", softmax(X))
print("激活函数softmax输出形状:", softmax(X).shape)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1.3.2 点积运算
X1 = np.array([[1, 2], [3, 4]])
X2 = np.array([[5, 6, 7], [8, 9, 10]])
X3 = np.dot(X1, X2)
print("X3:", X3)