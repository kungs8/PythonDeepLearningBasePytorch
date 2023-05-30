import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1.4.1 更改数组的形状
# 1. reshape 改变向量的维度
arr = np.arange(10)
print("arr:", arr)
# 将向量arr维度变换为2行5列
print(arr.reshape(2, 5))
# 指定维度时可以只指定行数或列数，其它用-1代替，且所指定的行数或列数一定要能被整除
print(arr.reshape(5, -1))
print(arr.reshape(-1, 5))

# 2. resize 改变向量的维度(修改向量本身)
arr = np.arange(10)
print("arr:", arr)
# 将向量arr维度变换为2行5列
arr.resize(2, 5)
print("改为2行5列:\n", arr)

# 3. T 向量转置
arr = np.arange(12).reshape(3, 4)
# 向量arr为3行4列
print("arr:", arr)
# 向量arr转置为4行3列
print("arr.T:", arr.T)

# 4. ravel 向量展平
arr = np.arange(6).reshape(2, -1)
print("arr:", arr)
# 按照列优先，展平
print("按照列优先，展平:", arr.ravel('F'))
# 按照行优先，展平
print("按照行优先，展平:", arr.ravel())

# 5. flatten 把矩阵转换为向量，这种需求经常出现在卷积网络与全连接层之间
a = np.floor(10*np.random.random((3, 4)))
print("a:", a)
print("a.flatten:", a.flatten())

# 6. squeeze 这是一个主要用来降维的函数，把矩阵中含1的维度去掉。Pytorch中还有一种与之相反的操作(torch.unqueeze)
arr = np.arange(3).reshape(3, 1)
print("arr.shape:", arr.shape)
print("arr.squeeze().shape:", arr.squeeze().shape)
arr1 = np.arange(6).reshape(3, 1, 2, 1)
print("arr1.shape:", arr1.shape)
print("arr1.squeeze().shape:", arr1.squeeze().shape)

# 7. transpose 对高维矩阵进行轴对换，这个在深度学习中经常使用。比如把图片中表示颜色顺序的RGB改为GBR
arr2 = np.arange(24).reshape(2, 3, 4)
print("arr2.shape:", arr2.shape)
print("arr2.transpose(1, 2, 0).shape:", arr2.transpose(1, 2, 0).shape)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1.4.2 合并数组
# 1. append
# 合并一维数组
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.append(a, b)
print("c:", c)

# 合并多维数组
a = np.arange(4).reshape(2, 2)
b = np.arange(4).reshape(2, 2)
# 按行合并
c = np.append(a, b, axis=0)
print("按行合并后的结果c:", c)
print("合并后数据维度c:", c.shape)
# 按列合并
d = np.append(a, b, axis=1)
print("按列合并后的结果d:", d)
print("合并后数据维度d:", d.shape)

# 2. concatenate 沿指定轴连接数组或矩阵
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
c = np.concatenate((a, b), axis=0)
d = np.concatenate((a, a), axis=1)
print("c concatenate:", c)
print("d concatenate:", d)

# 3. stack 沿指定轴堆叠数组或矩阵
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.stack((a, b), axis=0)
d = np.stack((a, b), axis=1)
print("stack c:", c)
print("stack d:", d)