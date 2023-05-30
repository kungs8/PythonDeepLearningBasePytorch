import numpy as np


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1.1.1 从已有数据中创建数组
# 1) 将列表转换成ndarray
lst1 = [3.14, 2.17, 0, 1, 2]
nd1 = np.array(lst1)
print(nd1)
print(type(nd1))

# 2) 嵌套列表可以转换成多维ndarray
lst2 = [[3.14, 2.17, 0, 1, 2], [1, 2, 3, 4, 5]]
nd2 = np.array(lst2)
print(nd2)
print(type(nd2))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1.1.2 利用random模块生成数组
nd3 = np.random.random([3, 3])
print(nd3)
print("nd3的形状为：", nd3.shape)

# 为了每次生成同一份数据，可以指定一个随机种子，使用shuffle函数打乱生成的随机数
np.random.seed(123)
nd4 = np.random.randn(2, 3)
print(nd4)
np.random.shuffle(nd4)
print("随机打乱后数据：\n", nd4)
print(type(nd4))

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1.1.3 创建特定形状的多维数组
# 生成全是0 的 3*3 矩阵
nd5 = np.zeros([3, 3])
print("nd5:", nd5)
# 生成与nd5形状一样的全0矩阵
print("nd5 like:", np.zeros_like(nd5))
# 生成全是 1 的 3*3 矩阵
nd6 = np.ones([3, 3])
print("nd6:", nd6)
# 生成3阶的单位矩阵
nd7 = np.eye(3)
print("nd7:", nd7)
# 生成3阶对角矩阵
nd8 = np.diag([1, 2, 3])
print("nd8:", nd8)

# 生成的数据临时保存起来，以备后续使用
nd9 = np.random.random([5, 5])
np.savetxt(X=nd9, fname="./Champter1/data/save_data/nd9.txt")
nd10 = np.loadtxt("./Champter1/data/save_data/nd9.txt")
print("nd10:", nd10)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1.1.4 利用arange、linespace函数生成数组
# np.arange([start,] stop[, step,], dtype=None) 其中start与stop用来指定范围，step用来设定步长(可为小数)， 类似python内置函数range
print("range 10:", np.array(10))
print("range(0, 10):", np.arange(0, 10))
print("range(1, 4, 0.5):", np.arange(1, 4, 0.5))
print("range(9, -1, -1):", np.arange(9, -1, -1))

# np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None) 可以根据输入的指定数据范围以及等份数量，自动生成一个线性等分向量，
# 其中 endpoint(包含终点)默认为True，等分数量num默认为50. 如果将restep设置为True，则会返回一个带步长的ndarray
print("不带步长的ndarray：", np.linspace(start=0, stop=1, num=10))
print("带步长的ndarray：", np.linspace(start=0, stop=1, num=10, retstep=True))  # 步长=(1-0)/9=0.1111111
print("带步长的ndarray：", np.linspace(start=0.1, stop=1, num=10, retstep=True))  # 步长=(1-0.1)/9=0.1