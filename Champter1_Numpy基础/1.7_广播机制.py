import numpy as np
A = np.arange(0, 40, 10).reshape(4, 1)
B = np.arange(0, 3)
print("A矩阵的形状:{}, B矩阵的形状:{}".format(A.shape, B.shape))

C = A + B
print("C矩阵的形状:{}".format(C.shape))
print("C矩阵:", C)