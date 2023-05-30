import time
import math
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. math与numpy函数的性能比较
x = [i*0.001 for i in np.arange(1000000)]
start = time.clock()
for i, t in enumerate(x):
    x[i] = math.sin(t)
print("math.sin:", time.clock() - start)

x = [i*0.001 for i in np.arange(1000000)]
x = np.array(x)
start = time.clock()
np.sin(x)
print("numpy.sin:", time.clock() - start)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. 循环与向量运算比较
x1 = np.random.rand(1000000)
x2 = np.random.rand(1000000)
# 使用循环计算向量点积
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i]*x2[i]
toc = time.process_time()
print("dot = {}\n for loop----- Computation time = {}ms".format(dot, toc-tic))

# 使用numpy函数求点积
tic = time.process_time()
dot = 0
dot = np.dot(x1, x2)
toc = time.process_time()
print("dot = {}\n verctor version----- Computation time = {}ms".format(dot, toc-tic))