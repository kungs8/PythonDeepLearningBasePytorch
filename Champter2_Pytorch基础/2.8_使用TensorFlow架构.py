# -*- encoding: utf-8 -*-
'''
@File    :   2.8_使用TensorFlow架构.py
@Time    :   2020/06/09 09:33:50
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''

# here put the import lib
# 1) 导入库及生成训练数据
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# print(tf.Version)
# 生成训练数据
np.random.seed(100)
x = np.linspace(-1, 1, 100).reshape(100, 1)
y = 3*np.power(x, 2)+2+0.2*np.random.rand(x.size).reshape(100, 1)
print(x)

# 2) 初始化参数
# 创建两个占位符，分别用来存放输入数据x和目标值y
# 运行计算图时，导入数据
x1 = tf.placeholder(tf.float32, shape=(None, 1))
y1 = tf.placeholder(tf.float32, shape=(None, 1))

# 创建权重变量w和b，并用随机值初始化
w = tf.Variable(tf.random.uniform([1], 0, 1.0))
b = tf.Variable(tf.zeros([1]))

# 3) 实现前向传播及损失函数
# 前向传播，计算预测值
y_pred = np.power(x, 2)*w+b

# 计算损失值
loss = tf.reduce_mean(tf.square(y-y_pred))

# 计算有关参数w、b关于损失函数的梯度
grad_w, grad_b = tf.gradients(loss, [w, b])

# 用梯度下降法更新参数
# 执行计算图时给new_w和new_b赋值
# 对TensorFlow来说，更新参数是计算图的一部分内容
learning_rate = 0.01
new_w = w.assign(w - learning_rate*grad_w)
new_b = b.assign(b - learning_rate*grad_b)

# 训练模型
# 已构建计算图，接下来创建TensorFlow session，准备执行计算图
with tf.Session() as sess:
    # 执行之前需要初始化变量w、b
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        # 循环执行计算图，每次需要把x1、y1赋给x、y
        # 每次执行计算图时，需要计算关于new_w和new_b的损失值
        # 返回numpy多维数组
        loss_value, v_w, v_b = sess.run([loss, new_w, new_b], feed_dict={x1: x, y1: y})

        if step%200 == 0:  # 每200次打印一次训练结果
            print("损失值:{:.4f}, 权重:{}, 偏移量:{}".format(loss_value, v_w, v_b))
# 5) 可视化结果
plt.figure()
plt.scatter(x, y)
plt.plot(x, v_w*x**2+v_b)
plt.show()

# -------------------------------------------------------------------------------------------------------------------------------------------------
# TensorFlow 使用静态图，其特点是先构造图形(如果不显示说明，TensorFlow会自动构建一个缺省图形)，然后启动Session，执行相关程序。
# 这个时候程序才开始运行，前面都是铺垫，所以也没有运行结果。

# 而Pytorch的动态图，动态的最关键的一点就是它是交互式的，而且执行每个命令马上就可看到结果，这对训练、发现问题、纠正问题非常方便，
# 且其构图是一个叠加(动态)过程，期间我们可以随时添加内容。这些特征对应训练和调试过程无疑是非常有帮助的。