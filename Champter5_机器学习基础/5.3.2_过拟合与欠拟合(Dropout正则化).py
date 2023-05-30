# -*- encoding: utf-8 -*-
'''
@Time    :   2020/7/17:上午11:42
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''

# ----------------------------------------------------------------------------------------------------------------------
# Dropout 在训练过程中，按一定的比例(比例参数可设置)随机忽略或屏蔽一些神经元。
# 这些神经元会被随机"抛弃"，也就是说它们在正向传播中对于下游神经元的贡献效果暂时消失了，反向传播时该神经元也不会有任何权重的更新。
# 故通过传播过程，Dropout将产生和L2范数相同的收缩权重的效果。

# 随着神经网络模型的不断学习，神经元的权值会与整个网络的上下文相匹配。
# 神经元的权重针对某些特征进行调优，进而产生一些特殊化。
# 周围的神经元则会依赖于这种特殊化，但如果过于特殊化，模型会因为对训练数据的过拟合而变得脆弱不堪。
# 神经元在训练过程中的这种依赖于上下文的现象被称为复杂的协同适应(Complex Co-Adaptations)。

# 加入了 Dropout 后，输入的特征都是有可能会被随机清除的，所以该神经元不会再特别依赖于任何一个输入特征，也就是说不会给任何一个输入设置太大的权重。
# 由于网络模型对神经元特定的权重不那么敏感。这反过来又提升了模型的泛化能力，不容易对训练数据过拟合。

# Dropout 训练的集成包括所有从基础网络除去非输出单元形成子网络。
# Dropout 训练所有子网络组成的集合，其中子网络是从基本网络中删除非输出单元所构建的。
# 从具有两个可见单元和两个隐藏单元的基本网络开始，这 4 个单元有 16 个可能的子集。
# 上述的例子中，所得的大部分网络没有输入单元或没有从输入连接到输出的路径。当层较宽时，丢弃所有从输入到输出的可能路径的概率变小，所以这个问题对于层较宽的网络不是很重要。

# Dropout在训练阶段和测试阶段是不同的，一般在训练时使用，测试阶段不使用。(不过测试时，为了平衡(因训练时舍弃了部分节点或输出)，一般将输出按Dropout Rate比例缩小)
# Dropout一般原则：
# 1.通常丢弃率控制在20%-50%比较好，可以从20%开始尝试。(比例太低起不到效果，比例太高导致模型欠学习)
# 2.在大的网络模型上应用
#   当Dropout应用在较大的网络模型时，更有可能得到效果的提升，模型有更多的机会学习到多种独立的表征
# 3.在输入层和隐藏层都使用Dropout
#   对不同的层，设置的keep_prob也不同，一般来说神经元较少的层，会设keep_prob为1.0或接近于1.0的数；神经元较多的层，则会将keep_prob设置得较小，如0.5或更小
# 4.增加学习速率和冲量
#   把学习速率扩大10-100倍，冲量值调高到0.9-0.99
# 5.限制网络模型的权重
#   大的学习速率往往会导致大的权重值。对网络的权重值做最大范数的正则化，被证明能提升模型性能
#
# 批量正则化：
# Batch Normalization 不仅可以有效的解决梯度消失问题，而且还可以让调试参数更加简单，在提高训练模型效率的同时，还可以让神经网络模型更加"健壮"。
# BN是对隐藏层对标准化处理，它与输入的标准化处理(Normalizing Inputs)是有区别的，Normalizing Inputs是使所有输入的均值为0，方差为1；BN可使各隐藏层输入的均值和方差为任意值。
# 实际上，从激活函数的角度来看，如果各隐藏层的输入均值在靠近0的区域，即处于激活函数的线性区域，这样不利于训练好的非线性神经网络，而且得到的模型效果也不会太好。
# BN使用的范围尝试：一般在神经网络训练时遇到收敛速度很慢，或梯度爆炸等无法训练的状况时，可以尝试用BN来解决。
#                 在一般情况下，也可以加入BN来加快训练速度，提高模型精度，还可以大大地提高训练模型的效率。
# BN功能：
#     1) 可以选择比较大的初始学习率，让训练速度飙升。之前还需要慢慢地调整学习率，甚至在网络训练到一半的时候，还需要想着学习率进一步调小的比例选择多少比较合适。
#         可以采用初始很大的学习率，然而学习率的衰减速度也很快，因为这个算法收敛很快。即使选择较小的学习率，也比之前的收敛速度快，因为它具有快速训练收敛的特性。
#     2) 不用再去理会过拟合中Dropout、L2正则项参数的选择问题，采用BN算法后，可以移除这两项参数，或者可以选择更小的L2正则约束参数率，因为BN具有提高网络泛化能力的特性。
#     3) 再也不需要使用局部响应归一化层。
#     4) 可以把训练数据彻底打乱。


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import torch

# 获取数据
from tensorboardX import SummaryWriter

bostons = load_boston()
# 获取data与target
data, labels = bostons.data, bostons.target

dim = data.shape[1]
print(dim)

# 分训练集和测试集
data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.2, random_state=0)  # test_size:样本占比，如果是整数的话就是样本的数量; random_state:是随机数的种子
print("data_train shape:", data_train.shape)

# 对训练数据进行标准化
mean = data_train.mean(axis=0)  # 均值
std = data_train.std(axis=0)  # 方差
data_train -= mean
data_train /= std

# 对测试数据进行标准化
data_test -= mean
data_test /= std

train_data = torch.from_numpy(data_train)
dtype = torch.FloatTensor
print(train_data.type(dtype))

# 实例化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = torch.from_numpy(data_train).float()
train_label = torch.from_numpy(label_train).float()
test_data = torch.from_numpy(data_test).float()
test_label = torch.from_numpy(label_test).float()

# 正则化
net1_overfitting = torch.nn.Sequential(
    torch.nn.Linear(13, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1))
net1_nb = torch.nn.Sequential(
    torch.nn.Linear(13, 16),
    torch.nn.BatchNorm1d(num_features=16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 32),
    torch.nn.BatchNorm1d(num_features=32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1))
net2_nb = torch.nn.Sequential(
    torch.nn.Linear(13, 8),
    torch.nn.BatchNorm1d(num_features=8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 4),
    torch.nn.BatchNorm1d(num_features=4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1))
net1_dropped = torch.nn.Sequential(
    torch.nn.Linear(13, 16),
    torch.nn.Dropout(p=0.5),  # 对所有元素中每个元素按照概率0.5更改为零. Dropout2d:对每个通道按照概率0.5置为0
    torch.nn.ReLU(),
    torch.nn.Linear(16, 32),
    torch.nn.Dropout(p=0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1))

# 定义损失函数和优化器
lossfunc = torch.nn.MSELoss()
optimizer_ofit = torch.optim.Adam(net1_overfitting.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(net1_dropped.parameters(), lr=0.01)
optimizer_nb = torch.optim.Adam(net2_nb.parameters(), lr=0.01)

writer = SummaryWriter(log_dir="./res_data/logs")

for epoch in range(200):
    net1_overfitting.train()
    net1_dropped.train()
    net2_nb.train()

    pred_ofit = net1_overfitting(train_data)
    pred_drop = net1_dropped(train_data)
    pred_nb = net2_nb(train_data)

    # 正向及反向传播
    loss_ofit = lossfunc(input=pred_ofit, target=train_label)
    loss_drop = lossfunc(input=pred_drop, target=train_label)
    loss_nb = lossfunc(input=pred_nb, target=train_label)

    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    optimizer_nb.zero_grad()

    loss_ofit.backward()
    loss_drop.backward()
    loss_nb.backward()

    optimizer_ofit.step()
    optimizer_drop.step()
    optimizer_nb.step()

    # 保存loss的数据与epoch的值
    writer.add_scalars("train_group_loss", {"loss_ofit":loss_ofit.item(), "loss_drop":loss_drop.item(), "loss_nb":loss_nb.item()}, epoch)

    # test数据，关闭Dropout功能,因为在测试的数据不需要进行训练
    net1_overfitting.eval()
    net1_dropped.eval()
    net2_nb.eval()

    test_pre_ofit = net1_overfitting(test_data)
    test_pre_drop = net1_dropped(test_data)
    test_pre_nb = net2_nb(test_data)

    test_loss_ofit = lossfunc(input=test_pre_ofit, target=test_label)
    test_loss_drop = lossfunc(input=test_pre_drop, target=test_label)
    test_loss_nb = lossfunc(input=test_pre_nb, target=test_label)
    # 保存loss的数据与epoch的值
    writer.add_scalars("test_group_loss", {"test_loss_ofit": test_loss_ofit.item(), "test_loss_drop": test_loss_drop.item(), "test_loss_nb":test_loss_nb.item()}, epoch)