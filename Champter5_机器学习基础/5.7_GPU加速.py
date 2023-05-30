# -*- encoding: utf-8 -*-
'''
@File    :   5.7_GPU加速.py
@Time    :   2022/04/12 16:12:29
@Author  :   yanpenggong
@Version :   1.0
@Email   :   yanpenggong@163.com
@Copyright : 侵权必究
'''

# here put the import lib
# ----------------------------------------------------------------------------------------------------------------------
# 5.7.1 单GPU加速
# 使用GPU之前，需要确保GPU是可以使用的，可通过torch.conda.is_available()方法的返回值来进行判断。返回True则具有能够使用的GPU。
# 通过torch.cuda.device_count() 方法可以获得能够使用的GPU数量。

# 如何查看平台GPU的配置信息？在命令行输入命令"nvidia-smi" 即可(适合于linux或Windows环境)。

# 把数据从内存转到GPU，一般针对张量(我们需要的数据)和模型。对张量(类型为FloatTensor/LongTensor等)，一律直接使用方法 ".to(device)"或".cuda()"
# import torch
# device = torch.device("cuda:0" if torch.is_available() else "cpu")
# # 或者 device = torch.device("cuda:0")
# device1 = torch.device("cuda:1")
# for batch_idx, (img, label) in enumerate(train_loader):
#     img = img.to(device)
#     label = label.to(device)

# 对模型来说，也是同样的方式，使用 ".to(device)"或".cuda"来将网络放到GPU显存。
# mode = Net()  # 实例化网络
# model.to(device)  # 使用序号为0的GPU
# # 或 model.to(device1)  # 使用序号为1的GPU

# ----------------------------------------------------------------------------------------------------------------------
# 5.7.2 多GPU加速
# 这里主要介绍单主机多GPU的情况，单机多GPU主要采用了 DataParallel函数，而不是 DistributedParallel，后者一般用于多主机多GPU，当然也可以用于单机多GPU。

# 使用多卡训练的方式有很多，当然前提是我们的设备中存在两个及以上的GPU。
# 使用时直接使用 model 传入 torch.nn.DataParallel函数即可。代码如下：
# import torch
# net = torch.nn.DataParallel(model)  # 模型传入
# 这时，默认所有存在的显卡都会被使用。
# 如果你的电脑有很多显卡，但只想利用其中一部分，如只使用编号为 0、 1、 3、 4的4个GPU，那么可以采用以下方式：
# device_ids = [0, 1, 2, 3]  # 假设有4个GPU，其id设置
# input_data = input_data.to(device=device_ids[0])  # 数据传入
# net = torch.nn.DataParallel(model)  # 模型传入
# net.to(device)
# 或者
# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, [0, 1, 2, 3]))
# net = torch.nn.DataParallel(model)
# 其中 CUDA_VISIBLE_DEVICES 表示当前可以被 Pytorch 程序检测到的GPU。

# ----------------------------------------------------------------------------------------------------------------------
# 下面为单机多GPU的实现代码：
# 1. 背景说明
# 这里使用波士顿房价数据为例，共506个样本，13个特征。数据划分为训练集和测试集，然后用 data.DataLoader 转换为可批加载的方式。
# 采用 nn.Dataparallel 并发机制，环境有2个GPU。当然，数据量很小，按理不宜用 nn.DataParallel，这里只是为例更好地说明使用方法。
# 2. 加载数据
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F

boston = load_boston()
X, y = (boston.data, boston.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# 组合训练数据及标签
myset = list(zip(X_train, y_train))
# 3. 把数据转换为批处理加载方式。批次大小为128， 打乱数据。
from torch.utils import data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.FloatTensor
train_loader = data.DataLoader(dataset=myset, batch_size=128, shuffle=True)

# 4. 定义网络
class Net1(nn.Module):
    """
    使用sequential 构建网络，Sequential()函数的功能是将网络的层组合到一起
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net1, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
    
    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        x1 = F.relu(self.layer2(x1))
        x2 = self.layer3(x1)
        # 显示每个GPU分配的数据大小
        print(f"\tIn Model: input_size {x.size()}, output_size {x2.size()}")
        return x2

# 5. 把模型转换为多GPU并发处理格式
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 实例化网络
model = Net1(in_dim=13, n_hidden_1=16, n_hidden_2=32, out_dim=1)
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs")
    # dim = 0 [64, xxx] -> [32, ...], [32, ...] on 2GPUs
    model = nn.DataParallel(model)
model.to(device)

# 6. 选择优化器及损失函数
optimizer_orig = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.MSELoss()

# 7.模型训练，并可视化损失值
from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir="logs")
for epoch in range(100):
    model.train()
    for data, label in train_loader:
        input = data.type(dtype).to(device)
        label = label.type(dtype).to(device)
        output = model(input)
        loss = loss_func(output, label)
        # 反向传播
        optimizer_orig.zero_grad()
        loss.backward()
        optimizer_orig.step()
        print(f"Outside: input_size {input.size()}, output_size {output.size()}")
    writer.add_scalar("train_loss_paral", loss, epoch)

# # 8.通过web查看损失值的变化情况。
# 单机多GPU也可以使用 DistributedParallel，它多用于分布式训练，但也可以用在单机多GPU的训练，配置比使用 nn.DataParallel 稍微麻烦点，但是训练速度和效果更好一点。
# 具体配置为：
#     # 初始化使用nccl后端
#     torch.distributed.init_process_group(backend="nccl")
#     # 模型并行化
#     model = torch.nn.parallels.DistributedDataParallel(model)
#     # 单机运行时使用下列方法启动
#     python -m torch.distributed.launch main.py

# ----------------------------------------------------------------------------------------------------------------------
# 5.7.3 使用GPU注意事项
# 使用GPU可以提升训练的速度，但如果使用不当，可能影响使用效率，具体使用时要注意以下几点：
#     - GPU 的数量尽量为偶数，奇数的GPU有可能会出现异常中断的情况
#     - GPU 很快，但数据量较小时，效果可能没有单GPU好，甚至还不如CPU
#     - 如果内存不够大，使用多GPU训练的时候可以设置 pin_memory 为False，当然使用精度稍微低一点的数据类型有时也有效果。