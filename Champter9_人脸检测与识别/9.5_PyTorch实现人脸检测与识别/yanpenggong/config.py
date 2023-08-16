# -*- encoding: utf-8 -*-
"""
@File       : config.py.py
@Time       : 2023/8/14 17:23
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import torch

configurations = {
    1: dict(
        SEED=1337,  # 随即种子以便复现
        DATA_ROOT="./data/other_my_face_align/others",  # 存储训练/验证/测试数据的父根
        MODEL_ROOT="./model",  # 训练的模型路径
        LOG_ROOT="./log",  # 训练/验证时的日志路径
        BACKBONE_RESUME_ROOT="./buffer/model",  # 预训练模型的路径。这个路径可以是一个本地文件路径，也可以是一个远程URL。
        HEAD_RESUME_ROOT="./buffer/model",  # 模型头部的初始权重路径。通过加载预训练的头部权重，你可以在特定任务上快速地微调模型，而无需从头开始训练整个模型。

        BACKBONE_NAME="IR_SE_50",  # 支持: ["Resnet_50", "Resnet_101", "Resnet_152", "IR_50", "IR_101", "IR_152", "IR_SE_50", "IR_SE_101", "IR_SE_152"]
        HEAD_NAME="ArcFace",  # 支持: ["Softmax", "ArcFace", "CosFace", "ShereFace", "Am_softmax"]
        LOSS_NAME="Focal",  # 支持: ["Focal", "Softmax"]

        INPUT_SIZE=[112, 112],  # 支持: [112, 112] 和 [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # 将输入标准化为 [-1,1]
        RGB_STD=[0.5, 0.5, 0.5],  # 通道上的标准差进行标准化
        EMBEDDING_SIZE=512,  # 特征维度
        BATCH_SIZE=512,  # 训练/验证/测试 的批次大小
        DROP_LAST=True,  # 是否丢弃不足批次大小的数据
        LR=0.1,  # 初始化的学习率
        NUM_EPOCH=125,  # 训练轮数
        WEIGHT_DECAY=5e-4,  # 正则化的权重衰减参数
        MOMENTUM=0.9,  # 动量参数，通常与随机梯度下降（SGD）一起使用
        STAGES=[35, 65, 95],  # 训练过程中不同的阶段或阶段组合，每个阶段可能有不同的训练策略、学习率、数据增强等。可以帮助模型逐步地进行训练，从而提高训练的效果和稳定性。不同的阶段可以适用于不同的训练需求，比如预热（warm-up）、学习率退火（learning rate annealing）等。

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=True,  # 是否在训练深度学习模型时使用多个GPU进行加速。
        GPU_ID=[0, 1, 2, 3],  # 指定GPU id
        PIN_MEMORY=True,  # 是否将加载的数据固定在内存中
        NUM_WORKERS=0  # 并行加载数据的工作进程数量
    )
}