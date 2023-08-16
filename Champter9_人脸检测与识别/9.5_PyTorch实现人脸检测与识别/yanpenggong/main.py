# -*- encoding: utf-8 -*-
"""
@File       : main.py
@Time       : 2023/8/15 09:28
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import os
import torch
from torch import nn
import torch.utils.data
from torch import optim
from torchvision import transforms
from torchvision import datasets
from tensorboardX import SummaryWriter
from config import configurations
from utils.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, separate_resnet_bn_paras, schedule_lr, AverageMeter, warm_up_lr,  accuracy, perform_val, buffer_val, get_time
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from .backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace,CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from tqdm import tqdm


def main():
    # 1. 加载配置信息和数据
    cfg = configurations[1]

    SEED = cfg["SEED"]  # 随机种子
    torch.manual_seed(seed=SEED)

    DATA_ROOT = cfg["DATA_ROOT"]  # 存储训练/验证/测试数据的父根
    MODEL_ROOT = cfg["MODEL_ROOT"]  # 训练的模型路径
    LOG_ROOT = cfg["LOG_ROOT"]  # 训练/验证时的日志路径
    BACKBONE_RESUME_ROOT = cfg["BACKBONE_RESUME_ROOT"]  # 预训练模型的路径。这个路径可以是一个本地文件路径，也可以是一个远程URL。
    HEAD_RESUME_ROOT = cfg["HEAD_RESUME_ROOT"]  # 模型头部的初始权重路径。通过加载预训练的头部权重，你可以在特定任务上快速地微调模型，而无需从头开始训练整个模型。

    BACKBONE_NAME = cfg["BACKBONE_NAME"]  # 支持: ["ResNet_50", "ResNet_101", "ResNet_152", "IR_50", "IR_101", "IR_152", "IR_SE_50", "IR_SE_101", "IR_SE_152"]
    HEAD_NAME = cfg["HEAD_NAME"]  # 支持: ["Softmax", "ArcFace", "CosFace", "ShereFace", "Am_softmax"]
    LOSS_NAME = cfg["LOSS_NAME"]  # 支持: ["Focal", "Softmax"]

    INPUT_SIZE = cfg["INPUT_SIZE"]  # 支持: [112, 112] 和 [224, 224]
    RGB_MEAN = cfg["RGB_MEAN"]  # 将输入标准化为 [-1,1]
    RGB_STD = cfg["RGB_STD"]  # 通道上的标准差进行标准化
    EMBEDDING_SIZE = cfg["EMBEDDING_SIZE"]  # 特征维度
    BATCH_SIZE = cfg["BATCH_SIZE"]  # 训练/验证/测试 的批次大小
    DROP_LAST = cfg["DROP_LAST"]  # 是否丢弃不足批次大小的数据
    LR = cfg["LR"]  # 初始化的学习率
    NUM_EPOCH = cfg["NUM_EPOCH"]  # 训练轮数
    WEIGHT_DECAY = cfg["WEIGHT_DECAY"]  # 正则化的权重衰减参数
    MOMENTUM = cfg["MOMENTUM"]  # 动量参数，通常与随机梯度下降（SGD）一起使用
    STAGES = cfg["STAGES"]  # 训练过程中不同的阶段或阶段组合，每个阶段可能有不同的训练策略、学习率、数据增强等。可以帮助模型逐步地进行训练，从而提高训练的效果和稳定性。不同的阶段可以适用于不同的训练需求，比如预热（warm-up）、学习率退火（learning rate annealing）等。

    DEVICE = cfg["DEVICE"]
    MULTI_GPU = cfg["MULTI_GPU"]  # 是否在训练深度学习模型时使用多个GPU进行加速。
    GPU_ID = cfg["GPU_ID"]  # 指定GPU Id
    PIN_MEMORY = cfg["PIN_MEMORY"]  # 是否将加载的数据固定在内存中
    NUM_WORKERS = cfg["NUM_WORKERS"]  # 并行加载数据的工作进程数量
    print("="*60)
    print("Overall Configurations:")
    print(cfg)
    print("="*60)

    writer = SummaryWriter(LOG_ROOT)  # 用于缓冲中间结果的编写器

    train_transform = transforms.Compose([  # 内置在线数据增强
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]),  # 较小的一侧已调整大小
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),  # 随机裁剪图像
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
    ])

    dataset_train = datasets.ImageFolder(os.path.join(DATA_ROOT, "imgs"), train_transform)

    # 创建加权随机采样器来处理不平衡数据
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, sampler=sampler,
                                               pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=DROP_LAST)

    NUM_CLASS = len(train_loader.dataset.classes)
    print(f"NUmber of training classes: {NUM_CLASS}")

    lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, vgg2_fp, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame, calfw_issame, cplfw_issame, vgg2_fp_issame = get_val_data(DATA_ROOT)

    # 2. model & loss & optimizer
    BACKBONE_DICT = {
        "ResNet_50": ResNet_50(INPUT_SIZE),
        "ResNet_101": ResNet_101(INPUT_SIZE),
        "ResNet_152": ResNet_152(INPUT_SIZE),
        "IR_50": IR_50(INPUT_SIZE),
        "IR_101": IR_101(INPUT_SIZE),
        "IR_152": IR_152(INPUT_SIZE),
        "IR_SE_50": IR_SE_50(INPUT_SIZE),
        "IR_SE_101": IR_SE_101(INPUT_SIZE),
        "IR_SE_152": IR_SE_152(INPUT_SIZE),
    }
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    print("=" * 60)
    print(BACKBONE)
    print(f"{BACKBONE_NAME} Backbone generated.")
    print("=" * 60)

    HEAD_DICT = {
        "ArcFace": ArcFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
        "CosFace": CosFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
        "SphereFace": SphereFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
        "Am_softmax": Am_softmax(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
    }
    HEAD = HEAD_DICT[HEAD_NAME]
    print("=" * 60)
    print(HEAD)
    print(f"{HEAD_NAME} Head generated.")
    print("=" * 60)

    LOSS_DICT = {
        "Focal": FocalLoss(),
        "Softmax": nn.CrossEntropyLoss()
    }
    LOSS = LOSS_DICT[LOSS_NAME]
    print("=" * 60)
    print(LOSS)
    print(f"{LOSS_NAME} Loss generated.")
    print("=" * 60)

    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE)  # 将batch_norm参数与其他参数分开，不对batch_norm参数进行权重衰减以提高泛化性
        _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE)  # 将batch_norm参数与其他参数分开，不对batch_norm参数进行权重衰减以提高泛化性
        _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)
    OPTIMIZER = optim.SGD([{"params": backbone_paras_wo_bn + head_paras_wo_bn, "weight_decay": WEIGHT_DECAY}, {"params": backbone_paras_only_bn}], lr=LR, momentum=MOMENTUM)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer generated.")
    print("=" * 60)


    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT and HEAD_RESUME_ROOT:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT) and os.path.isfile(HEAD_RESUME_ROOT):
            print(f"Loading backbone Checkpoint '{BACKBONE_RESUME_ROOT}'")
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
            print(f"Loading Head Checkpoint '{HEAD_RESUME_ROOT}'")
            HEAD.load_state_dict(torch.load(HEAD_RESUME_ROOT))
        else:
            print(f"No Checkpoint found at '{BACKBONE_RESUME_ROOT}' and '{HEAD_RESUME_ROOT}'.Please Have a check or continue to train from scratch.")
        print("=" * 60)

    # GPU 设置
    if MULTI_GPU:
        # 多GPU设置
        BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
        BACKBONE = BACKBONE.to(device=DEVICE)
    else:
        # 单GPU设置
        BACKBONE = BACKBONE.to(device=DEVICE)

    # train & validation & save checkpoint
    DISP_FREQ = len(train_loader) // 100  # 显示训练损失和准确性的频率
    NUM_EPOCH_WARM_UP = NUM_EPOCH // 25  # 使用第一个1/25 epochs 进行预热
    NUM_EPOCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # 使用第一个1/25 epochs 进行预热
    batch = 0  # batch 索引

    for epoch in range(NUM_EPOCH):  # 开始训练
        if epoch == STAGES[0]:  # 对任何 预热后的训练stage进行调整LR，一旦观察到 不好，您也可以选择手动调整 LR（稍作修改）
            schedule_lr(OPTIMIZER)
        elif epoch == STAGES[1]:
            schedule_lr(OPTIMIZER)
        elif epoch == STAGES[2]:
            schedule_lr(OPTIMIZER)

        # 设置训练模式
        BACKBONE.train()
        HEAD.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for inputs, labels in tqdm(iter(train_loader)):
            # adjust LR for each training batch during warm up
            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (batch + 1 <= NUM_EPOCH_WARM_UP):
                warm_up_lr(batch + 1, NUM_EPOCH_WARM_UP, LR, OPTIMIZER)

            # 计算输出
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            features = BACKBONE(inputs)
            outputs = HEAD(features, labels)
            loss = LOSS(outputs, labels)

            # 确定准确率和记录损失
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))
            top5.update(prec5.data.item(), inputs.size(0))

            # 计算梯度并执行 SGD 步骤
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            # 展示每个 DISP_FREQ 训练损失和准确率
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                print("="*60)
                print(f"Epoch {epoch+1}/{NUM_EPOCH} Batch {batch+1}/{len(train_loader)*NUM_EPOCH}\t"
                      f"Training Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                      f"Training Prec@1 {top1.val:.3f} ({top1.val:.3f})\t"
                      f"Training Prec@5 {top5.val:.3f} ({top5.val:.3f})\t")
                print("="*60)

            batch += 1

        # 每步训练的统计信息(用于可视化的缓冲区)
        epoch_loss = losses.avg
        epoch_acc = top1.avg
        writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
        print("=" * 60)
        print(f"Epoch {epoch + 1}/{NUM_EPOCH}\t"
              f"Training Loss {losses.val:.4f} ({losses.avg:.4f})\t"
              f"Training Prec@1 {top1.val:.3f} ({top1.val:.3f})\t"
              f"Training Prec@5 {top5.val:.3f} ({top5.val:.3f})\t")
        print("=" * 60)

        # 执行验证并保存每个时期的检查点
        # 每个时期的验证统计数据（用于可视化的缓冲区）
        print("=" * 60)
        print("Perform Evaluation on LFW, CFP_FF, CFP_PP, AgeDB, CALFW, CPLFW and VGG2_FP, and Save Checkpoints...")
        accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame)
        buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch + 1)
        accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_ff, cfp_ff_issame)
        buffer_val(writer, "CFP_FF", accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff, epoch + 1)
        accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_fp, cfp_fp_issame)
        buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch + 1)
        accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, agedb, agedb_issame)
        buffer_val(writer, "AgeDB", accuracy_agedb, best_threshold_agedb, roc_curve_agedb, epoch + 1)
        accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, calfw, calfw_issame)
        buffer_val(writer, "CALFW", accuracy_calfw, best_threshold_calfw, roc_curve_calfw, epoch + 1)
        accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cplfw, cplfw_issame)
        buffer_val(writer, "CPLFW", accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw, epoch + 1)
        accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, vgg2_fp, vgg2_fp_issame)
        buffer_val(writer, "VGGFace2_FP", accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp, epoch + 1)
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCH},\t"
            f"Evaluation: LFW Acc: {accuracy_lfw},\t"
            f"CFP_FF Acc: {accuracy_cfp_ff},\t"
            f"CFP_FP Acc: {accuracy_cfp_fp},\t"
            f"AgeDB Acc: {accuracy_agedb},\t"
            f"CALFW Acc: {accuracy_calfw},\t"
            f"CPLFW Acc: {accuracy_cplfw},\t"
            f"VGG2_FP Acc: {accuracy_vgg2_fp}")
        print("=" * 60)

        # 保存每epoch 的检查点
        if MULTI_GPU:
            torch.save(BACKBONE.module.state_dict(), os.path.join(MODEL_ROOT, f"Backbone_{BACKBONE_NAME}_Epoch_{epoch+1}_Batch_{batch}_Time_{get_time()}_checkpoint.pth"))
        else:
            torch.save(BACKBONE.state_dict(), os.path.join(MODEL_ROOT, f"Backbone_{BACKBONE_NAME}_Epoch_{epoch+1}_Batch_{batch}_Time_{get_time()}_checkpoint.pth"))
        torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT, f"Head_{HEAD_NAME}_Epoch_{epoch+1}_Batch_{batch}_Time_{get_time()}_checkpoint.pth"))


if __name__ == '__main__':
    main()