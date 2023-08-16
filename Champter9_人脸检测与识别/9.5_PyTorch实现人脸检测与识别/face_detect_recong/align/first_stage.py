# -*- encoding: utf-8 -*-
"""
@File       : first_stage.py
@Time       : 2023/8/11 15:09
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import math

import numpy as np
import torch
from PIL import Image
from .box_utils import _preprocess, nms

# import some packages
# --------------------



def run_first_stage(image, net, scale, threshold):
    """
    Run P-Net, generate bounding boxes, and do NMS
    :param image: an instance oof PIL.Image
    :param net: an instance of pytorch's nn.Module, P-Net
    :param scale: float, scale wiidth and height of the image by this number
    :param threshold: float, 从网络预测生成边界框时人脸概率的阈值
    :return:
        a float numpy array of shape [n_boxes, 9],
        bounding boxes with scores and offsets (4+1+4)
    """
    # 缩放图像并将其转换为浮点数组
    width, height = image.size
    sw, sh = math.ceil(width * scale), math.ceil(height * scale)
    img = image.resize((sw, sh), Image.BILINEAR)
    img = np.asarray(img, "float32")
    with torch.no_grad():
        img = torch.FloatTensor(_preprocess(img))
        output = net(img)
        # 每个滑动窗口处出现人脸的概率
        probs = output[1].data.numpy()[0, 1, :, :]
        # 转换为真实边界框
        offsets = output[0].data.numpy()

        boxes = _generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None

        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]


def _generate_bboxes(probs, offsets, scale, threshold):
    """
    在可能有人脸的地方生成边界框
    :param probs: a float numpy array of shape [n, m]
    :param offsets: a float numpy array of shape [1, 4, n, m]
    :param scale: float, 图像的宽度和高度按此数字缩放
    :param threshold: float
    :return:
        a float numpy array of shape [n_boxes, 9]
    """
    # 应用 P-Net, 以步长 2 移动 12*12 窗口
    stride = 2
    cell_size = 12

    # 获取可能是人脸的边界框索引
    inds = np.where(probs > threshold)

    if inds[0].size == 0:
        return np.array([])

    # 边界框的变换
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    # 它们是这样定义的：
    #     w= x2 - x1 + 1
    #     h= y2 - y1 + 1
    #     x1_true = x1 + tx1 * w
    #     x2_true = x2 + tx2 * w
    #     y1_true = y1 + ty1 * h
    #     y2_true = y2 + ty2 * h

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    # P-Net 应用于缩放图像, 所以需要重新缩放边界框
    bounding_boxes = np.vstack([
        np.round((stride * inds[1] + 1.0) / scale),
        np.round((stride * inds[0] + 1.0) / scale),
        np.round((stride * inds[1] + 1.0 + cell_size) / scale),
        np.round((stride * inds[0] + 1.0 + cell_size) / scale),
        score, offsets
    ])
    return bounding_boxes.T