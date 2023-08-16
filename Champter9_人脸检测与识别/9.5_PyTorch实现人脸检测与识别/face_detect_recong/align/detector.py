# -*- encoding: utf-8 -*-
"""
@File       : detector.py.py
@Time       : 2023/8/11 10:11
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import numpy as np
import torch

# import some packages
# --------------------
from .get_nets import PNet, RNet, ONet
from .first_stage import run_first_stage
from .box_utils import nms, calibrate_box, convert_to_square, get_image_boxes


def detect_faces(image,
                 min_face_size=20.0,
                 thresholds=[0.6, 0.7, 0.8],
                 nms_thresholds=[0.7, 0.7, 0.7]):
    """
    人脸检测
    :param image: PIL.Image 的一个实例
    :param min_face_size: float
    :param thresholds: a list of length 3
    :param nms_thresholds: a list of length 3
    :return:
        two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
        bounding boxes and facial landmarks.
    """
    # 加载模型
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    onet.eval()

    # 构建图像金字塔
    width, height = image.size
    min_length = min(height, width)

    min_detection_size = 12
    factor = 0.707

    # 缩放图像的比例
    scales = []

    # 缩放图像，以便我们可以检测到的最小尺寸
    # 等于我们想要检测的最小面部尺寸
    m = min_detection_size / min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m * factor**factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1 ====================
    # it will be returned
    bounding_boxes = []

    # run P-Net on different scales
    for s in scales:
        boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
        bounding_boxes.append(boxes)

    # collect boxes (and offsets, and scores) from different scales
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    bounding_boxes = np.vstack(bounding_boxes)

    # 根据非极大抑制筛选边界框
    keep = nms(bounding_boxes[:, 0:5], overlap_threshold=nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]

    # 使用 P-Net 预测的偏移量来变换边界框 -> [n_boxes, 5]
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])

    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 2 ====================
    img_boxes = get_image_boxes(bounding_boxes, image, size=24)
    with torch.no_grad():
        img_boxes = torch.FloatTensor(img_boxes)
        output = rnet(img_boxes)
        offsets = output[0].data.numpy()  # shape [n_boxes, 4]
        probs = output[1].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1, ))
    offsets = offsets[keep]

    keep = nms(bounding_boxes, overlap_threshold=nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 3 ====================
    img_boxes = get_image_boxes(bounding_boxes, image, size=48)
    if len(img_boxes) == 0:
        return [], []
    with torch.no_grad():
        img_boxes = torch.FloatTensor(img_boxes)
        output = onet(img_boxes)
        landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
        offsets = output[1].data.numpy()  # shape [n_boxes, 4]
        probs = output[2].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1, ))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # 计算标志点
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds[2], mode="min")
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        return bounding_boxes, landmarks