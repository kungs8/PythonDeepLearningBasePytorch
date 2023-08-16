# -*- encoding: utf-8 -*-
"""
@File       : box_utils.py
@Time       : 2023/8/11 15:36
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import numpy as np
from PIL import Image

# import some packages
# --------------------
def _preprocess(img):
    """
    喂给网络模型的预处理步骤
    :param img: a float numpy array of shape [h, w, c]
    :return:
        a float numpy array of shape [1, c, h, w]
    """
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5) * 0.0078125
    return img


def nms(boxes, overlap_threshold=0.5, mode="union"):
    """
    Non-maximum suppression(非极大抑制)
    :param boxes: a float numpy array of shape [n, 5],where each row is (xmin, ymin, xmax, ymax, score)
    :param overlap_threshold: float
    :param mode: "union" or "min"
    :return:
        list with indices of the selected boxes
    """
    # 如果没有检测框，返回空list
    if len(boxes) == 0:
        return []

    # 所选索引列表
    pick = []

    # 获取边界框的坐标
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
    # 计算面积
    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)  # 按递增序列

    while len(ids) > 0:
        # 获取最大值的索引
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # 计算得分最大的框与其余框的交集
        # 交叉框的左上角
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])
        # 交叉框的右下角
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])
        # 交叉框的宽、高
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # 交叉面积
        inter = w * h
        if mode == "min":
            overlap = inter / np.minimum(area[i], area[ids[:last]])
        elif mode == "union":
            # 并集上的交集(IoU)
            overlap = inter / (area[i] + area[ids[:last]] - inter)

        # 删除重叠太大的所有框
        ids = np.delete(
            ids,
            np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )
    return pick


def calibrate_box(bboxes, offsets):
    """
    将边界框转换为更像真实的边界框, "offsets" 是网络的输出之一
    :param bboxes: a float numpy array od shape [n, 5]
    :param offsets: a float numpy array od shape [n, 4]
    :return:
        a float numpy array od shape [n, 5]
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    translation = np.hstack([w, h, w, h]) * offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes


def convert_to_square(bboxes):
    """
    将边界框转换为正方形
    :param bboxes: a float numpy array of shape [n, 5]
    :return:
        a float numpy array of shape [n, 5]
    """
    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes


def correct_bboxes(bounding_boxes, width, height):
    """
    裁剪太大的框并获得相对于切口的坐标
    :param bounding_boxes: a float numpy array of shape [n, 5], where each row is (xmin, ymin, xmax, ymax, score)
    :param width: float
    :param height: float
    :return:
        dy, dx, edy, edx: a int numpy arrays of shape [n], 相对于切口的坐标
        y, x, ey, ex: a int numpy arrays of shape [n], 角点 ymin, xmin, ymax, xmax
        h, w: a int numpy arrays of shape [n], 只是边界框的高度和宽度
        按照以下顺序：[dy, edy, dx, edx, y, ey, x, ex, w, h]
    """
    x1, y1, x2, y2 = [bounding_boxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    num_boxes = bounding_boxes.shape[0]

    # "e" stands for end
    # (x, y) -> (ex, ey)
    x, y, ex, ey = x1, y1, x2, y2

    # 从图像中切出框
    # (x, y, ex, ey) 是 图像中框的修正坐标
    # (dx, dy, edx, edy) 是 图像中框的坐标
    dx = np.zeros((num_boxes,))
    dy = np.zeros((num_boxes,))
    edx = w.copy() - 1.0
    edy = h.copy() - 1.0

    # 盒子的右下角太靠右
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 -ex[ind]
    ex[ind] = width - 1.0
    # 如果盒子的右下角太低
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0
    # 盒子的右下角太靠左
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0
    # 如果盒子的右下角太高
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype("int32") for i in return_list]
    return return_list


def get_image_boxes(bounding_boxes, image, size=24):
    """
    从图像中剪出方框
    :param bounding_boxes: a float numpy array of shape [n, 5]
    :param image: an instance of PIL.Image
    :param size: int, 切口尺寸
    :return:
        a float numpy array of shape [n, 3, size, size]
    """
    num_boxes = len(bounding_boxes)
    width, height = image.size

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bounding_boxes, width, height)
    img_boxes = np.zeros((num_boxes, 3, size, size), "float32")

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), "uint8")
        img_array = np.asarray(image, "uint8")
        img_box[dy[i]:(edy[i]+1), dx[i]:(edx[i]+1), :] = img_array[y[i]:(ey[i]+1), x[i]:(ex[i]+1), :]

        # resize
        img_box = Image.fromarray(img_box)
        img_box = img_box.resize((size, size), Image.BILINEAR)
        img_box = np.asarray(img_box, "float32")
        img_boxes[i, :, :, :] = _preprocess(img_box)
    return img_boxes