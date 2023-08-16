# -*- encoding: utf-8 -*-
"""
@File       : aliign_trans.py
@Time       : 2023/8/14 11:29
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import cv2
# import some packages
# --------------------
import numpy as np
from .matlab_cp2tform import get_similarity_transform_for_cv2


# 参考面部点，坐标列表（x，y）
REFERENCE_FACIAL_POINTS = [  # crop_size=(112, 112) 的默认参考面部点；应针对其他crop_size相应调整 REFERENCE_FACIAL_POINTS
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156],
]

DEFAULT_CROP_SIZE = (96, 112)


class FaceWarpException(Exception):
    def __str__(self):
        return f"In File {__file__}:{super.__str__(self)}"


def get_reference_facial_points(output_size=None,
                                inner_padding_factor=0.0,
                                outer_padding=(0, 0),
                                default_square=False):
    """
    根据作物设置获取参考5个要点
        1. 设置默认的 物体尺寸
            if default_square:
                crop_size = (112, 112)
            else:
                crop_size = (96, 112)
        2. 在每一侧以 crop_size 填充 inner_padding_factor
        3. 将 crop_size 调整为 (output_size -outer_padding * 2)，用outer_padding 填充到output_size
        4. 输出参考5个点
    :param output_size: (w, h) or None, 对齐人脸图像的大小
    :param inner_padding_factor: (w_factor, h_factor), (w, h)内部的填充因子
    :param outer_padding: (w_pad, h_pad), 每行是一对(x, y)坐标
    :param default_square: True or False,
            if True:
                default crop_size = (112, 112)
            else:
                default crop_size = (96, 112)

            notes: 如果output_size 是None，则 (output_size - outer_padding) = some_scale * (default crop_size * (1.0 + inner_padding_factor))
    :return:
        reference_5point: 5*2 np.array, 每行是一对(x, y)坐标
    """
    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    # 0. 将内部区域变成正方形
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    if (output_size and output_size[0] == tmp_crop_size[0] and output_size[1] == tmp_crop_size[1]):
        return tmp_5pts

    if (inner_padding_factor == 0 and outer_padding == (0, 0)):
        if output_size is None:
            return tmp_5pts
        else:
            raise FaceWarpException(f"No padding to do, output_size must be None or {tmp_crop_size}")

    # 检查输出的size
    if not (0 < inner_padding_factor <= 1.0):
        raise FaceWarpException("Not (0 <= inner_padding <= 1.0)")

    if ((inner_padding_factor > 0) or (outer_padding[0] > 0) or (outer_padding[1] > 0)) and (output_size is None):
        output_size = tmp_crop_size * (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)

    if not (outer_padding[0] < output_size[0] and outer_padding[1] < output_size[1]):
        raise FaceWarpException("Not (outer_padding[0] < output_size[0] and outer_padding[1] < output_size[1])")

    # 1) 根据 inner_padding_factor 填充内部区域
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)
    # 2) 调整填充内部区域的大小
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2
    if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]:
        raise FaceWarpException("Must have (output_size - outer_padding) = some_scale * (crop_size * (1.0 + inner_padding_factor))")

    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    tmp_5pts = tmp_5pts * scale_factor
    tmp_crop_size = size_bf_outer_pad

    # 3) 添加 outer_padding 以使 output_size
    reference_5point = tmp_5pts + np.array(outer_padding)
    tmp_crop_size = output_size
    return reference_5point


def get_affine_transform_matrix(src_pts, dst_pts):
    """
    获取输入到输出的仿射变换矩阵 `tfm`
    :param src_pts: K*2 np.array, 原始点矩阵，每行是一对(x, y)坐标
    :param dst_pts: K*2 np.array, 输出点矩阵，每行是一对(x, y)坐标
    :return:
        tfm: 2*3 np.array, 从原始点到输出点的变换矩阵
    """
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    # 求解线性最小二乘问题, 其中
    # =====================================================================
    # np.linalg.lstsq(a, b, rcond='warn')
    #     - a: 系数矩阵，也称为设计矩阵。通常是一个包含特征值的二维数组
    #     - b: 结果向量，也称为观测值。通常是一个包含目标值的一维数组
    #     - rcond（可选）：控制矩阵的秩。默认是'warn'，表示会在矩阵秩过低时发出警告。
    # 该函数返回一个包含以下信息的元组：
    #     - x: 线性最小二乘解
    #     - residuals: 残差（观测值与拟合值之间的差异）
    #     - rank: 矩阵a的秩
    #     - s: a的奇异值
    # =====================================================================
    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

    if rank == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rank == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])
    return tfm


def warp_and_crop_face(src_img, facial_pts, reference_pts=None, crop_size=(96, 112), align_type="similarity"):
    """
    将仿射变换 "trans" 应用于 UV
    :param src_img: 3*3 np.array, 输入的图像
    :param facial_pts: 以下2种
                1. K 坐标 (x,y) 的列表
                2. K*2 或 2*K np.array, 每行或每列都是一对坐标(x,y)
    :param reference_pts:  以下3种
                1. K 坐标 (x,y) 的列表
                2. K*2 或 2*K np.array, 每行或每列都是一对坐标(x,y)
                3. None, 如果是None, 使用默认的面部参考点
    :param crop_size: (w, h), 输出的面部图像大小
    :param align_type: 转换类型，以下3种
                1. "similarity": 使用相似转换
                2. "cv2_affine": 使用前 3 个点进行仿射变换, 使用 `cv2.getAffineTransform()`
                3. "affine": 使用所有点进行仿射变换
    :return:
        face_img: 输出 大小为(w, h)的面部图像数据，即 crop_size
    """
    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        else:
            default_square = False
            inner_padding_factor = 0
            outer_padding = (0, 0)
            outer_size = crop_size

            reference_pts = get_reference_facial_points(outer_size, inner_padding_factor, outer_padding, default_square)

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise FaceWarpException("reference_pts.shape must be (K, 2) or (2, K) and K > 2.")
    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise FaceWarpException("src_pts_shp.shape must be (K, 2) or (2, K) and K > 2.")
    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException("facial_pts and reference_pts must huave the same shape.")

    if align_type == "cv2_affine":
        tfm = cv2.getAffineTransform(src=src_pts[0:3], dst=ref_pts[0:3])
    elif align_type == "affine":
        tfm = get_affine_transform_matrix(src_pts, dst_pts=ref_pts)
    else:
        tfm = get_similarity_transform_for_cv2(src_pts, dst_pts=ref_pts)

    # 执行仿射变换
    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))
    return face_img