# -*- encoding: utf-8 -*-
"""
@File       : matlab_cp2tform.py
@Time       : 2023/8/14 15:42
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
# import some packages
# --------------------
import numpy as np
from numpy.linalg import lstsq, inv, norm
from numpy.linalg import matrix_rank as rank


def findNonreflectiveSimilarity(uv, xy, options=None):
    options = {"K": 2}

    K = options["K"]
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # 使用reshape以保持是一列向量
    y = xy[:, 1].reshape((-1, 1))  # 使用reshape以保持是一列向量

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))  # 使用reshape以保持是一列向量
    v = uv[:, 1].reshape((-1, 1))  # 使用reshape以保持是一列向量
    U = np.vstack((u, v))

    # 矩阵的秩是否满足要求
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U, rcond=None)
        r = np.squeeze(r)
    else:
        raise Exception("cp2tform two unique points req")

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]
    t_inv = np.array([
        [sc, -ss, 0],
        [ss, sc, 0],
        [tx, ty, 1]
    ])

    # 计算矩阵的逆矩阵
    t = inv(t_inv)
    t[:, 2] = np.array([0, 0, 1])
    return t, t_inv


def tfromfwd(trans, uv):
    """
    应用仿射变换 'trans' 到uv
    :param trans: 3*3 np.array, 变换矩阵
    :param uv: K*2 np.array, 每行或每列都是一对坐标(x,y)
    :return:
        xy: K*2 np.array, 每行或每列都是一对变换坐标(x,y)
    """
    uv = np.hstack((uv, np.ones((uv.shape[0], 1))))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy

def findSimilarity(uv, xy):
    options = {"K": 2}

    # Solve for trans1
    trans1, trans1_inv = findNonreflectiveSimilarity(uv, xy, options)

    # Solve for trans2
    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]
    trans2r, trans2r_inv = findNonreflectiveSimilarity(uv, xyR, options)

    # 手动反射变换以撤消在 xyR 上完成的反射
    TreflectY = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    trans2 = np.dot(trans2r, TreflectY)

    # 指出 trans1 或 trans2 更好
    xy1 = tfromfwd(trans1, uv)
    norm1 = norm(xy1 - xy)  # 计算向量的2范数
    xy2 = tfromfwd(trans2, uv)
    norm2 = norm(xy2 - xy)  # 计算向量的2范数

    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        return trans2, trans2r_inv


def get_similarity_transform(src_pts, dst_pts, reflective=True):
    """
    寻找相似变换矩阵 'trans'
        u = src_pts[:, 0]
        v = src_pts[:, 1]
        x = dst_pts[:, 0]
        y = dst_pts[:, 1]
        [x, y, 1] = [u, v, 1] * trans
    :param src_pts: K*2 np.array, 每行或每列都是一对坐标(x,y)
    :param dst_pts: K*2 np.array, 每行或每列都是一对转换坐标(x,y)
    :param reflective: True or False,
            if True:
                使用反射相似的变换
            else:
                使用非反射相似的变换
    :return:
        :param trans: 3*3 np.array, 从 uv 到 xy 的转换矩阵
        :param trans_inv: 3*3 np.array, trans 的逆，从 xy 到 uv 的变换矩阵
    """
    if reflective:
        trans, trans_inv = findSimilarity(src_pts, dst_pts)
    else:
        trans, trans_inv = findNonreflectiveSimilarity(src_pts, dst_pts)
    return trans, trans_inv


def cvt_tform_for_cv2(trans):
    """
    将变换矩阵 'trans' 转换为 'cv2_trans'，可以直接被cv2.warpAffine()使用:
        u = src_pts[:, 0]
        v = src_pts[:, 1]
        x = dst_pts[:, 0]
        y = dst_pts[:, 1]
        [x, y].T = cv_trans * [u, v, 1].T
    :param trans: 3*3 np.array, 从 uv 到 xy 的转换矩阵
    :return:
        cv2_trans: 2*3 np.array, 从 src_pts 到 dst_pts 的转换矩阵, 可以直接被cv2.warpAffine()使用
    """
    cv2_trans = trans[:, 0:2].T
    return cv2_trans


def get_similarity_transform_for_cv2(src_pts, dst_pts, reflective=True):
    """
    寻找 'cv2_trans' 相似变换矩阵, 可以直接被 cv2.warpAffine() 使用:
        u = src_pts[:, 0]
        v = src_pts[:, 1]
        x = dst_pts[:, 0]
        y = dst_pts[:, 1]
        [x, y].T = cv_trans * [u, v, 1].T
    :param src_pts: K*2 np.array, 每行或每列都是一对坐标(x,y)
    :param dst_pts: K*2 np.array, 每行或每列都是一对转换坐标(x,y)
    :param reflective: True or False,
            if True:
                使用反射相似的变换
            else:
                使用非反射相似的变换
    :return:
        cv2_trans: 2*3 np.array, 从 src_pts 到 dst_pts 变换矩阵，可以直接被 cv2.warpAffine() 使用
    """
    trans, trans_inv = get_similarity_transform(src_pts, dst_pts, reflective)
    cv2_trans = cvt_tform_for_cv2(trans)
    return cv2_trans