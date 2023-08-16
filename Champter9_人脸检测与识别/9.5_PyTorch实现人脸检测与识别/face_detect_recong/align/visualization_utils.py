# -*- encoding: utf-8 -*-
"""
@File       : visualization_utils.py
@Time       : 2023/8/14 09:59
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""

# import some packages
# --------------------
from PIL import ImageDraw


def show_results(img, bounding_boxes, facial_landmarks=[]):
    """
    绘制边界框和面部标志
    :param img: an instance of PIL.Image
    :param bounding_boxes: a float numpy array of shape [n, 5]
    :param facial_landmarks: a float numpy array of shape [n, 10]
    :return:
        an instance of PIL.Image
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline="blue", width=3)

    inx = 0
    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([
                (p[i] - 1.0, p[i+5] - 1.0),
                (p[i] + 1.0, p[i+5] + 1.0),
            ], outline="red", width=5)
    img_copy.show()
    return img_copy