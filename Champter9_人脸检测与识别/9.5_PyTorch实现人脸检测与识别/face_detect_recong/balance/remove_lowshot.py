# -*- encoding: utf-8 -*-
"""
@File       : remove_lowshot.py
@Time       : 2023/8/14 17:01
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
import os
import shutil


def remove_low_sample_img(opt):
    """
    删除小于指定张数的图像
    :param opt: 传递的参数
    :return:
    """
    # 获取需要删除图像文件夹的路径
    source_root = opt.source_root
    # remove the classes with less than min_num sample
    min_num= opt.rm_min_num

    # delete '*.DS_Store' existed in the source_root
    cwd = os.getcwd()
    os.chdir(source_root)
    os.system("find . --name '*.DS_Store'  -type f -delete")
    os.chdir(cwd)

    for subfolder in os.listdir(source_root):
        file_num = len(os.listdir(os.path.join(source_root, subfolder)))
        if file_num <= min_num:
            print(f"Class {subfolder} has less than {min_num} samples, removed.")
            shutil.rmtree(os.path.join(source_root, subfolder))

