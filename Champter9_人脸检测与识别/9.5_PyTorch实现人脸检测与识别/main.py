# -*- encoding: utf-8 -*-
"""
@File       : 9.5_PyTorch实现人脸检测与识别.py
@Time       : 2023/8/11 10:03
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""

# 数据集由两部分构成：
#     - 别人的图像
#     - 另外一个人的图像

# import some packages
# --------------------
import argparse
import os

import cv2
import numpy as np
import time
import traceback
import torch
from torch import nn
from torch.utils import model_zoo
from tqdm import tqdm
from PIL import Image
from face_detect_recong.align.detector import detect_faces
from face_detect_recong.align.visualization_utils import show_results
from face_detect_recong.align.align_trans import get_reference_facial_points, warp_and_crop_face
from face_detect_recong.balance.remove_lowshot import remove_low_sample_img


def get_opt():
    """
    获取参数配置信息
    :return:
    """
    parser = argparse.ArgumentParser(description="面部识别")
    parser.add_argument("--func_id", "-func_id", type=int, default=4, help="函数功能的id")
    parser.add_argument("--source_root", "-source_root", type=str, default="./data/other_my_face/others", help="指定源图像路径")
    parser.add_argument("--dest_root", "-dest_root", type=str, default="./data/temp/other_my_face/others", help="指定源图像识别结果路径")
    parser.add_argument("--crop_size", "-crop_size", type=int, default=128, help="指定对齐面的大小、对齐和裁剪填充")
    parser.add_argument("--rm_min_num", "-rm_min_num", type=int, default=10, help="同种图像的最小数量，达不到则删除")
    opt = parser.parse_args()
    return opt


def valid_demo(opt):
    """1. 验证检测代码"""
    # 0. 读取并展示结果
    img1 = Image.open("./data/other_my_face/my/my/myf112.jpg")
    img2 = Image.open("./data/other_my_face/others/Woody_Allen/Woody_Allen_0002.jpg")
    # 1.1. 验证检测代码
    bounding_boxes, landmarks = detect_faces(img1)  # 检测图像中所有面部的边界框和坐标
    show_results(img1, bounding_boxes, landmarks)  # 展示结果
    # 1.2. 验证检测代码
    bounding_boxes, landmarks = detect_faces(img2)  # 检测图像中所有面部的边界框和坐标
    show_results(img2, bounding_boxes, landmarks)  # 展示结果


def all_img_face_align(opt):
    """2. 批量图像进行人脸对齐识别"""
    # 获取批量图像的数据路径
    source_root = opt.source_root
    # 获取识别出的结构存储路径
    dest_root = opt.dest_root
    # 指定对齐面的大小、对齐和裁剪填充
    crop_size = opt.crop_size
    scale = crop_size / 112
    reference = get_reference_facial_points(default_square=True) * scale

    # 删除 source_root中存在的 ".DS_Store"
    cwd = os.getcwd()
    os.chdir(source_root)
    os.system("find . -name '*.DS_Store' -type f -delete")
    os.chdir(cwd)

    if not os.path.isdir(dest_root):
        os.makedirs(dest_root)

    for subfolder in tqdm(os.listdir(source_root)):
        if not os.path.isdir(os.path.join(dest_root, subfolder)):
            os.mkdir(os.path.join(dest_root, subfolder))
        for image_name in os.listdir(os.path.join(source_root, subfolder)):
            print(f"Processing {os.path.join(source_root, subfolder, image_name)}")
            img = Image.open(os.path.join(source_root, subfolder, image_name))
            try:
                _, landmarks = detect_faces(img)
            except Exception:
                print(f"{os.path.join(source_root, subfolder, image_name)} is discarded due to exception.")
                continue
            if len(landmarks) == 0:  # 如果标记没有被检测出来，则img被丢弃
                print(f"{os.path.join(source_root, subfolder, image_name)} is discarded due to non-detected landmarks.")
                continue
            facial5points = [[landmarks[0][j], landmarks[0][j+5]] for j in range(5)]
            # 仿射变换
            warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
            img_warped = Image.fromarray(warped_face)
            # 保存图像
            img_warped.save(os.path.join(dest_root, subfolder, f"{image_name.split('.')[0]}.png"))


def remove_low_sample_imgs(opt):
    """3. 对检测后进行预处理，删除检测后图像小于 rm_min_num 张的人"""
    remove_low_sample_img(opt)

class Config(object):
    env = "default"
    backbone = "resnet18"
    classify = "softmax"

    metirc = "arc_margin"
    easy_margin = False
    # 是否使用压缩奖惩网络模块(Squeeze and Excitation Blocks)
    use_se = False
    loss = "focal_loss"

    display = False
    finetune = False

    lfw_root = "./data/other_my_face_align/others"
    lfw_test_list = "./data/other_my_face_align/other_test_pair.txt"
    test_model_path = "./data/lfw/resnet18_110.pth"
    save_interval = 10

    train_batch_size = 16
    test_batch_size = 60

    input_shape = (1, 128, 128)

    optimizer = "sgd"

    use_gpu = True
    gpu_id = "0, 1"
    num_workers = 4

    max_epoch = 2
    lr = 1e-1
    lr_step = 10
    lr_decay = 0.95
    weight_decay = 5e-4


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc5 = nn.Linear(512 * 8 * 8, 512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3*3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def resnet_face18(pretrained=False, **kwargs):
    """
    Constructs a ResNet-18 model
    :param pretrained: if True, returns a model pre-trained on ImageNet
    :param kwargs:
    :return:
    """

    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model_urls = {
            "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
            "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
        }
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


def get_lfw_list(pair_list):
    with open(pair_list, "r") as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()
        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def get_features(device, model, test_list, batch_size):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print(f"read {img_path} error.")
        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)
        if image.shape[0] % batch_size == 0 or i == (len(test_list) - 1):
            cnt += 1
            data = torch.from_numpy(images)
            data = data.to(device)
            output = model(data)
            output = output.data.cpu().numpy()

            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None
    return feature, cnt


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, "r") as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)
    acc, th = cal_accuracy(sims, labels)
    return acc, th

def lfw_test(device, model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_features(device, model, img_paths, batch_size)
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    print(f"Accuracy: {acc}, threashold: {th}")
    return acc



def face_recongination(opt):
    """4. 人脸识别"""
    opt = Config()

    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    if opt.backbone == "resnet18":
        model = resnet_face18(opt.use_se)
    elif opt.backbone == "resnet34":
        model = resnet34()
    elif opt.backbone == "resnet50":
        model = resnet50()
    # 采用多GPU多数据并行处理机制
    model = nn.DataParallel(model)
    # 装载预训练模型
    model.load_state_dict(torch.load(opt.test_model_path, map_location=device))
    model.to(device)

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    model.eval()
    lfw_test(device, model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)


global function_dict
# 创建字典，键是索引，值是函数的名称
function_dict = {
    # bash : `python main.py --func_id 1`
    1: 'valid_demo',  # 验证检测代码
    # bash : `python main.py --func_id 2 --source_root "./data/other_my_face/others" --dest_root "./data/temp/other_my_face/others" --crop_size 128`
    # bash : `python main.py --func_id 2 --source_root "./data/other_my_face/my" --dest_root "./data/temp/other_my_face/my" --crop_size 128`
    2: 'all_img_face_align',  # 批量图像检测
    # bash : `python main.py --func_id 3 --source_root "./data/temp/other_my_face/others" --rm_min_num 4`
    3: "remove_low_sample_imgs",  # 对检测后进行预处理，删除检测后图像小于 rm_min_num 张的人
    4: "face_recongination",  # 人脸识别
}


def main():
    # 获取参数
    opt = get_opt()

    # 获取函数索引
    func_id = opt.func_id
    # 从字典中获取函数的名称
    function_name = function_dict.get(func_id)
    if function_name:
        try:
            # 使用 eval() 调用函数
            function = eval(function_name)
            function(opt)  # 调用函数
        except Exception as e:
            print("<---Error ,with eval function--->\n", traceback.format_exc())
    else:
        print(f"<---Function not found for index:{func_id}--->")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"<---all run with {(end_time - start_time):.6f}ms--->")