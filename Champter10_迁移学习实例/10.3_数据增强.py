# -*- encoding: utf-8 -*-
"""
@File       : 10.3_数据增强.py
@Time       : 2023/8/17 17:55
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
# 提高模型的泛化能力最重要的3大因素是数据、模型和损失函数，其中数据又是3个因素中最重要的因素。
# 但数据的获取往往不充分或成本比较高。换用其它方法，既快速又便捷地增加数据量，如图像识别、语言识别等等。
# 数据增强技术来增加数据量:
#     - 水平、垂直翻转图像
#     - 裁剪
#     - 色彩变换
#     - 扩展
#     - 旋转
#
# 通过数据增强(Data Augmentation)技术不仅可以扩大训练数据集的规模、降低模型对某些属性的依赖，从而提高模型的泛化能力，同时可以对图像进行不同方式的裁剪，
# 时感兴趣的物体出现在不同的位置，从而减轻模型对物体出现位置的依赖性。并通过调整亮度、色彩等因素来降低模型对色彩的敏感度。
#
# 当然对图像左这些预处理时，不宜使用会改变其类别的转换。如手写的数字，如果使用旋转90度，就有可能把9变成6，或者把6变成9，把随机噪声添加到输入数据或隐藏单元中也是方法之一。
#

def show_img(img_tensor):
    """将张量转换为 NumPy 数组并展示"""
    image_np = img_tensor.numpy().transpose(1, 2, 0)  # 将通道维度移到最后
    image_cv2 = np.uint8(image_np * 255)  # 将值映射到 0-255 范围
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)  # 转换为 BGR 格式
    return image_cv2


# 1. 按比例缩放=================================
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
img = Image.open("./data/cat/cat.jpg")
print(f"原图像的大小: {img.size}")
new_img = transforms.Resize((100, 200))(img)
print(f"缩放后图像的大小: {new_img.size}")
# 水平合并图像
merged_image_horizontal = Image.new("RGB", (img.width + new_img.width, max(img.height, new_img.height)))
merged_image_horizontal.paste(img, (0, 0))
merged_image_horizontal.paste(new_img, (img.width, 0))
cv2.imshow("1.merged_image_horizontal", cv2.cvtColor(np.array(merged_image_horizontal), cv2.COLOR_RGB2BGR))

# 2. 裁剪=================================
# 随机裁剪有两种方式：
#     - 对图像在随机位置进行截取，可传入裁剪大小，使用函数为: torchvision.transforms.RandomCrop()
#     - 在中心，按比例裁剪，函数为: torchvision.transforms.CenterCrop()
# 随机裁剪出200*200的区域
random_img1 = transforms.RandomCrop(200)(img)
cv2.imshow("2.random_img1", cv2.cvtColor(np.array(random_img1), cv2.COLOR_RGB2BGR))

# 3. 翻转=================================
# 翻转猫还是猫，不会改变其类别。通过翻转图像可以增加其多样性，所以随机翻转也是一种非常有效地手段。
# 随机翻转的方法：
#     - 随机水平翻转: torchvision.transforms.RandomHorizontalFlip()
#     - 随机垂直翻转: torchvision.transforms.RandomVerticalFlip()
#     - 随机旋转: torchvision.transforms.RandomRotation()
transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),  # p=1.0 表示始终进行水平翻转
    transforms.ToTensor()
])
h_flip = transform1(img)

transform2 = transforms.Compose([
    transforms.RandomVerticalFlip(p=1.0),  # p=1.0 表示始终进行水平翻转
    transforms.ToTensor()
])
v_flip = transform2(img)

transform3 = transforms.Compose([
    transforms.RandomRotation(degrees=(-30, 30)),  # 在 -30 到 30 度范围内进行随机旋转
    transforms.ToTensor()
])
r_flip = transform3(img)

# 使用 OpenCV 显示图像
cv2.imshow("3.RandomHorizontalFlip", show_img(h_flip))
cv2.imshow("3.RandomVerticalFlip", show_img(v_flip))
cv2.imshow("3.RandomRotation", show_img(r_flip))


# 4.改变颜色=================================
# 除了形状变化外，颜色变化又是另外一种增强方式，其可设置为：亮度变化、对比度变化、颜色变化等。
# torchvision.transforms.ColorJitter()
#     - brightness：控制亮度扰动的范围，取值为一个表示扰动强度的浮点数或一个范围 (min, max)。值为 0 表示没有亮度扰动，值为 1 表示最大扰动。
#     - contrast：控制对比度扰动的范围，取值为一个表示扰动强度的浮点数或一个范围 (min, max)。值为 0 表示没有对比度扰动，值为 1 表示最大扰动。
#     - saturation：控制饱和度扰动的范围，取值为一个表示扰动强度的浮点数或一个范围 (min, max)。值为 0 表示没有饱和度扰动，值为 1 表示最大扰动。
#     - hue：控制色调扰动的范围，取值为一个表示扰动强度的浮点数或一个范围 (min, max)。值为 0 表示没有色调扰动，值为 0.5 表示最大扰动。
transform4 = transforms.Compose([
    transforms.ColorJitter(hue=0.5),  # 随机从 [-0.5, 0.5] 之间对颜色变化
    transforms.ToTensor()
])
color_img = transform4(img)
cv2.imshow("4.ColorJitter", show_img(color_img))


# 5.组合多种增强方法
# 可以使用 torchvision.transforms.Compose() 函数把上述变化组合在一起
transform5 = transforms.Compose([
    transforms.Resize(200),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(96),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    # transforms.ToTensor()
])
compose_img = [transform5(img) for i in range(10)]
# 将经过变换的图像按 3x3 排列
rows = len(compose_img) // 3 + (len(compose_img) % 3 > 0)
cols = 3
combined_image = np.zeros((rows * compose_img[0].height, cols * compose_img[0].width, 3), dtype=np.uint8)
# 填充合并的图像
for i in range(rows):
    for j in range(cols):
        idx = i * cols + j
        if idx >= len(compose_img):
            continue
        combined_image[i * compose_img[0].height: (i + 1) * compose_img[0].height,
                       j * compose_img[0].width: (j + 1) * compose_img[0].width] = np.array(compose_img[idx])
cv2.imshow("5.Compose", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()