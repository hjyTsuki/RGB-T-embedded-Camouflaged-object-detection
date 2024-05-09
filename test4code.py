import cv2
import os
import numpy as np
import timm
import torch
from torch import nn
#
# def read_color_array(path: str):
#     assert path.endswith(".jpg") or path.endswith(".png")
#     bgr_array = cv2.imread(path, cv2.IMREAD_COLOR)
#     assert bgr_array is not None, f"Image Not Found: {path}"
#     rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
#     return rgb_array
#
# def read_images_from_path(folder_path):
#     """
#     从指定路径读取所有图片并返回一个列表。
#     :param folder_path: 图片所在文件夹的路径
#     :return: 包含所有图片的列表
#     """
#     image_list = []
#     for filename in os.listdir(folder_path):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#             img_path = os.path.join(folder_path, filename)
#             img = read_color_array(img_path)
#             if img is not None:
#                 image_list.append(img)
#     return image_list
#
# def calculate_mean_and_std(image_list):
#     """
#     计算图像列表的均值和标准差。
#     :param image_list: 图像列表
#     :return: 均值和标准差
#     """
#
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     for X in image_list:
#         for d in range(3):
#             mean[d] += X[:, :, d].mean()/255
#             std[d] += X[:, :, d].std()/255
#     mean.div_(len(image_list))
#     std.div_(len(image_list))
#
#     return mean, std
#
# # 用法示例
# image_folder_path = 'D:\\Yang\\datasets\\VT5000\\VT5000\\Train\\T'
#
# images = read_images_from_path(image_folder_path)
# if len(images) > 0:
#     mean_values, std_values = calculate_mean_and_std(images)
#     print(f"均值 (mean)：{mean_values}")
#     print(f"标准差 (std)：{std_values}")
# else:
#     print("未找到任何图片。请检查文件夹路径是否正确。")

encoder1 = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
encoder2 = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)

x = torch.randn(2, 3, 384, 384)
encoder_shared_level1 = nn.Sequential(encoder1.conv1, encoder1.bn1, encoder1.act1)
ans = encoder1(x)
x1 = encoder_shared_level1(x)
print(ans)



