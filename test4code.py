import cv2
import os
import numpy as np
import timm
import torch
from einops import rearrange
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
#
# encoder1 = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
# encoder2 = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
#
# x = torch.randn(2, 3, 384, 384)
# encoder_shared_level1 = nn.Sequential(encoder1.conv1, encoder1.bn1, encoder1.act1)
# ans = encoder1(x)
# x1 = encoder_shared_level1(x)
# print(ans)
#
# device_ids = [i for i in range(torch.cuda.device_count())]
# if torch.cuda.device_count() > 1:
#     print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B, -1, self.output_dim)
        x= self.norm(x)
        return x

# model_list = timm.list_models()
# for name in model_list:
#     print(name)
 # features_only=True
x = torch.randn(1, 3, 384, 384)
encoder1 = timm.create_model(model_name="convnextv2_base.fcmae_ft_in22k_in1k_384", pretrained=False, in_chans=3, features_only=True)
a = encoder1(x)
encoder_shared_level1 = nn.Sequential(encoder1.patch_embed, encoder1.stages[0])
encoder_shared_level2 = nn.Sequential(encoder1.stages[1])
encoder_rgb_private_level3 = encoder1.stages[2]
encoder_rgb_private_level4 = encoder1.stages[3]
feats = []
feats.append(encoder_shared_level1(x))
feats.append(encoder_shared_level2(feats[-1]))
feats.append(encoder_rgb_private_level3(feats[-1]))
feats.append(encoder_rgb_private_level4(feats[-1]))
feats.reverse()

up = FinalPatchExpand_X4(input_resolution=(96, 96), dim_scale=4, dim=128)
output = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, bias=False)
x = up(rearrange(feats[-1], 'b h w c->b (h w) c'))
x = rearrange(x, 'b (h w) c -> b c h w', h=384, w=384)
x = output(x)
x

