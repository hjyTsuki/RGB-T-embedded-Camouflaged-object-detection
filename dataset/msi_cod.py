# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy
import torch
import torchvision

from dataset.base_dataset import _BaseSODDataset
from dataset.transforms.resize import ms_resize, ss_resize, ts_resize
from dataset.transforms.rotate import UniRotate
from utils.builder import DATASETS
from utils.io.genaral import get_datasets_info_with_keys
from utils.io.image import read_color_array, read_gray_array, read_thermal_array


@DATASETS.register(name="msi_cod_te")
class MSICOD_TestDataset(_BaseSODDataset):
    def __init__(self, root: Tuple[str, dict], shape: Dict[str, int], interp_cfg: Dict = None):
        super().__init__(base_shape=shape, interp_cfg=interp_cfg)
        self.datasets = get_datasets_info_with_keys(dataset_infos=[root], extra_keys=["mask", "thermal"])
        self.total_image_paths = self.datasets["image"]
        self.total_mask_paths = self.datasets["mask"]
        self.total_thermal_paths = self.datasets["thermal"]

        self.image_norm = A.Normalize(mean=(0.485, 0.456, 0.406, 0.7583, 0.3712, 0.3361), std=(0.229, 0.224, 0.225, 0.1573, 0.1815, 0.1625))

    def __getitem__(self, index):
        image_path = self.total_image_paths[index]
        mask_path = self.total_mask_paths[index]
        thermal_path = self.total_thermal_paths[index]

        image = read_color_array(image_path)
        thermal = read_thermal_array(thermal_path)

        image_co_thermal = numpy.concatenate([image, thermal], axis=2)

        image = self.image_norm(image=image_co_thermal)["image"]

        thermal = image[:, :, 3:]
        image = image[:, :, 0:3]

        base_h = self.base_shape["h"]
        base_w = self.base_shape["w"]
        images = ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
        image_0_5 = torch.from_numpy(images[0]).permute(2, 0, 1)
        image_1_0 = torch.from_numpy(images[1]).permute(2, 0, 1)
        image_1_5 = torch.from_numpy(images[2]).permute(2, 0, 1)
        thermal = ss_resize(thermal, 1, base_h=base_h, base_w=base_w)
        thermal = torch.from_numpy(thermal).permute(2, 0, 1)

        return dict(
            data={
                "image1.5": image_1_5,
                "image1.0": image_1_0,
                "image0.5": image_0_5,
                "thermal": thermal
            },
            info=dict(
                mask_path=mask_path,
            ),
        )

    def __len__(self):
        return len(self.total_image_paths)


@DATASETS.register(name="msi_cod_tr")
class MSICOD_TrainDataset(_BaseSODDataset):
    def __init__(
        self, root: List[Tuple[str, dict]], shape: Dict[str, int], extra_scales: List = None, interp_cfg: Dict = None
    ):
        super().__init__(base_shape=shape, extra_scales=extra_scales, interp_cfg=interp_cfg)
        self.datasets = get_datasets_info_with_keys(dataset_infos=root, extra_keys=["mask", "thermal"])
        self.total_image_paths = self.datasets["image"]
        self.total_mask_paths = self.datasets["mask"]
        self.total_thermal_paths = self.datasets["thermal"]
        self.joint_trans = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                UniRotate(limit=10, interpolation=cv2.INTER_LINEAR, p=0.5),
                # thermal 的均值和std有待考证
                A.Normalize(mean=(0.485, 0.456, 0.406, 0.7538, 0.3609, 0.3417), std=(0.229, 0.224, 0.225, 0.1809, 0.2107, 0.1716)),
            ],
        )
        self.reszie = A.Resize

    def __getitem__(self, index):
        image_path = self.total_image_paths[index]
        mask_path = self.total_mask_paths[index]
        thermal_path = self.total_thermal_paths[index]

        image = read_color_array(image_path)
        mask = read_gray_array(mask_path, to_normalize=True, thr=0.5)
        # 读取thermal图像
        thermal = read_thermal_array(thermal_path)

        image_co_thermal = numpy.concatenate([image, thermal], axis=2)

        transformed = self.joint_trans(image=image_co_thermal, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        thermal = image[:, :, 3:]
        image = image[:, :, 0:3]

        base_h = self.base_shape["h"]
        base_w = self.base_shape["w"]
        images = ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
        image_0_5 = torch.from_numpy(images[0]).permute(2, 0, 1)
        image_1_0 = torch.from_numpy(images[1]).permute(2, 0, 1)
        image_1_5 = torch.from_numpy(images[2]).permute(2, 0, 1)

        thermals = ms_resize(thermal, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
        thermal_0_5 = torch.from_numpy(thermals[0]).permute(2, 0, 1)
        thermal_1_0 = torch.from_numpy(thermals[1]).permute(2, 0, 1)
        thermal_1_5 = torch.from_numpy(thermals[2]).permute(2, 0, 1)

        mask = ss_resize(mask, scale=1.0, base_h=base_h, base_w=base_w)
        mask_1_0 = torch.from_numpy(mask).unsqueeze(0)

        # edge_gt = cv2.Canny(mask_1_0, threshold1=100, threshold2=200)

        return dict(
            data={
                "image1.5": image_1_5,
                "image1.0": image_1_0,
                "image0.5": image_0_5,
                "thermal0.5": thermal_0_5,
                "thermal1.0": thermal_1_0,
                "thermal1.5": thermal_1_5,
                "mask": mask_1_0,
                # "thermal": thermal,
                # "edge": edge_gt
            }
        )

    def __len__(self):
        return len(self.total_image_paths)

