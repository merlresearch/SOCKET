# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2020 Jian Liang, Dapeng Hu, Jiashi Feng
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Code adapted from https://github.com/tim-learn/SHOT -- MIT License

import os
import os.path
import random

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def l_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:

            A_img = np.array(img)
            if A_img.dtype == "uint8":
                A_img = np.array(img) / (2**8 - 1)
            else:
                A_img = np.array(img) / (2**16 - 1)

            A_pil_to_tensor = torchvision.transforms.ToTensor()(A_img).unsqueeze_(0)
            A_np_to_tensor = torch.from_numpy(A_img)
            A_np_to_tensor_ex = A_np_to_tensor[None, None, :, :]
            A_pil_to_tensor = torch.cat((A_np_to_tensor_ex, A_np_to_tensor_ex, A_np_to_tensor_ex), 1)
            A_pil_to_tensor = A_pil_to_tensor.to(torch.float32)
            A_img = torchvision.transforms.ToPILImage()(A_pil_to_tensor.squeeze_(0))
    return A_img


class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode="RGB"):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in subfolders of: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == "RGB":
            self.loader = rgb_loader
        elif mode == "L":
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode="RGB"):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in subfolders of: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == "RGB":
            self.loader = rgb_loader
        elif mode == "L":
            self.loader = l_loader
        elif mode == "L1":
            self.loader = l1_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)
