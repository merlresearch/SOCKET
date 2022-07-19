# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from distutils.dir_util import copy_tree

dir1 = "./SUNRGBD/kv1/kv1_depth"
dir2 = "./SUNRGBD/kv2/kv2_depth"
dir3 = "./SUNRGBD/realsense/realsense_dpth"
dir4 = "./SUNRGBD/xtion/xtion_dpth"
dir5 = "./SUNRGBD/kv1/kv1_image"
dir6 = "./SUNRGBD/kv2/kv2_image"
dir7 = "./SUNRGBD/realsense/realsense_img"
dir8 = "./SUNRGBD/xtion/xtion_img"
list1 = os.listdir(dir1)
list2 = os.listdir(dir2)
list3 = os.listdir(dir3)
list4 = os.listdir(dir4)
list_int = list(set(list1) & set(list2) & set(list3) & set(list4))
list_uni = list(set(list1) | set(list2) | set(list3) | set(list4))
list_diff = list(set(list_uni) - set(list_int))

print(list_diff)

list_domains = ["TI"]
root_dir = "./SUN"
for i in range(len(list_domains)):
    path_temp_depth = os.path.join(root_dir, list_domains[i] + "_depth")
    path_temp_image = os.path.join(root_dir, list_domains[i] + "_image")
    if not os.path.exists(path_temp_depth):
        os.mkdir(path_temp_depth)
    if not os.path.exists(path_temp_image):
        os.mkdir(path_temp_image)
    for j in range(len(list_diff)):
        dpth_com_classes = os.path.join(path_temp_depth, list_diff[j])
        img_com_classes = os.path.join(path_temp_image, list_diff[j])
        if not os.path.exists(dpth_com_classes):
            os.mkdir(dpth_com_classes)
        if not os.path.exists(img_com_classes):
            os.mkdir(img_com_classes)
            # print(img_com_classes)
for k in range(len(list_diff)):
    path_source1 = os.path.join(os.path.join(root_dir, "SUN_depth"), list_diff[k])
    path_target1 = os.path.join(os.path.join(root_dir, "TI_depth"), list_diff[k])
    copy_tree(path_source1, path_target1)
    path_source2 = os.path.join(os.path.join(root_dir, "SUN_image"), list_diff[k])
    path_target2 = os.path.join(os.path.join(root_dir, "TI_image"), list_diff[k])
    copy_tree(path_source2, path_target2)
