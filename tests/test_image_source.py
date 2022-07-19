# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import os
import sys

sys.path.append("../")
import copy
import math
import os.path as osp
import pdb
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import loss
import network
from data_list import ImageList
from image_source import print_args, test_target, train_source
from loss import CrossEntropyLabelSmooth

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOCKET")
    parser.add_argument("--gpu_id", type=str, nargs="?", default="0", help="device id to run")
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=20, help="max iterations")
    parser.add_argument("--batch_size", type=int, default=2, help="batch_size")
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument("--dset", type=str, default="example_dataset", choices=["example_dataset"])
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--net", type=str, default="resnet50", help="vgg16, resnet50, resnet101")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--smooth", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="ckps/source")
    parser.add_argument("--da", type=str, default="uda", choices=["uda"])
    parser.add_argument("--trte", type=str, default="val", choices=["full", "val"])
    args = parser.parse_args()

    if args.dset == "example_dataset":
        names = ["blue", "red"]
        args.class_num = 2

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    args.mode_train = "RGB"
    args.mode_test = "RGB"

    folder = "../data/"
    args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_list.txt"
    args.test_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"

    args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s].upper())
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system("mkdir -p " + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, "log.txt"), "w")
    args.out_file.write(print_args(args) + "\n")
    args.out_file.flush()
    train_source(args)

    args.out_file = open(osp.join(args.output_dir_src, "log_test.txt"), "w")
    for i in range(len(names)):
        args.t = i
        args.name = names[args.s].upper() + names[args.t].upper()

        folder = "../data/"
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_list.txt"
        args.test_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"

        test_target(args)
