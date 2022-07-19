# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import copy
import math
import os
import os.path as osp
import random
import sys

sys.path.append("../")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import loss
import network
from data_list import ImageList, ImageList_idx
from knowledge_transfer import train_target

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOCKET")
    parser.add_argument("--gpu_id", type=str, nargs="?", default="0", help="device id to run")
    parser.add_argument(
        "--t", type=int, default=0, help="target"
    )  ## Choose which domain to set as target {0 to len(names)-1}
    # parser.add_argument('--s', type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=15, help="max iterations")
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--worker", type=int, default=1, help="number of workers")
    parser.add_argument("--dset", type=str, default="example_dataset", choices=["example_dataset"])
    parser.add_argument("--lr", type=float, default=1 * 1e-2, help="learning rate")
    parser.add_argument("--net", type=str, default="resnet50", help="vgg16, resnet50, res101")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")

    parser.add_argument("--gent", type=bool, default=True)
    parser.add_argument("--ent", type=bool, default=True)
    parser.add_argument("--threshold", type=int, default=0)
    parser.add_argument("--cls_par", type=float, default=0.3)
    parser.add_argument("--ent_par", type=float, default=1.0)
    parser.add_argument("--lr_decay1", type=float, default=0.1)
    parser.add_argument("--lr_decay2", type=float, default=1.0)

    parser.add_argument("--TIs", type=int, default=1, help="Task Irrelevant source")
    parser.add_argument("--TIt", type=int, default=0, help="Task Irrelevant target")

    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--distance", type=str, default="cosine", choices=["euclidean", "cosine"])
    parser.add_argument("--output", type=str, default="ckps/adapt_ours")
    parser.add_argument("--output_src", type=str, default="ckps/source")
    parser.add_argument("--da", type=str, default="uda", choices=["uda"])

    parser.add_argument(
        "--lambda_d", type=float, default=0.5, help="coefficient for feature distribution regularization"
    )
    parser.add_argument("--lambda_TI", type=float, default=0.5, help="TI embedding regularizer")
    parser.add_argument("--TI_disc", type=float, default=0.0, help="TI discriminator regularizer")

    args = parser.parse_args()

    # set names = multiple domains --- args.tgt decides the target domain index
    if args.dset == "example_dataset":
        names = ["red", "blue"]
        args.class_num = 2
        names_TI = ["TI_blue", "TI_red"]

    args.src = []
    for i in range(len(names)):
        if i == args.t:
            continue
        else:
            args.src.append(names[i])

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for i in range(len(names)):
        if i != args.t:
            continue
        folder = "../data/"
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"
        args.test_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"
        args.TIs_dset_path = folder + args.dset + "/" + names_TI[args.TIs] + "_list.txt"
        args.TIt_dset_path = folder + args.dset + "/" + names_TI[args.TIt] + "_list.txt"

        print(args.t_dset_path)

    args.output_dir_src = []
    for i in range(len(args.src)):

        args.output_dir_src.append(osp.join(args.output_src, args.da, args.dset, args.src[i].upper()))
    print(args.output_dir_src)

    args.output_dir = osp.join(args.output, args.da, args.dset, names[args.t])

    if not osp.exists(args.output_dir):
        os.system("mkdir -p " + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    extension_target = args.t_dset_path.rsplit("_")
    extension_source_TI = args.TIs_dset_path.rsplit("_")
    extension_target_TI = args.TIt_dset_path.rsplit("_")

    args.mode_target = "RGB"
    args.mode_TIs = "RGB"
    args.mode_TIt = "RGB"

    args.savename = "par_" + str(args.cls_par)

    train_target(args)
