# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2020 Jian Liang, Dapeng Hu, Jiashi Feng
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Code adapted from https://github.com/tim-learn/SHOT -- MIT License

import argparse
import copy
import math
import os
import os.path as osp
import random
import sys

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
from loss import CrossEntropyLabelSmooth


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == "uda":
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(" ")
            if int(reci[1]) in args.src_classes:
                line = reci[0] + " " + str(label_map_s[int(reci[1])]) + "\n"
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(" ")
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + " " + str(label_map_s[int(reci[1])]) + "\n"
                    new_tar.append(line)
                else:
                    line = reci[0] + " " + str(len(label_map_s)) + "\n"
                    new_tar.append(line)
        txt_test = new_tar.copy()

    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train(), mode=args.mode_train)
    dset_loaders["source_tr"] = DataLoader(
        dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False
    )
    dsets["source_te"] = ImageList(te_txt, transform=image_test(), mode=args.mode_train)
    dset_loaders["source_te"] = DataLoader(
        dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False
    )
    dsets["test"] = ImageList(txt_test, transform=image_test(), mode=args.mode_test)
    dset_loaders["test"] = DataLoader(
        dsets["test"], batch_size=train_bs * 2, shuffle=True, num_workers=args.worker, drop_last=False
    )

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    # print(all_output.size())
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = " ".join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def train_source(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == "res":
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == "vgg":
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(
        type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck
    ).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    # print(netC)

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{"params": v, "lr": learning_rate * 0.1}]
    for k, v in netB.named_parameters():
        param_group += [{"params": v, "lr": learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{"params": v, "lr": learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netB(netF(inputs_source)))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(
            outputs_source, labels_source
        )

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_te, _ = cal_acc(dset_loaders["source_te"], netF, netB, netC, False)
            log_str = "Task: {}, Iter:{}/{}; Accuracy = {:.2f}%".format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str + "\n")

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netB.train()
            netC.train()

    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF, netB, netC


def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == "res":
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == "vgg":
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(
        type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck
    ).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + "/source_F.pt"
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + "/source_B.pt"
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + "/source_C.pt"
    print(args.modelpath)
    netC.load_state_dict(torch.load(args.modelpath))
    print(netC)
    netF.eval()
    netB.eval()
    netC.eval()
    acc, _ = cal_acc(dset_loaders["test"], netF, netB, netC, False)
    log_str = "\nTraining: {}, Task: {}, Accuracy = {:.2f}%".format(args.trte, args.name, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOCKET")
    parser.add_argument("--gpu_id", type=str, nargs="?", default="0", help="device id to run")
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=20, help="max iterations")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument("--dset", type=str, default="SUN", choices=["SUN", "RGB_NIR", "DIML"])
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

    if args.dset == "SUN":
        names = [
            "kv1com_depth",
            "kv1com_image",
            "kv2com_depth",
            "kv2com_image",
            "realsensecom_depth",
            "realsensecom_image",
            "xtioncom_depth",
            "xtioncom_image",
        ]
        args.class_num = 17
    if args.dset == "RGB_NIR":
        names = ["nirscene_nir", "nirscene_rgb"]
        args.class_num = 6
    if args.dset == "DIML":
        names = ["DIML_image", "DIML_depth"]
        args.class_num = 6
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = "./data/"
    args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_list.txt"
    args.test_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"

    extension_train = args.s_dset_path.rsplit("_")
    extension_test = args.test_dset_path.rsplit("_")

    if extension_train[-2] == "depth":
        args.mode_train = "L"
    else:
        args.mode_train = "RGB"

    if extension_test[-2] == "depth":
        args.mode_test = "L"
    else:
        args.mode_test = "RGB"

    if extension_train[-2] == "nir":
        args.mode_train = "L"
    else:
        args.mode_train = "RGB"

    if extension_test[-2] == "nir":
        args.mode_test = "L"
    else:
        args.mode_test = "RGB"

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

        folder = "data/"
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_list.txt"
        args.test_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"

        test_target(args)
