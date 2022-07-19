# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 Dripta S. Raychaudhari
# Copyright (C) 2020 Jian Liang, Dapeng Hu, Jiashi Feng
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/driptaRC/DECISION -- MIT License and https://github.com/tim-learn/SHOT -- MIT License


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
        normalize = transforms.Normalize(mean=[0.6983, 0.3918, 0.4474], std=[0.1648, 0.1359, 0.1644])
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
        normalize = transforms.Normalize(mean=[0.6983, 0.3918, 0.4474], std=[0.1648, 0.1359, 0.1644])
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
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()
    txt_TI_source = open(args.TIs_dset_path).readlines()
    txt_TI_target = open(args.TIt_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train(), mode=args.mode_target)
    dset_loaders["target"] = DataLoader(
        dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False
    )
    dsets["target_"] = ImageList_idx(txt_tar, transform=image_train(), mode=args.mode_target)
    dset_loaders["target_"] = DataLoader(
        dsets["target_"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker, drop_last=False
    )
    dsets["test"] = ImageList_idx(txt_test, transform=image_test(), mode=args.mode_target)
    dset_loaders["test"] = DataLoader(
        dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker, drop_last=False
    )
    dsets["source_TI"] = ImageList_idx(txt_TI_source, transform=image_train(), mode=args.mode_TIs)
    dset_loaders["source_TI"] = DataLoader(
        dsets["source_TI"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False
    )
    dsets["target_TI"] = ImageList_idx(txt_TI_target, transform=image_train(), mode=args.mode_TIt)

    dset_loaders["target_TI"] = DataLoader(
        dsets["target_TI"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False
    )

    return dset_loaders


def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == "res":
        netF_list = [network.ResBase(res_name=args.net).cuda() for i in range(len(args.src))]
    elif args.net[0:3] == "vgg":
        netF_list = [network.VGGBase(vgg_name=args.net).cuda() for i in range(len(args.src))]

    w = torch.ones((len(args.src),))

    # Load models for training (encoder,classifier,discriminator)
    netB_list = [
        network.feat_bootleneck(
            type=args.classifier, feature_dim=netF_list[i].in_features, bottleneck_dim=args.bottleneck
        ).cuda()
        for i in range(len(args.src))
    ]
    netC_list = [
        network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
        for i in range(len(args.src))
    ]
    netG_list = [network.scalar(w[i]).cuda() for i in range(len(args.src))]
    critic = network.Discriminator(256, 500, 2)
    critic.train()

    param_group = []
    for i in range(len(args.src)):
        modelpath = args.output_dir_src[i] + "/source_F.pt"
        # print(modelpath)
        netF_list[i].load_state_dict(torch.load(modelpath))
        netF_list[i].eval()
        for k, v in netF_list[i].named_parameters():
            param_group += [{"params": v, "lr": args.lr * args.lr_decay1}]

        modelpath = args.output_dir_src[i] + "/source_B.pt"
        # print(modelpath)
        netB_list[i].load_state_dict(torch.load(modelpath))
        netB_list[i].eval()
        for k, v in netB_list[i].named_parameters():
            param_group += [{"params": v, "lr": args.lr * args.lr_decay2}]

        modelpath = args.output_dir_src[i] + "/source_C.pt"
        # print(modelpath)
        netC_list[i].load_state_dict(torch.load(modelpath))
        netC_list[i].eval()
        for k, v in netC_list[i].named_parameters():
            v.requires_grad = False

        for k, v in netG_list[i].named_parameters():
            param_group += [{"params": v, "lr": args.lr}]

    # Optimizer for source models
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    # Optimizer and criteria for discriminator
    criterion_disc = nn.CrossEntropyLoss()

    optimizer_critic = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

    src_mean = []
    src_var = []
    src_mean_all = []
    src_var_all = []
    tgt_loss_r_feature_layers = []
    tgt_loss_r_feature_layers_all = []
    for j in range(len(args.src)):
        source_copy = copy.deepcopy(netF_list[j])
        for module in source_copy.modules():
            if isinstance(module, nn.BatchNorm2d):

                src_mean.append(module.running_mean.data)
                src_var.append(module.running_var.data)
        # Extract batchnorm statistics for sources
        src_mean_all.append(src_mean)
        src_var_all.append(src_var)

        for module in netF_list[j].modules():
            if isinstance(module, nn.BatchNorm2d):
                tgt_loss_r_feature_layers.append(network.DistributionMatching(module))
        tgt_loss_r_feature_layers_all.append(tgt_loss_r_feature_layers)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    c = 0
    with torch.no_grad():
        # Extract TI features before training
        final_feature_source = []
        final_output_source = []
        for i in range(len(args.src)):
            iter_TI_source = iter(dset_loaders["source_TI"])
            start_test_TI = True
            all_fea_source_i = torch.zeros(len(dset_loaders["source_TI"]), args.bottleneck).cuda()
            all_out_source_i = torch.zeros(len(dset_loaders["source_TI"]), args.class_num).cuda()
            for j in range(len(dset_loaders["source_TI"])):

                inputs_TI, _, _ = iter_TI_source.next()
                inputs_TI = inputs_TI.cuda()

                features_source_all = netB_list[i](netF_list[i](inputs_TI))
                outputs_source_all = netC_list[i](features_source_all)
                if start_test_TI:
                    all_fea_source_i = features_source_all.float().cpu()
                    all_out_source_i = outputs_source_all.float().cpu()
                    start_test_TI = False
                else:
                    all_fea_source_i = torch.cat((all_fea_source_i, features_source_all.float().cpu()), 0)
                    all_out_source_i = torch.cat((all_out_source_i, outputs_source_all.float().cpu()), 0)
            final_feature_source.append(all_fea_source_i)
            final_output_source.append(all_out_source_i)

    for k in range(len(args.src)):
        final_feature_source[k] = final_feature_source[k].cuda()
    # Starting knowledge transfer phase
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        try:
            inputs_TI_target, _, index_TI_target = iter_source.next()

        except:
            iter_TI_target = iter(dset_loaders["target_TI"])
            inputs_TI_target, _, index_TI_target = iter_TI_target.next()

        source_copy.eval()

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            initc = []
            all_feas = []
            for i in range(len(args.src)):
                netF_list[i].eval()
                netB_list[i].eval()
                temp1, temp2 = obtain_label(dset_loaders["target_"], netF_list[i], netB_list[i], netC_list[i], args)
                temp1 = torch.from_numpy(temp1).cuda()
                temp2 = torch.from_numpy(temp2).cuda()
                initc.append(temp1)
                all_feas.append(temp2)
                netF_list[i].train()
                netB_list[i].train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        outputs_all = torch.zeros(len(args.src), inputs_test.shape[0], args.class_num)
        weights_all = torch.ones(inputs_test.shape[0], len(args.src))
        outputs_all_w = torch.zeros(inputs_test.shape[0], args.class_num)

        init_ent = torch.zeros(1, len(args.src))

        for i in range(len(args.src)):
            features_test = netB_list[i](netF_list[i](inputs_test))

            outputs_test = netC_list[i](features_test)

            softmax_ = nn.Softmax(dim=1)(outputs_test)
            ent_loss = torch.mean(loss.Entropy(softmax_))
            init_ent[:, i] = ent_loss
            weights_test = netG_list[i](features_test)
            outputs_all[i] = outputs_test
            weights_all[:, i] = weights_test.squeeze()

        z = torch.sum(weights_all, dim=1)
        z = z + 1e-16

        weights_all = torch.transpose(torch.transpose(weights_all, 0, 1) / z, 0, 1)

        outputs_all = torch.transpose(outputs_all, 0, 1)

        z_ = torch.sum(weights_all, dim=0)

        z_2 = torch.sum(weights_all)
        z_ = z_ / z_2

        for i in range(inputs_test.shape[0]):
            outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])
        # pseudo-label loss
        if args.cls_par > 0:
            initc_ = torch.zeros(initc[0].size()).cuda()
            temp = all_feas[0]
            all_feas_ = torch.zeros(temp[tar_idx, :].size()).cuda()
            for i in range(len(args.src)):
                initc_ = initc_ + z_[i] * initc[i].float()
                src_fea = all_feas[i]
                all_feas_ = all_feas_ + z_[i] * src_fea[tar_idx, :]
            dd = torch.cdist(all_feas_.float(), initc_.float(), p=2)
            pred_label = dd.argmin(dim=1)
            pred_label = pred_label.int()
            pred = pred_label.long()
            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_all_w, pred.cpu())
        else:
            classifier_loss = torch.tensor(0.0)
        # IM loss
        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_all_w)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        # distribution matching loss

        multiplier = [10.0] + [
            1.0 for _ in range(len(tgt_loss_r_feature_layers) - 1)
        ]  # more weight for the first bn layer

        loss_d = torch.tensor(0.0)

        weighted_mean_src = []
        weighted_var_src = []
        for j in range(len(src_mean_all[0])):
            final_mean = torch.zeros_like(src_mean_all[0][j])
            final_var = torch.zeros_like(src_var_all[0][j])
            for k in range(len(src_mean_all)):
                weight_mean_src = netG_list[k](src_mean_all[k][j])
                weight_mean_src = torch.squeeze(weight_mean_src, 1)
                temp_mean = src_mean_all[k][j] * weight_mean_src
                temp_var = src_var_all[k][j] * weight_mean_src
                final_mean = final_mean + temp_mean
                final_var = final_var + temp_var
            weighted_mean_src.append(final_mean)
            weighted_var_src.append(final_var)

        weighted_mean_tgt = []
        weighted_var_tgt = []
        for j in range(len(tgt_loss_r_feature_layers_all[0])):
            final_mean = torch.zeros_like(src_mean_all[0][j])
            final_var = torch.zeros_like(src_var_all[0][j])
            for k in range(len(src_mean_all)):
                weight_mean_src = netG_list[k](tgt_loss_r_feature_layers_all[k][j].tgt_feat_mean)
                weight_mean_src = torch.squeeze(weight_mean_src, 1)
                temp_mean = tgt_loss_r_feature_layers_all[k][j].tgt_feat_mean * weight_mean_src
                temp_var = tgt_loss_r_feature_layers_all[k][j].tgt_feat_var * weight_mean_src
                final_mean = final_mean + temp_mean
                final_var = final_var + temp_var
            weighted_mean_tgt.append(final_mean)
            weighted_var_tgt.append(final_var)

        for idx in range(len(weighted_mean_src)):
            loss_d += (
                torch.norm(weighted_mean_tgt[idx] - weighted_mean_src[idx], 2)
                + torch.norm(weighted_var_tgt[idx] - weighted_var_src[idx], 2)
            ).cpu() * multiplier[idx]

        classifier_loss += args.lambda_d * loss_d
        # TI feature matching
        if args.lambda_TI > 0:
            weights_all_TI = torch.ones(inputs_TI_target.shape[0], len(args.src))

            inputs_TI_target = inputs_TI_target.cuda()

            inputs_all_w = torch.zeros(inputs_TI_target.shape[0], args.bottleneck)

            inputs_all = torch.zeros(len(args.src), inputs_TI_target.shape[0], args.bottleneck)
            for i in range(len(args.src)):

                feat_TI_source = final_feature_source[i][index_TI_target, :]
                feat_TI_source = feat_TI_source.cpu()
                feat_TI_target = netB_list[i](netF_list[i](inputs_TI_target))
                feat_TI_target = feat_TI_target.cpu()

                diff_source_TI = feat_TI_source - feat_TI_target
                weights_TI = netG_list[i](feat_TI_target)
                weights_all_TI[:, i] = weights_TI.squeeze()

                inputs_all[i] = diff_source_TI

            z1 = torch.sum(weights_all_TI, dim=1)
            z1 = z1 + 1e-16
            weights_all_TI = torch.transpose(torch.transpose(weights_all_TI, 0, 1) / z1, 0, 1)

            inputs_all = torch.transpose(inputs_all, 0, 1)

            for j in range(inputs_TI_target.shape[0]):
                inputs_all_w[j] = torch.matmul(torch.transpose(inputs_all[j], 0, 1), weights_all_TI[j])

            # TI_loss = torch.linalg.norm(inputs_all_w)
            TI_loss = torch.norm(inputs_all_w)

            classifier_loss += args.lambda_TI * TI_loss
        # Discriminator loss
        if args.TI_disc > 0:
            weights_all_TI = torch.ones(inputs_TI_target.shape[0], len(args.src))

            inputs_TI_target = inputs_TI_target.cuda()

            inputs_all_tw = torch.zeros(inputs_TI_target.shape[0], args.bottleneck)

            inputs_all_sw = torch.zeros(inputs_TI_target.shape[0], args.bottleneck)

            inputs_all_t = torch.zeros(len(args.src), inputs_TI_target.shape[0], args.bottleneck)

            inputs_all_s = torch.zeros(len(args.src), inputs_TI_target.shape[0], args.bottleneck)

            inputs_all = torch.zeros(len(args.src), inputs_TI_target.shape[0], args.bottleneck)
            for i in range(len(args.src)):

                feat_TI_source = final_feature_source[i][index_TI_target, :]
                feat_TI_source = feat_TI_source.cpu()
                feat_TI_target = netB_list[i](netF_list[i](inputs_TI_target))
                feat_TI_target = feat_TI_target.cpu()

                weights_TI = netG_list[i](feat_TI_target)
                weights_all_TI[:, i] = weights_TI.squeeze()
                inputs_all_t[i] = feat_TI_target
                inputs_all_s[i] = feat_TI_source

            z1 = torch.sum(weights_all_TI, dim=1)
            z1 = z1 + 1e-16
            weights_all_TI = torch.transpose(torch.transpose(weights_all_TI, 0, 1) / z1, 0, 1)
            inputs_all_t = torch.transpose(inputs_all_t, 0, 1)
            inputs_all_s = torch.transpose(inputs_all_s, 0, 1)

            for j in range(inputs_TI_target.shape[0]):
                inputs_all_tw[j] = torch.matmul(torch.transpose(inputs_all_t[j], 0, 1), weights_all_TI[j])

            for j in range(inputs_TI_target.shape[0]):
                inputs_all_sw[j] = torch.matmul(torch.transpose(inputs_all_s[j], 0, 1), weights_all_TI[j])

            feat_concat = torch.cat((inputs_all_sw, inputs_all_tw), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = torch.ones(inputs_all_sw.size(0)).long()
            label_tgt = torch.zeros(inputs_all_tw.size(0)).long()
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion_disc(pred_concat, label_concat)

            pred_tgt = critic(inputs_all_tw)

            # prepare fake labels
            label_tgt1 = torch.ones(inputs_all_tw.size(0)).long()

            # compute loss for target encoder
            loss_tgt = criterion_disc(pred_tgt, label_tgt1)

            classifier_loss += args.TI_disc * (10 * loss_tgt + loss_critic)

        nk = 1
        classifier_loss.backward()
        if (iter_num + 1) % nk == 0:
            classifier_loss = classifier_loss / nk
            optimizer.step()
            optimizer.zero_grad()
            # optimizer_critic.step()
            # optimizer_critic.zero_grad()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            for i in range(len(args.src)):
                netF_list[i].eval()
                netB_list[i].eval()
            acc, _ = cal_acc_multi(dset_loaders["test"], netF_list, netB_list, netC_list, netG_list, args)
            log_str = "Iter:{}/{}; Accuracy = {:.2f}%".format(iter_num, max_iter, acc)
            print(log_str + "\n")
            for i in range(len(args.src)):
                torch.save(
                    netF_list[i].state_dict(),
                    osp.join(args.output_dir, "target_F_" + str(i) + "_" + args.savename + ".pt"),
                )
                torch.save(
                    netB_list[i].state_dict(),
                    osp.join(args.output_dir, "target_B_" + str(i) + "_" + args.savename + ".pt"),
                )
                torch.save(
                    netC_list[i].state_dict(),
                    osp.join(args.output_dir, "target_C_" + str(i) + "_" + args.savename + ".pt"),
                )
                torch.save(
                    netG_list[i].state_dict(),
                    osp.join(args.output_dir, "target_G_" + str(i) + "_" + args.savename + ".pt"),
                )


def obtain_label(loader, netF, netB, netC, args):

    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs.float()))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

    return initc, all_fea


def cal_acc_multi(loader, netF_list, netB_list, netC_list, netG_list, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs_all = torch.zeros(len(args.src), inputs.shape[0], args.class_num)
            weights_all = torch.ones(inputs.shape[0], len(args.src))
            outputs_all_w = torch.zeros(inputs.shape[0], args.class_num)

            for i in range(len(args.src)):
                features = netB_list[i](netF_list[i](inputs))
                outputs = netC_list[i](features)
                weights = netG_list[i](features)
                outputs_all[i] = outputs
                weights_all[:, i] = weights.squeeze()

            z = torch.sum(weights_all, dim=1)
            z = z + 1e-16

            weights_all = torch.transpose(torch.transpose(weights_all, 0, 1) / z, 0, 1)

            outputs_all = torch.transpose(outputs_all, 0, 1)

            for i in range(inputs.shape[0]):
                outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])

            if start_test:
                all_output = outputs_all_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy * 100, mean_ent


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


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
    parser.add_argument("--dset", type=str, default="SUN", choices=["SUN", "RGB_NIR", "DIML"])
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
    if args.dset == "SUN":
        # names = ['xtioncom_depth', 'kv2com_image']
        # names = ['kv1com_depth', 'kv1com_image','kv2com_depth', 'kv2com_image', 'realsensecom_depth', 'realsensecom_image']
        # names = ['xtioncom_depth','kv1com_image', 'kv2com_image', 'realsensecom_image' ]
        names = ["kv1com_depth", "kv1com_image"]
        args.class_num = 17
        # names_TI = ['TI_depth', 'TI_image']
        names_TI = ["TI_depth", "TI_image"]
        # names_TI = ['TI_1class_depth', 'TI_1class_image']

        # names_TI = ['TI_large_depth', 'TI_large_image']
        # names_TI = ['TI_diml1_depth', 'TI_diml_image']
    if args.dset == "RGB_NIR":
        names = ["nirscene_nir", "nirscene_rgb"]
        names_TI = ["TI_nir", "TI_rgb"]
        args.class_num = 6
    if args.dset == "DIML":
        names = ["DIML_depth", "DIML_image"]
        args.class_num = 6
        # names_TI = ['TI_diml_depth', 'TI_diml_image']
        names_TI = ["TI_depth", "TI_image"]
        # names_TI = ['TI_1class_depth', 'TI_1class_image']

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
        folder = "./data/"
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

    if extension_target[-2] == "depth" or extension_target[-2] == "nir":
        args.mode_target = "L"
    else:
        args.mode_target = "RGB"

    if extension_source_TI[-2] == "depth" or extension_source_TI[-2] == "nir":
        args.mode_TIs = "L"
    else:
        args.mode_TIs = "RGB"

    if extension_target_TI[-2] == "depth" or extension_target_TI[-2] == "nir":
        args.mode_TIt = "L"
    else:
        args.mode_TIt = "RGB"

    args.savename = "par_" + str(args.cls_par)

    train_target(args)
