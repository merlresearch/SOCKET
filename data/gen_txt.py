# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

dataset = "SUN"

# domains = ['kv1com_image','kv1com_depth', 'kv2com_image','kv2com_depth','realsensecom_image','realsensecom_depth', 'xtioncom_image', 'xtioncom_depth']
domains = ["TI_depth", "TI_image"]

for domain in domains:
    log = open(dataset + "/" + domain + "_list.txt", "w")
    directory = os.path.join(dataset, domain)
    classes = [x[0] for x in os.walk(directory)]
    classes = classes[1:]
    classes.sort()
    # print(classes[0])

    for idx, f in enumerate(classes):

        files = os.listdir(f)

        for file in files:
            s = os.path.abspath(os.path.join(f, file)) + " " + str(idx) + "\n"
            log.write(s)
    log.close()
