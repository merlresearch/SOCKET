<!--
Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# SOCKET: SOurce-free Cross-modal KnowledgE Transfer

Code for the ECCV 2022 paper on Cross-Modal Knowledge Transfer Without Task-Relevant Source Data is released here.

Code is based on

SHOT: https://github.com/tim-learn/SHOT -- MIT License

DECISION: https://github.com/driptaRC/DECISION -- MIT License

Main references:

SHOT: Liang et al., "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation", ICML 2020

DECISION: Ahmed et al., "Unsupervised multi-source domain adaptation without access to source data", CVPR 2021

## Features

SOCKET allows transfering knowledge from neural networks trained on a source sensor modality (such as RGB) for one or more domains where large amount of annotated data may be available to an unannotated target dataset from a different sensor modality. It makes use of task-irrelevant paired source-target images in order to promote feature alignment between the two modalities as well as distribution matching between the source batch norm features (mean and variance) and the target features.

Description of files:

1. image_source.py : train source models for each domain separately.


2. knowledge_transfer.py: The main code that does the knowledge transfer, given trained source models.


3. network.py: Network definition.


4. loss.py: Smoothed cross-entropy loss definition used for training source models.


5. data_list.py: data loaders for training source as well as knowledge transfer.

## Installation

The required packages to run SOCKET are provided in environment.yml

## Usage

### Setting up data:

An important part of the code is setting up the data in the required format. In the case of SUN RGB-D, there are two modalities -- RGB and depth --  and four "domains" which refer to the type of sensor used to acquire the data -- Kinect V1 (KV1), Kinect V2 (KV2), RealSense, Xtion. We used 17 classes common to all these domains as the training data for the source models in one modality and the corresponding images in the other modality as the target data. The paired images belonging to the remaining classes from all the domains are grouped together to form the Task-Irrelevant (TI) data.

Following are all the folders needed to reproduce the results for SUN RGB-D shown in the SOCKET paper. Each of these folders should have subfolders for each of the 17 common classes which contain the images:

data/SUN/kv1com_image -- Kinect V1 RGB
data/SUN/kv1com_depth -- Kinect V1 depth

data/SUN/kv2com_image -- Kinect V2 RGB
data/SUN/kv2com_depth -- Kinect V2 depth

data/SUN/realsensecom_image -- Realsense RGB
data/SUN/realsensecom_depth -- Realsense depth

data/SUN/xtioncom_image -- Xtion RGB
data/SUN/xtioncom_depth -- Xtion depth

data/SUN/TI_image -- Task Irrelevant RGB
data/SUN/TI_depth -- Task Irrelevant Depth


Once these folders are created, please run gen_txt.py and choose the right domains in this code to create the txt files that contain the full image paths and the class labels. The images in the right folders and these txt files are all that are needed to run `image_source.py followed by knowledge_transfer.py


### There are two steps to test SOCKET.

(1) Training the source models given training data from some modality. This is achieved by choosing the source modality (and domain) in `image_source.py` and other parameters and running `image_source.py`. This trains the source model and puts the checkpoints in the right folder.

(2) Using the trained source model along with Task-Irrelevant data (TI data) to perform knowledge transfer to a target modality by running `knowledge_transfer.py`. This is achieved by choosing the source modality (and domain) which in turn chooses the right trained source model, and choosing the target modality (and domain)


### Testing the code

You can test the installation easily by first running `tests/test_image_source.py` and then `tests/test_knowledge_transfer.py`. They use the example dataset provided under `data/`.

## Citation

If you use the software, please cite the following:

```
@inproceedings{ahmed2021unsupervised,
  title={Cross-Modal Knowledge Transfer Without Task-Relevant Source Data},
  author={Ahmed, Sk Miraj and Lohit, Suhas and Peng, Kuan-Chuan and Jones, Michael and Roy-Chowdhury, Amit K},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2022}
}
```

## Contact

Sk Miraj Ahmed: sahme047@ucr.edu,
Suhas Lohit: slohit@merl.com

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## Copyright and License

All files, except `data_list.py`, `image_source.py`, `knowledge_transfer.py`, `loss.py` and `network.py`:

```
Copyright (c) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
```

`data_list.py`, `image_source.py`, `loss.py`, `knowledge_transfer.py`, and `network.py` were adapted from https://github.com/tim-learn/SHOT
`data_list.py`, `image_source.py`, `knowledge_transfer.py`, `loss.py` and `network.py` were adapted from https://github.com/tim-learn/SHOT
(license included in [LICENSES/MIT.txt](LICENSES/MIT.txt)):
`knowledge_transfer.py` and `network.py` were adapted from https://github.com/driptaRC/DECISION
```
Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
Copyright (C) 2021 Dripta S. Raychaudhari
Copyright (C) 2020 Jian Liang, Dapeng Hu, Jiashi Feng
```


Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.
