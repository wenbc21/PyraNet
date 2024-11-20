# Pytorch-PyraNet

## Introduction
This is a pytorch version reproduce code of 'Learning Feature Pyramids for Human Pose Estimation', ICCV 2017. Link of paper: https://arxiv.org/pdf/1708.01101.pdf.

## The network
The network designed based on stacked hourglass network. 

Pyranet replace the Residual module with Pyramid Residual module.

The authors design a Pyramid Residual module to use features and information of multi-scale

In this repo, PRM-A, PRM-B and PRM-C has been realized. Following the comment you can choose these three Pyramid Residual Module. 
See the definition of Pyramid Residual Module and the network architecture in [model.py](model.py).

## Requirements
The code has been tested with Ubuntu 20.04 and CUDA 11.8.
The training process was conducted on one NVIDIA GeForce RTX 3090.

## Datasets
Mpii human pose dataset and LSP human pose dataset.


## Demo
You may download the weights from the link to do some inference on our demo images
```
https://drive.google.com/file/d/17qRIv1Ryx0WJxKyzvF3h_xTlhrFJRG_Q/view?usp=sharing
```

## Evaluate
You may evaluate our predictions from MPII and LSP dataset by running
```
python tools/eval_PCK.py exp/lsp/preds_200.mat
```
on LSP dataset, and
```
python tools/eval_PCKh.py exp/mpii/preds_200.mat
```
on MPII dataset.

You can get a result like that :
```
Head,   Shoulder, Elbow,  Wrist,   Hip,     Knee,    Ankle,  Mean
95.02   93.07     85.00   78.79    83.94    79.47    75.60   84.56
```

## Visualize
You need to download the dataset first for visualization and training. Please follow [README](data/README.md) to do so.
After downloading the dataset, you may run 
```
python tools/visualize.py exp/lsp/preds_200.mat
```
for LSP visualization, and
```
python tools/visualize.py exp/mpii/preds_200.mat
```
for MPII visualization.


## Train
For example, if you train the PRM network with 300 epochs, batch 6 and 2 stacked hourglass module with one gpu:
```
CUDA_VISIBLE_DEVICES = 0 python main.py -nEpochs 300 -trainBatch 6 -nStack 2
```

## Acknowledgement
Thanks for the authors of 'Learning Feature Pyramids for Human Pose Estimation'.
