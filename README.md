# PyraNet

## Introduction
This is a pytorch version reproduce code of 'Learning Feature Pyramids for Human Pose Estimation', ICCV 2017. Link of paper: https://arxiv.org/pdf/1708.01101.pdf.

## The network
The network designed based on stacked hourglass network. 

Pyranet replace the Residual module with Pyramid Residual module. The network:
![avatar](https://github.com/IcewineChen/pytorch-PyraNet/blob/master/imgs/network.png)

The authors design a Pyramid Residual module to use features and information of multi-scale:
![avatar](https://github.com/IcewineChen/pytorch-PyraNet/blob/master/imgs/prm.png)

In this repo, PRM-A, PRM-B and PRM-C has been realized. Following the comment you can choose these three Pyramid Residual Module. 
See the definition of Pyramid Residual Module in models/prm.py and the network architecture in models/network.py. 

## Requirements
The code has been tested with Ubuntu 20.04 and CUDA 11.8.
The training process was conducted on one NVIDIA GeForce RTX 3090.

## Datasets
Mpii human pose dataset. 

## Train
For example, if you train the PRM network with 300 epochs, batch 6 and 2 stacked hourglass module with one gpu:
```
CUDA_VISIBLE_DEVICES = 0 python main.py -nEpochs 300 -trainBatch 6 -nStack 2
```

## Evaluation

Result: using tools/eval_pckh.py for evaluation.You can get a result like that :
```
Head,   Shoulder, Elbow,  Wrist,   Hip,     Knee,    Ankle,  Mean
95.02   93.07     85.00   78.79    83.94    79.47    75.60   84.56
```

## Acknowledgement
Thanks for the authors of 'Learning Feature Pyramids for Human Pose Estimation'.
