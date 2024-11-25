# PyraNet

## Introduction
This is a reimplement of paper 'Learning Feature Pyramids for Human Pose Estimation', ICCV 2017. Thanks for the authors of paper.

Paper link: https://arxiv.org/pdf/1708.01101.pdf

## Methods
PyraNet replace the Residual module in Stacked Hourglass Network with Pyramid Residual Module, in order to capture multi-scale features.

PRM-A, PRM-B, PRM-C and PRM-D can be selected by commenting and un-commenting corresponding lines in [model.py](model.py)

## Requirements
The code has been tested under Ubuntu 20.04 and CUDA 11.8.
The training process was conducted on one NVIDIA GeForce RTX 3090.

If you need to inference a demo with human detection (highly recommend), you need to install mmcv and mmdet.
You can also inference without detection by running [inference_nodet.py](inference_nodet.py)

## Demo
You may download the weights from the link to do some inference on our demo images.
Download PyraNet weight and place them under [exp/lsp](exp/lsp) and [exp/mpii](exp/mpii).
```
https://drive.google.com/file/d/1pgvWvol9yJNcOh_e4eDcqyAoQuurgP5y/view?usp=sharing
```
Download Faster R-CNN weight and place it in [utils/mmdet](utils/mmdet)
```
https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
```
Assume you have a series of images to be inferred, you can put them in demo/kunkun/images, and run
```
CUDA_VISIBLE_DEVICES = 0 python inference.py --loadModel exp/mpii/model_200.pth --dataDir demo/kunkun
```

## Train
You should follow [README](data/README.md) to download dataset and organize them. Then you can train PyraNet by running
```
CUDA_VISIBLE_DEVICES = 0 python main.py -epochs 200
```

## Evaluation
If you have the mat file produced during the training process, you can evaluate PCK and PCKh by running [eval_PCK.py](tools/eval_PCK.py) and [eval_PCKh.py](tools/eval_PCKh.py), for example
```
python tools/eval_PCKh.py exp/mpii/preds_200.mat
```
or
```
python tools/eval_PCK.py exp/lsp/preds_200.mat
```

The results are like:
```
Head,   Shoulder, Elbow,  Wrist,   Hip,     Knee,    Ankle,  Mean
96.73   95.87     90.06   85.15    92.45    86.46    82.71   90.04
```

## Visualize
If you have the mat file produced during the training process, you can visualize the result files by running [visualize.py](tools/visualize.py)
```
python tools/visualize.py exp/mpii/preds_200.mat
```
for MPII visualization, and
```
python tools/visualize.py exp/lsp/preds_200.mat
```
for LSP visualization. 
Remember to comment and uncomment corresponding lines in the file.
