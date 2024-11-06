# PyraNet

## The network
The network designed based on stacked hourglass network. 

## Datasets
Mpii human pose dataset. Details of path setting: see ref.py. You can set the path in this file.

## Train
For example, if you train the PRM network with 300 epochs, batch 6 and 2 stacked hourglass module with one gpu:
```
CUDA_VISIBLE_DEVICES = 0 python main.py -nEpochs 300 -trainBatch 6 -nStack 2
```
