from numpy.random import randn
import utils.human_prior as hp
import torch
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
             
def Rnd(x):
    return max(-2 * x, min(2 * x, randn() * x))
  
def Flip(img):
    return img[:, :, ::-1].copy()  
  
def ShuffleLR(x):
    for e in hp.shuffleRef:
        x[e[0]], x[e[1]] = x[e[1]].copy(), x[e[0]].copy()
    return x
