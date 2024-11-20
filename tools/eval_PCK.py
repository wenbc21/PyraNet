#from https://github.com/bearpaw/pytorch-pose/blob/master/evaluation/eval_PCK.py
import sys
from scipy.io import loadmat
from numpy import transpose
import skimage.io as sio
import numpy as np
import json
import os
from h5py import File
# local partNames = {'RAnk','RKne','RHip','LHip','LKne','LAnk',
#                     'Pelv','Thrx','Neck','Head',
#                     'RWri','RElb','RSho','LSho','LElb','LWri'}

with open(f'data/lsp/LEEDS_annotations.json', "r") as joints_file:
    data = json.load(joints_file)

imgname = []
part = []
center = []
scale = []
size = []
visible = []

for d in data :
    if d['isValidation'] == 1.0:
        imgname.append(d['img_paths'])
        part.append(np.array(d['joint_self'])[:, :2])
        visible.append(np.array(d['joint_self'])[:, 2])
        center.append(d['objpos'])
        scale.append(d['scale_provided'])
        size.append(min(d['img_width'], d['img_height']))

imgname = np.asarray(imgname)
part = transpose(np.asarray(part), [1,2,0])
visible = transpose(np.asarray(visible), [1, 0])
center = np.asarray(center)
scale = np.multiply(np.asarray(scale), np.asarray(size))

threshold = 0.2

#predictions
predfile = sys.argv[1]
preds = loadmat(predfile)['preds']              # (1000, 16, 2)
pos_pred_src = transpose(preds, [1, 2, 0])      # (16, 2, 1000)

# index (int)
head = 9
lsho = 13
lelb = 14
lwri = 15
lhip = 3
lkne = 4
lank = 5

rsho = 12
relb = 11
rwri = 10
rkne = 1
rank = 0
rhip = 2


uv_error = pos_pred_src - part
uv_err = np.linalg.norm(uv_error, axis=1)
scaled_uv_err = np.divide(uv_err, scale)
scaled_uv_err = np.multiply(scaled_uv_err, visible)
jnt_count = np.sum(visible, axis=1)
less_than_threshold = np.multiply((scaled_uv_err < threshold), visible)
PCK = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

name = predfile.split(os.sep)[-1]
PCK = np.ma.array(PCK, mask=False)
PCK.mask[6:8] = True
print("Head,   Shoulder, Elbow,  Wrist,   Hip,     Knee,    Ankle,  Mean")
print('{:.2f}   {:.2f}     {:.2f}   {:.2f}    {:.2f}    {:.2f}    {:.2f}   {:.2f}'.format(PCK[head], 0.5 * (PCK[lsho] + PCK[rsho])\
        , 0.5 * (PCK[lelb] + PCK[relb]),0.5 * (PCK[lwri] + PCK[rwri]), 0.5 * (PCK[lhip] + PCK[rhip]), 0.5 * (PCK[lkne] + PCK[rkne]) \
        , 0.5 * (PCK[lank] + PCK[rank]), np.mean(PCK)))
