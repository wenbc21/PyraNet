#from https://github.com/bearpaw/pytorch-pose/blob/master/evaluation/eval_PCKh.py
import sys
from scipy.io import loadmat
from numpy import transpose
import skimage.io as sio
import numpy as np
import os
from h5py import File


# # dict_keys(['__header__', '__version__', '__globals__', 'type', 'output_joints', 
# # 'annolist', 'keypointsAll', 'RELEASE_img_index', 'RELEASE_person_index'])
# detection = loadmat('tools/data/detections.mat')
# # [[    7    18    28 ... 24952 24980 24985]] (1, 2958)
# det_idxs = detection['RELEASE_img_index']

threshold = 0.5
SC_BIAS = 0.8

# dict_keys(['__header__', '__version__', '__globals__', 
# 'jnt_missing', 'pos_pred_src', 'pos_gt_src', 'headboxes_src', 'dataset_joints'])
dict_val = loadmat('tools/data/detections_our_format.mat')

dataset_joints = dict_val['dataset_joints']     # (1, 16)
jnt_missing = dict_val['jnt_missing']           # (16, 2958)
pos_pred_src = dict_val['pos_pred_src']         # (16, 2, 2958)
pos_gt_src = dict_val['pos_gt_src']             # (16, 2, 2958)
headboxes_src = dict_val['headboxes_src']       # (2, 2, 2958)

#predictions
predfile = sys.argv[1]
preds = loadmat(predfile)['preds']              # (2958, 16, 2)
pos_pred_src = transpose(preds, [1, 2, 0])      # (16, 2, 2958)

# index (int)
head = np.where(dataset_joints == 'head')[1][0]
lsho = np.where(dataset_joints == 'lsho')[1][0]
lelb = np.where(dataset_joints == 'lelb')[1][0]
lwri = np.where(dataset_joints == 'lwri')[1][0]
lhip = np.where(dataset_joints == 'lhip')[1][0]
lkne = np.where(dataset_joints == 'lkne')[1][0]
lank = np.where(dataset_joints == 'lank')[1][0]

rsho = np.where(dataset_joints == 'rsho')[1][0]
relb = np.where(dataset_joints == 'relb')[1][0]
rwri = np.where(dataset_joints == 'rwri')[1][0]
rkne = np.where(dataset_joints == 'rkne')[1][0]
rank = np.where(dataset_joints == 'rank')[1][0]
rhip = np.where(dataset_joints == 'rhip')[1][0]

jnt_visible = 1 - jnt_missing
uv_error = pos_pred_src - pos_gt_src
uv_err = np.linalg.norm(uv_error, axis=1)
headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
headsizes = np.linalg.norm(headsizes, axis=0)
headsizes *= SC_BIAS
scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
scaled_uv_err = np.divide(uv_err, scale)
scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
jnt_count = np.sum(jnt_visible, axis=1)
less_than_threshold = np.multiply((scaled_uv_err < threshold), jnt_visible)
PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

name = predfile.split(os.sep)[-1]
PCKh = np.ma.array(PCKh, mask=False)
PCKh.mask[6:8] = True
print("Head,   Shoulder, Elbow,  Wrist,   Hip,     Knee,    Ankle,  Mean")
print('{:.2f}   {:.2f}     {:.2f}   {:.2f}    {:.2f}    {:.2f}    {:.2f}   {:.2f}'.format(PCKh[head], 0.5 * (PCKh[lsho] + PCKh[rsho])\
        , 0.5 * (PCKh[lelb] + PCKh[relb]),0.5 * (PCKh[lwri] + PCKh[rwri]), 0.5 * (PCKh[lhip] + PCKh[rhip]), 0.5 * (PCKh[lkne] + PCKh[rkne]) \
        , 0.5 * (PCKh[lank] + PCKh[rank]), np.mean(PCKh)))
