#from https://github.com/bearpaw/pytorch-pose/blob/master/evaluation/eval_PCKh.py
import sys
from scipy.io import loadmat
from numpy import transpose
import skimage.io as sio
import numpy as np
import os
from h5py import File
import matplotlib
import cv2 as cv
import json
import matplotlib.pyplot as plt
import math

# MPII
#predictions
predfile = sys.argv[1]
preds = loadmat(predfile)['preds']              # (2958, 16, 2)
pos_pred_src = transpose(preds, [1, 2, 0])      # (16, 2, 2958)
pa = [2, 3, 7, 7, 4, 5, 8, 9, 10, 0, 12, 13, 8, 8, 14, 15]

f = File('data/mpii/annot/val.h5', 'r')
data = f['imgname']
data = [d.decode('utf-8') for d in data]
f.close()
save_dir = "exp/mpii/results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i in range(len(data)):
    fn = data[i]
    imagePath = 'data/mpii/images/' + fn
    oriImg = sio.imread(imagePath)
    points = pos_pred_src[:, :, i]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170,0,255],[255,0,255]]
    canvas = oriImg
    stickwidth = 4
    x = points[:, 0]
    y = points[:, 1]

    for n in range(len(x)):
        for child in range(len(pa)):
            if pa[child] == 0:
                continue
            

            x1 = x[pa[child] - 1]
            y1 = y[pa[child] - 1]
            x2 = x[child]
            y2 = y[child]
            
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)

            cv.line(canvas, (x1, y1), (x2, y2), colors[child], 10)


    canvas_with_alpha = cv.cvtColor(canvas, cv.COLOR_BGR2BGRA)
    save_path = os.path.join(save_dir, 'vis_' + fn)
    cv.imwrite(save_path, canvas_with_alpha)



# # LSP
# #predictions
# predfile = sys.argv[1]
# preds = loadmat(predfile)['preds']              # (1000, 16, 2)
# pos_pred_src = transpose(preds, [1, 2, 0])      # (16, 2, 1000)
# pa = [2, 3, 7, 7, 4, 5, 8, 9, 10, 0, 12, 13, 8, 8, 14, 15]

# with open(f'data/lsp/LEEDS_annotations.json', "r") as joints_file:
#     data = json.load(joints_file)
# imgname = ["data/" + d['img_paths'] for d in data if d ['isValidation'] == 1.0]
# save_dir = "exp/lsp/results"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# for i in range(len(imgname)):
#     imagePath = imgname[i]
#     fn = os.path.split(imagePath)[-1]
#     oriImg = sio.imread(imagePath)
#     points = pos_pred_src[:, :, i]

#     colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
#               [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
#               [170,0,255],[255,0,255]]
#     canvas = oriImg
#     stickwidth = 4
#     x = points[:, 0]
#     y = points[:, 1]

#     for n in range(len(x)):
#         for child in range(len(pa)):
#             if pa[child] == 0:
#                 continue
            

#             x1 = x[pa[child] - 1]
#             y1 = y[pa[child] - 1]
#             x2 = x[child]
#             y2 = y[child]
            
#             x1, y1 = int(x1), int(y1)
#             x2, y2 = int(x2), int(y2)

#             cv.line(canvas, (x1, y1), (x2, y2), colors[child], 5)

#     canvas_with_alpha = cv.cvtColor(canvas, cv.COLOR_BGR2BGRA)
#     save_path = os.path.join(save_dir, 'vis_' + fn)
#     cv.imwrite(save_path, canvas_with_alpha)
    