import torch.utils.data as data
import numpy as np
import torch
from h5py import File
import cv2
import scipy.io
import json
import utils.human_prior as hp
from utils.utils import Rnd, Flip, ShuffleLR
from utils.img import Crop, DrawGaussian, Transform

class MPII(data.Dataset):
    def __init__(self, args, split):
        print('==> initializing 2D {} data.'.format(split))
        annot = {}
        tags = ['part','center','scale']
        f = File('{}/mpii/annot/{}.h5'.format(args.dataDir, split), 'r')
        data = f['imgname']
        data = [d.decode('utf-8') for d in data]
        annot['imgname'] = np.asarray(data).copy()
        for tag in tags:
            annot[tag] = np.asarray(f[tag]).copy()
        f.close()

        print('Loaded 2D {} {} samples'.format(split, len(annot['scale'])))
        
        self.split = split
        self.args = args
        self.annot = annot
    
    def LoadImage(self, index):
        path = '{}/mpii/images/{}'.format(self.args.dataDir, self.annot['imgname'][index])
        img = cv2.imread(path)
        return img
    
    def GetPartInfo(self, index):
        pts = self.annot['part'][index].copy()
        c = self.annot['center'][index].copy()
        s = self.annot['scale'][index]
        s = s * 200
        return pts, c, s
            
    def __getitem__(self, index):
        img = self.LoadImage(index)
        pts, c, s = self.GetPartInfo(index)
        r = 0
        
        if self.split == 'train':
            s = s * (2 ** Rnd(hp.scale))
            r = 0 if np.random.random() < 0.6 else Rnd(hp.rotate)
        inp = Crop(img, c, s, r, hp.inputRes) / 256.
        out = np.zeros((hp.nJoints, hp.outputRes, hp.outputRes))
        for i in range(hp.nJoints):
            if pts[i][0] > 1:
                pt = Transform(pts[i], c, s, r, hp.outputRes)
                out[i] = DrawGaussian(out[i], pt, hp.hmGauss) 
        if self.split == 'train':
            if np.random.random() < 0.5:
                inp = Flip(inp)
                out = ShuffleLR(Flip(out))
            inp[0] = np.clip(inp[0] * (np.random.random() * (0.4) + 0.6), 0, 1)
            inp[1] = np.clip(inp[1] * (np.random.random() * (0.4) + 0.6), 0, 1)
            inp[2] = np.clip(inp[2] * (np.random.random() * (0.4) + 0.6), 0, 1)
            meta = np.zeros(1)
        else:
            meta = {'index' : index, 'center' : c, 'scale' : s, 'rotate': r}
        
        return inp, out, meta
        
    def __len__(self):
        return len(self.annot['scale'])


class LSP(data.Dataset):
    def __init__(self, args, split):
        print('==> initializing 2D {} data.'.format(split))
        with open(f'{args.dataDir}/lsp/LEEDS_annotations.json', "r") as joints_file:
            data = json.load(joints_file)
        
        annot = {}
        imgname = []
        part = []
        center = []
        scale = []
        
        for d in data :
            if (split == "train" and d['isValidation'] == 0.0) or (split == "val" and d['isValidation'] == 1.0):
                imgname.append(d['img_paths'])
                part.append(np.array(d['joint_self'])[:, :2])
                center.append(d['objpos'])
                scale.append(d['scale_provided'])
        
        annot['imgname'] = np.asarray(imgname).copy()
        annot['part'] = np.asarray(part).copy()
        annot['center'] = np.asarray(center).copy()
        annot['scale'] = np.asarray(scale).copy()

        print('Loaded 2D {} {} samples'.format(split, len(annot['scale'])))
        
        self.split = split
        self.args = args
        self.annot = annot
    
    def LoadImage(self, index):
        path = '{}/{}'.format(self.args.dataDir, self.annot['imgname'][index])
        img = cv2.imread(path)
        return img
    
    def GetPartInfo(self, index):
        pts = self.annot['part'][index].copy()
        c = self.annot['center'][index].copy()
        s = self.annot['scale'][index]
        s = s * 200 * (368 / 256)
        return pts, c, s
            
    def __getitem__(self, index):
        img = self.LoadImage(index)
        pts, c, s = self.GetPartInfo(index)
        r = 0
        
        if self.split == 'train':
            s = s * (2 ** Rnd(hp.scale))
            r = 0 if np.random.random() < 0.6 else Rnd(hp.rotate)
        inp = Crop(img, c, s, r, hp.inputRes) / 256.
        out = np.zeros((hp.nJoints, hp.outputRes, hp.outputRes))
        for i in range(hp.nJoints):
            if pts[i][0] > 1:
                pt = Transform(pts[i], c, s, r, hp.outputRes)
                out[i] = DrawGaussian(out[i], pt, hp.hmGauss) 
        if self.split == 'train':
            if np.random.random() < 0.5:
                inp = Flip(inp)
                out = ShuffleLR(Flip(out))
            inp[0] = np.clip(inp[0] * (np.random.random() * (0.4) + 0.6), 0, 1)
            inp[1] = np.clip(inp[1] * (np.random.random() * (0.4) + 0.6), 0, 1)
            inp[2] = np.clip(inp[2] * (np.random.random() * (0.4) + 0.6), 0, 1)
            meta = np.zeros(1)
        else:
            meta = {'index' : index, 'center' : c, 'scale' : s, 'rotate': r}
        
        return inp, out, meta
        
    def __len__(self):
        return len(self.annot['scale'])
