import torch.utils.data as data
import numpy as np
import torch
from h5py import File
import cv2
import utils.human_prior as human_prior
from utils.utils import Rnd, Flip, ShuffleLR
from utils.img import Crop, DrawGaussian, Transform

class MPII(data.Dataset):
    def __init__(self, opt, split):
        print('==> initializing 2D {} data.'.format(split))
        annot = {}
        tags = ['part','center','scale']
        f = File('{}/mpii/annot/{}.h5'.format(opt.dataDir, split), 'r')
        data = f['imgname']
        data = [d.decode('utf-8') for d in data]
        annot['imgname'] = np.asarray(data).copy()
        for tag in tags:
            annot[tag] = np.asarray(f[tag]).copy()
        f.close()

        print('Loaded 2D {} {} samples'.format(split, len(annot['scale'])))
        
        self.split = split
        self.opt = opt
        self.annot = annot
    
    def LoadImage(self, index):
        path = '{}/mpii/images/{}'.format(self.opt.dataDir, self.annot['imgname'][index])
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
            s = s * (2 ** Rnd(human_prior.scale))
            r = 0 if np.random.random() < 0.6 else Rnd(human_prior.rotate)
        inp = Crop(img, c, s, r, human_prior.inputRes) / 256.
        out = np.zeros((human_prior.nJoints, human_prior.outputRes, human_prior.outputRes))
        for i in range(human_prior.nJoints):
            if pts[i][0] > 1:
                pt = Transform(pts[i], c, s, r, human_prior.outputRes)
                out[i] = DrawGaussian(out[i], pt, human_prior.hmGauss) 
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
    def __init__(self, opt, split):
        print('==> initializing 2D {} data.'.format(split))
        annot = {}
        tags = ['part','center','scale']
        f = File('{}/lsp/joints.mat'.format(opt.dataDir), 'r')
        data = f['imgname']
        data = [d.decode('utf-8') for d in data]
        annot['imgname'] = np.asarray(data).copy()
        for tag in tags:
            annot[tag] = np.asarray(f[tag]).copy()
        f.close()

        print('Loaded 2D {} {} samples'.format(split, len(annot['scale'])))
        
        self.split = split
        self.opt = opt
        self.annot = annot
    
    def LoadImage(self, index):
        path = '{}/mpii/images/{}'.format(self.opt.dataDir, self.annot['imgname'][index])
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
            s = s * (2 ** Rnd(human_prior.scale))
            r = 0 if np.random.random() < 0.6 else Rnd(human_prior.rotate)
        inp = Crop(img, c, s, r, human_prior.inputRes) / 256.
        out = np.zeros((human_prior.nJoints, human_prior.outputRes, human_prior.outputRes))
        for i in range(human_prior.nJoints):
            if pts[i][0] > 1:
                pt = Transform(pts[i], c, s, r, human_prior.outputRes)
                out[i] = DrawGaussian(out[i], pt, human_prior.hmGauss) 
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

