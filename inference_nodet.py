import os
import argparse

import torch
import torch.utils.data
import cv2
import numpy as np
from tqdm import tqdm
import utils.human_prior as hp
from model import PyraNet
from utils.utils import Flip, ShuffleLR
pa = [2, 3, 7, 7, 4, 5, 8, 9, 10, 0, 12, 13, 8, 8, 14, 15]
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170,0,255],[255,0,255]]


def get_args_parser():
    parser = argparse.ArgumentParser('Pyranet inference script for pose estimation', add_help=False)
    
    # experiment
    parser.add_argument('--dataDir', type = str, default = './demo', help = 'data path')
    parser.add_argument('--device', type = int, default = 0, help = 'GPU id')
    
    # model
    parser.add_argument('--loadModel', default = 'none', help = 'Provide full path to a previously trained model')
    parser.add_argument('--nFeats', type = int, default = 256, help = '# features in the hourglass')
    parser.add_argument('--nStack', type = int, default = 2, help = '# hourglasses to stack')
    parser.add_argument('--nModules', type = int, default = 2, help = '# residual modules at each hourglass')
    parser.add_argument('--numOutput', type = int, default = hp.nJoints, help = '# output joint number')

    return parser


def main(args):
    
    model = PyraNet(args.nStack, args.nModules, args.nFeats, args.numOutput)
    model = model.cuda()
    
    if args.loadModel == 'none':
        print("You have to use pretrained model! ")
        exit()
    
    checkpoint = torch.load(args.loadModel, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    img_dir = os.path.join(args.dataDir, "images")
    res_dir = os.path.join(args.dataDir, "results")
    os.makedirs(res_dir, exist_ok=True)
    
    image_path = [item.path for item in os.scandir(img_dir) if item.is_file()]
    
    for img_i in tqdm(range(len(image_path))) :
        
        img_path = image_path[img_i]
        save_path = os.path.join(res_dir, os.path.split(img_path)[-1])
        
        img_ori = cv2.imread(img_path)
        height, width = img_ori.shape[:2]
        
        if width > height:
            new_width = 256
            new_height = int(height * 256 / width)
        else:
            new_height = 256
            new_width = int(width * 256 / height)

        resized_image = cv2.resize(img_ori, (new_width, new_height))

        top = (256 - new_height) // 2
        bottom = 256 - new_height - top
        left = (256 - new_width) // 2
        right = 256 - new_width - left

        padded_image_ori = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded_image = padded_image_ori.transpose(2, 0, 1).astype(float) / 256.
        padded_image = torch.tensor(padded_image).unsqueeze(0)
    
        input_var = torch.autograd.Variable(padded_image).float().cuda()
        output = model(input_var)
        
        input_ = padded_image.cpu().numpy()
        input_[0] = Flip(input_[0]).copy()
        inputFlip_var = torch.autograd.Variable(torch.from_numpy(input_).view(1, input_.shape[1], hp.inputRes, hp.inputRes)).float().cuda()
        outputFlip = model(inputFlip_var)
        outputFlip = ShuffleLR(Flip((outputFlip[args.nStack - 1].data).cpu().numpy()[0])).reshape(1, hp.nJoints, hp.outputRes, hp.outputRes)
        output_ = ((output[args.nStack - 1].data).cpu().numpy() + outputFlip) / 2
        
        hm = (output[args.nStack - 1].data).cpu().numpy()
        res = hm.shape[2]
        hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
        idx = np.argmax(hm, axis = 2)
        preds = np.zeros((hm.shape[0], hm.shape[1], 2))
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                preds[i, j, 0], preds[i, j, 1] = idx[i, j] % res, idx[i, j] / res
        preds = (preds[0] - 32) * (256/64) + 128 - (left, top)
        preds = (preds - (new_width/2, new_height/2))
        preds[:, 0] = preds[:, 0] * width / new_width
        preds[:, 1] = preds[:, 1] * height / new_height
        preds += (width/2, height/2)
        
        canvas = img_ori
        x = preds[:, 0]
        y = preds[:, 1]

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

                cv2.line(canvas, (x1, y1), (x2, y2), colors[child], 10)

        canvas_with_alpha = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
        cv2.imwrite(save_path, canvas_with_alpha)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pyranet inference script for pose estimation', parents=[get_args_parser()])
    args = parser.parse_args()
    
    main(args)