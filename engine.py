import torch
import numpy as np
from utils.utils import AverageMeter, Flip, ShuffleLR
from utils.eval import Accuracy, getPreds, finalPreds
import cv2
import utils.human_prior as human_prior
import sys
from tqdm import tqdm

def step(split, epoch, opt, dataLoader, model, criterion, optimizer = None):
    if split == 'train':
        model.train()
    else:
        model.eval()
    Loss, Acc = AverageMeter(), AverageMeter()
    preds = []

    nIters = len(dataLoader)
    dataLoader = tqdm(dataLoader, file=sys.stdout)
    for i, (input, target, meta) in enumerate(dataLoader):
        input_var = torch.autograd.Variable(input).float().cuda()
        target_var = torch.autograd.Variable(target).float().cuda()
        output = model(input_var)

        loss = criterion(output[0], target_var)
        for k in range(1, opt.nStack):
            loss += criterion(output[k], target_var)
        
        Loss.update(loss.data.item(), input.size(0))
        Acc.update(Accuracy((output[opt.nStack - 1].data).cpu().numpy(), (target_var.data).cpu().numpy()))
        if split == 'train':
            # train
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            input_ = input.cpu().numpy()
            input_[0] = Flip(input_[0]).copy()
            inputFlip_var = torch.autograd.Variable(torch.from_numpy(input_).view(1, input_.shape[1], human_prior.inputRes, human_prior.inputRes)).float().cuda()
            outputFlip = model(inputFlip_var)
            outputFlip = ShuffleLR(Flip((outputFlip[opt.nStack - 1].data).cpu().numpy()[0])).reshape(1, human_prior.nJoints, human_prior.outputRes, human_prior.outputRes)
            output_ = ((output[opt.nStack - 1].data).cpu().numpy() + outputFlip) / 2
            preds.append(finalPreds(output_, meta['center'], meta['scale'], meta['rotate'])[0])

        dataLoader.desc = '{split} Epoch: [{0}][{1}/{2}] | Loss {loss.avg:.4f} | Acc {Acc.avg:.4f} ({Acc.val:.4f})'.format(
            epoch, i, nIters, loss=Loss, Acc=Acc, split = split)

    return Loss.avg, Acc.avg, preds


def train(epoch, opt, train_loader, model, criterion, optimizer):
    return step('train', epoch, opt, train_loader, model, criterion, optimizer)

def val(epoch, opt, val_loader, model, criterion):
    return step('val', epoch, opt, val_loader, model, criterion)
