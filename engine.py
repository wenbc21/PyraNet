import torch
from utils.utils import AverageMeter, Flip, ShuffleLR
from utils.eval import Accuracy, getPreds, finalPreds
import utils.human_prior as hp
import sys
from tqdm import tqdm


def train(epoch, args, train_loader, model, criterion, optimizer):
    model.train()
    Loss, Acc = AverageMeter(), AverageMeter()
    preds = []

    nIters = len(train_loader)
    train_loader = tqdm(train_loader, file=sys.stdout)
    for i, (input, target, meta) in enumerate(train_loader):
        input_var = torch.autograd.Variable(input).float().cuda()
        target_var = torch.autograd.Variable(target).float().cuda()
        output = model(input_var)

        loss = criterion(output[0], target_var)
        for k in range(1, args.nStack):
            loss += criterion(output[k], target_var)
        
        Loss.update(loss.data.item(), input.size(0))
        Acc.update(Accuracy((output[args.nStack - 1].data).cpu().numpy(), (target_var.data).cpu().numpy()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loader.desc = 'train Epoch: [{0}][{1}/{2}] | Loss {loss.avg:.4f} | Acc {Acc.avg:.4f} ({Acc.val:.4f})'.format(
            epoch, i, nIters, loss=Loss, Acc=Acc)

    return Loss.avg, Acc.avg, preds


def val(epoch, args, val_loader, model, criterion):

    model.eval()
    Loss, Acc = AverageMeter(), AverageMeter()
    preds = []

    nIters = len(val_loader)
    val_loader = tqdm(val_loader, file=sys.stdout)
    for i, (input, target, meta) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input).float().cuda()
        target_var = torch.autograd.Variable(target).float().cuda()
        output = model(input_var)

        loss = criterion(output[0], target_var)
        for k in range(1, args.nStack):
            loss += criterion(output[k], target_var)
        
        Loss.update(loss.data.item(), input.size(0))
        Acc.update(Accuracy((output[args.nStack - 1].data).cpu().numpy(), (target_var.data).cpu().numpy()))
        
        input_ = input.cpu().numpy()
        input_[0] = Flip(input_[0]).copy()
        inputFlip_var = torch.autograd.Variable(torch.from_numpy(input_).view(1, input_.shape[1], hp.inputRes, hp.inputRes)).float().cuda()
        outputFlip = model(inputFlip_var)
        outputFlip = ShuffleLR(Flip((outputFlip[args.nStack - 1].data).cpu().numpy()[0])).reshape(1, hp.nJoints, hp.outputRes, hp.outputRes)
        output_ = ((output[args.nStack - 1].data).cpu().numpy() + outputFlip) / 2
        preds.append(finalPreds(output_, meta['center'], meta['scale'], meta['rotate'])[0])

        val_loader.desc = 'val Epoch: [{0}][{1}/{2}] | Loss {loss.avg:.4f} | Acc {Acc.avg:.4f} ({Acc.val:.4f})'.format(
            epoch, i, nIters, loss=Loss, Acc=Acc)

    return Loss.avg, Acc.avg, preds
