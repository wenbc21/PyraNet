import os
import argparse

import torch
import torch.utils.data
import torch.optim as optim
import utils.human_prior as hp
from dataset import MPII, LSP
from model import PyraNet
from engine import train, val
import scipy.io as sio


def get_args_parser():
    parser = argparse.ArgumentParser('Pyranet training and evaluation script for pose estimation', add_help=False)
    
    # experiment
    parser.add_argument('--dataDir', type = str, default = './data', help = 'data path')
    parser.add_argument('--expDir', type = str, default = './exp', help = 'exp path')
    parser.add_argument('--expID', default = 'default', help = 'Experiment ID')
    parser.add_argument('--device', type = int, default = 0, help = 'GPU id')
    
    # model
    parser.add_argument('--loadModel', default = 'none', help = 'Provide full path to a previously trained model')
    parser.add_argument('--nFeats', type = int, default = 256, help = '# features in the hourglass')
    parser.add_argument('--nStack', type = int, default = 2, help = '# hourglasses to stack')
    parser.add_argument('--nModules', type = int, default = 2, help = '# residual modules at each hourglass')
    parser.add_argument('--numOutput', type = int, default = hp.nJoints, help = '# output joint number')
    
    # training
    parser.add_argument('--LR', type = float, default = 7e-4, help = 'Learning Rate')
    parser.add_argument('--momentum', type = float, default = 0.0, help = 'momentum')
    parser.add_argument('--weight_decay', type = float, default = 0.0, help = 'weight decay')
    parser.add_argument('--alpha', type = float, default = 0.99, help = 'alpha')
    parser.add_argument('--epsilon', type = float, default = 1e-8, help = 'epsilon')
    parser.add_argument('--epochs', type = int, default = 200, help = '#training epochs')
    parser.add_argument('--val_intervals', type = int, default = 20, help = '#valid intervel')
    parser.add_argument('--batchsize', type = int, default = 24, help = 'Mini-batch size')
    parser.add_argument('--num_workers', type=int, default=4)

    return parser


def main(args):
    
    # train_dataset = MPII(args, 'train')
    # val_dataset = MPII(args, 'val')
    train_dataset = LSP(args, 'train')
    val_dataset = LSP(args, 'val')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = args.batchsize, 
        shuffle = True,
        num_workers = args.num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size = 1, 
        shuffle = False,
        num_workers = args.num_workers
    )
    
    model = PyraNet(args.nStack, args.nModules, args.nFeats, args.numOutput)
    optimizer = torch.optim.RMSprop(
        model.parameters(), 
        lr=args.LR, 
        alpha = args.alpha,
        eps = args.epsilon,
        weight_decay = args.weight_decay,
        momentum = args.momentum
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 170], gamma=0.1)

    model = model.cuda()
    criterion = torch.nn.MSELoss().cuda()
    
    start_epoch = 1
    if args.loadModel != 'none':
        checkpoint = torch.load(args.loadModel, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.epochs + 1):
        loss, acc, _ = train(epoch, args, train_loader, model, criterion, optimizer)
        log_file = open(os.path.join(args.saveDir, "training_log.txt"), 'a+')
        log_file.write("Epoch {}, training loss {:6f}, training acc {:6f}\n".format(epoch, loss, acc))
        if epoch % args.val_intervals == 0:
            loss, acc, preds = val(epoch, args, val_loader, model, criterion)
            torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}, 
                        os.path.join(args.saveDir, 'model_{}.pth'.format(epoch)))
            sio.savemat(os.path.join(args.saveDir, 'preds_{}.mat'.format(epoch)), mdict = {'preds': preds})
            log_file.write("Epoch {}, validation loss {:6f}, validation acc {:6f}\n".format(epoch, loss, acc))
        log_file.close()
        
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pyranet training and evaluation script for pose estimation', parents=[get_args_parser()])
    args = parser.parse_args()

    args.saveDir = os.path.join(args.expDir, args.expID)
    if not os.path.exists(args.saveDir):
        os.makedirs(args.saveDir)
    
    main(args)