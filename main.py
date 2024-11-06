import os
import argparse

import torch
import torch.utils.data
import ref as ref
from dataset import MPII
from model import PyramidHourglassNet
from engine import train, val
from utils.utils import adjust_learning_rate
import scipy.io as sio


def get_args_parser():
    parser = argparse.ArgumentParser('Pyranet training and evaluation script for pose estimation', add_help=False)
    parser.add_argument('-expID', default = 'default', help = 'Experiment ID')
    parser.add_argument('-device', type = int, default = 2, help = 'GPU id')
    parser.add_argument('-test', action = 'store_true', help = 'test')
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('-demo', default = '', help = 'path/to/demo/image')
    parser.add_argument('-loadModel', default = 'none', help = 'Provide full path to a previously trained model')
    parser.add_argument('-nFeats', type = int, default = 256, help = '# features in the hourglass')
    parser.add_argument('-nStack', type = int, default = 2, help = '# hourglasses to stack')
    parser.add_argument('-nModules', type = int, default = 2, help = '# residual modules at each hourglass')
    parser.add_argument('-numOutput', type = int, default = ref.nJoints, help = '# ouput')
    
    parser.add_argument('-LR', type = float, default = 2.5e-4, help = 'Learning Rate')
    parser.add_argument('-dropLR', type = int, default = 1000000, help = 'drop LR')
    parser.add_argument('-momentum', type = float, default = 0.0, help = 'momentum')
    parser.add_argument('-weightDecay', type = float, default = 0.0, help = 'weightDecay')
    parser.add_argument('-alpha', type = float, default = 0.99, help = 'alpha')
    parser.add_argument('-epsilon', type = float, default = 1e-8, help = 'epsilon')
    parser.add_argument('-nEpochs', type = int, default = 100, help = '#training epochs')
    parser.add_argument('-valIntervals', type = int, default = 2, help = '#valid intervel')
    parser.add_argument('-trainBatch', type = int, default = 12, help = 'Mini-batch size')
    parser.add_argument('-arch', default = 'hg', help = 'hg | hg-reg | resnet-xxx')
    
    parser.add_argument('-dataDir', type = str, default = './data', help = 'data path')
    parser.add_argument('-expDir', type = str, default = './exp', help = 'exp path')

    return parser


def main(args):
    
    train_dataset = MPII(args, 'val')
    val_dataset = MPII(args, 'val')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = args.trainBatch, 
        shuffle = True,
        num_workers = args.num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size = 1, 
        shuffle = False,
        num_workers = args.num_workers
    )
    
    model = PyramidHourglassNet(args.nStack, args.nModules, args.nFeats, args.numOutput)
    optimizer = torch.optim.RMSprop(
        model.parameters(), 
        lr=args.LR, 
        alpha = args.alpha,
        eps = args.epsilon,
        weight_decay = args.weightDecay,
        momentum = args.momentum
    )

    if args.loadModel != 'none':
        checkpoint = torch.load(args.loadModel)
        if type(checkpoint) == type({}):
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint.state_dict()    
        model.load_state_dict(state_dict)
    
    model = model.cuda()
    criterion = torch.nn.MSELoss().cuda()

    for epoch in range(1, args.nEpochs + 1):
        log_dict_train, _ = train(epoch, args, train_loader, model, criterion, optimizer)
        
        if epoch % args.valIntervals == 0:
            log_dict_val, preds = val(epoch, args, val_loader, model, criterion)
            torch.save(model, os.path.join(args.saveDir, 'model_{}.pth'.format(epoch)))
            sio.savemat(os.path.join(args.saveDir, 'preds_{}.mat'.format(epoch)), mdict = {'preds': preds})
        
        if epoch % args.dropLR == 0:
            lr = args.LR * (0.1 ** (epoch // args.dropLR))
            print('Drop LR to {}'.format(lr))
            adjust_learning_rate(optimizer, lr)
            
    torch.save(model.cpu(), os.path.join(args.saveDir, 'model_cpu.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pyranet training and evaluation script for pose estimation', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.test:
        args.expID = args.expID + 'TEST'
    args.saveDir = os.path.join(args.expDir, args.expID)

    if not os.path.exists(args.saveDir):
        os.makedirs(args.saveDir)
    
    main(args)