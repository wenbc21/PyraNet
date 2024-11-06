# -*- coding:utf-8 -*-
import torch.nn as nn
import math
import torch.nn.functional as F


class Concat(nn.Module):
    # for PRM-C
    def __init__(self):
        super(Concat,self).__init__()
    
    def forward(self, input):
        return torch.cat(input,1)

class DownSample(nn.Module):
    def __init__(self,scale):
        super(DownSample, self).__init__()
        self.scale = scale

    def forward(self, x):
        sample = F.interpolate(x,scale_factor=self.scale)
        return sample

class BnResidualConv1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(BnResidualConv1,self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(self.in_channels,self.out_channels,kernel_size=1,padding=0)

    def forward(self, x):
        x = self.bn(x)
        return self.conv(F.relu(x))

class BnResidualConv3(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(BnResidualConv3,self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.bn = nn.BatchNorm2d(self.in_channels)
        self.conv = nn.Conv2d(self.in_channels,self.out_channels,kernel_size=3,padding=1)

    def forward(self, x):
        x = self.bn(x)
        return self.conv(F.relu(x))

class PRM(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(PRM,self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.reo1 = BnResidualConv1(in_channels=self.in_channels,out_channels=int(self.out_channels/2))
        # When choose PRM-A，uncomment reo2-reo4
        self.reo2 = BnResidualConv1(in_channels=self.in_channels,out_channels=int(self.out_channels/2))
        self.reo3 = BnResidualConv1(in_channels=self.in_channels,out_channels=int(self.out_channels/2))
        self.reo4 = BnResidualConv1(in_channels=self.in_channels,out_channels=int(self.out_channels/2))
        self.reo5 = BnResidualConv1(in_channels=self.in_channels,out_channels=int(self.out_channels))

        # downsample to multi-scale
        self.down1 = DownSample(scale=pow(2,-1))      
        self.down2 = DownSample(scale=pow(2,-0.75))
        self.down3 = DownSample(scale=pow(2,-0.5))
        self.down4 = DownSample(scale=pow(2,-0.25))

        self.ret1 = BnResidualConv3(in_channels=int(self.out_channels/2),out_channels=self.out_channels)
        self.ret2 = BnResidualConv3(in_channels=int(self.out_channels/2),out_channels=self.out_channels)
        self.ret3 = BnResidualConv3(in_channels=int(self.out_channels/2),out_channels=self.out_channels)
        self.ret4 = BnResidualConv3(in_channels=int(self.out_channels/2),out_channels=self.out_channels)
        # PRM-B
        self.smooth = BnResidualConv1(in_channels=self.out_channels,out_channels=self.out_channels)
        # PRM-C
        # self.smooth = BnResidualConv1(in_channels=self.out_channels*4,out_channels=self.out_channels)

    def forward(self, x):
        identity = self.reo5(x)
        size = identity.size()[-2],identity.size()[-1]
        # multi-scale information
        # BN + relu + 1x1 conv
        scale1 = self.reo1(x)
        # scale2 = self.reo2(x)
        # scale3 = self.reo3(x)
        scale2 = self.reo1(x)
        scale3 = self.reo1(x)
        scale4 = self.reo1(x)

        ratio1 = F.interpolate(self.ret1(self.down1(scale1)),size=size)
        ratio2 = F.interpolate(self.ret2(self.down2(scale2)),size=size)
        ratio3 = F.interpolate(self.ret3(self.down3(scale3)),size=size)
        ratio4 = F.interpolate(self.ret4(self.down4(scale4)),size=size)
        # PRM-B
        tmp_ret = ratio1+ratio2+ratio3+ratio4
        # PRM-C,replace smooth with PRM-C's smooth layer
        # tmp_ret = torch.cat((ratio1,ratio2,ratio3,ratio4),1)
        smooth = self.smooth(tmp_ret)
        ret = identity + smooth
        # size equal
        return ret
        #return identity+ratio1+ratio2+ratio3


class Hourglass(nn.Module):
    def __init__(self, n, nModules, nFeats):
        super(Hourglass, self).__init__()
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats
        
        _up1_, _low1_, _low2_, _low3_ = [], [], [], []
        for j in range(self.nModules):
            _up1_.append(PRM(self.nFeats, self.nFeats))
        self.low1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        for j in range(self.nModules):
            _low1_.append(PRM(self.nFeats, self.nFeats))
        if self.n > 1:
            self.low2 = Hourglass(n - 1, self.nModules, self.nFeats)
        else:
            for j in range(self.nModules):
                _low2_.append(PRM(self.nFeats, self.nFeats))
            self.low2_ = nn.ModuleList(_low2_)
        
        for j in range(self.nModules):
            _low3_.append(PRM(self.nFeats, self.nFeats))
        self.up1_ = nn.ModuleList(_up1_)
        self.low1_ = nn.ModuleList(_low1_)
        self.low3_ = nn.ModuleList(_low3_)
        
        self.up2 = nn.Upsample(scale_factor = 2)
        
    def forward(self, x):
        up1 = x
        for j in range(self.nModules):
            up1 = self.up1_[j](up1)
        
        low1 = self.low1(x)
        for j in range(self.nModules):
            low1 = self.low1_[j](low1)
        
        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for j in range(self.nModules):
                low2 = self.low2_[j](low2)
        
        low3 = low2
        for j in range(self.nModules):
            low3 = self.low3_[j](low3)
        up2 = self.up2(low3)
        
        return up1 + up2

class PyramidHourglassNet(nn.Module):
    def __init__(self, nStack, nModules, nFeats, numOutput):
        super(PyramidHourglassNet, self).__init__()
        self.nStack = nStack
        self.nModules = nModules
        self.nFeats = nFeats
        self.numOutput = numOutput

        # add a pyramid structure
        self.conv1 = nn.Conv2d(3,64,kernel_size=1,stride=2)
        self.prm1 = PRM(64,64)
        self.ipool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.prm2 = PRM(64,self.nFeats)
        self.relu = nn.ReLU(inplace=True)

        # stacked hourglass
        self.relu = nn.ReLU(inplace = True)
        self.r1 = PRM(64,128)
        # self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.r4 = PRM(128,128)
        self.r5 = PRM(128,self.nFeats)

        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_, _reg_ = [], [], [], [], [], [], []
        for i in range(self.nStack):
            _hourglass.append(Hourglass(4, self.nModules, self.nFeats))
            for j in range(self.nModules):
                _Residual.append(PRM(self.nFeats, self.nFeats))
            lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias = True, kernel_size = 1, stride = 1), 
                                                    nn.BatchNorm2d(self.nFeats), self.relu)
            _lin_.append(lin)
            _tmpOut.append(nn.Conv2d(self.nFeats, self.numOutput, bias = True, kernel_size = 1, stride = 1))
            if i < self.nStack - 1:
                _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, bias = True, kernel_size = 1, stride = 1))
                _tmpOut_.append(nn.Conv2d(self.numOutput, self.nFeats, bias = True, kernel_size = 1, stride = 1))
                
        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmpOut = nn.ModuleList(_tmpOut)
        self.ll_ = nn.ModuleList(_ll_)
        self.tmpOut_ = nn.ModuleList(_tmpOut_)
        # self.upout = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prm1(x)
        x = self.ipool(x)
        x = self.prm2(x)
        
        out = []
        
        for i in range(self.nStack):
            hg = self.hourglass[i](x)
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i * self.nModules + j](ll)
            ll = self.lin_[i](ll)
            tmpOut = self.tmpOut[i](ll)
            # tmpOut = F.upsample(tmpOut,scale_factor=2)
            out.append(tmpOut)
            if i < self.nStack - 1:
                ll_ = self.ll_[i](ll)
                tmpOut_ = self.tmpOut_[i](tmpOut)
                x = x + ll_ + tmpOut_

        return out

