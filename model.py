import torch.nn as nn
import torch.nn.functional as F
import math

class BnReluConv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BnReluConv1, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)
        return x

class BnReluConv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BnReluConv3, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)
        return x

class BnReluDilatedConv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BnReluDilatedConv3,self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)
        return x

class PRM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PRM, self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.out_channels_half = int(self.out_channels / 2)
        self.reo1 = BnReluConv1(in_channels=self.in_channels, out_channels=self.out_channels_half)              # for PRM-B, C, D
        self.reo2 = BnReluConv1(in_channels=self.in_channels, out_channels=self.out_channels_half)              # for PRM-A
        self.reo3 = BnReluConv1(in_channels=self.in_channels, out_channels=self.out_channels_half)              # for PRM-A
        self.reo4 = BnReluConv1(in_channels=self.in_channels, out_channels=self.out_channels_half)              # for PRM-A
        self.reo5 = BnReluConv1(in_channels=self.in_channels, out_channels=self.out_channels)                   # identity branch

        self.ret1 = BnReluConv3(in_channels=self.out_channels_half, out_channels=self.out_channels)             # for PRM-A, B, C
        self.ret2 = BnReluConv3(in_channels=self.out_channels_half, out_channels=self.out_channels)             # for PRM-A, B, C
        self.ret3 = BnReluConv3(in_channels=self.out_channels_half, out_channels=self.out_channels)             # for PRM-A, B, C
        self.ret4 = BnReluConv3(in_channels=self.out_channels_half, out_channels=self.out_channels)             # for PRM-A, B, C
        # self.ret1 = BnReluDilatedConv3(in_channels=self.out_channels_half, out_channels=self.out_channels)    # for PRM-D
        # self.ret2 = BnReluDilatedConv3(in_channels=self.out_channels_half, out_channels=self.out_channels)    # for PRM-D
        # self.ret3 = BnReluDilatedConv3(in_channels=self.out_channels_half, out_channels=self.out_channels)    # for PRM-D
        # self.ret4 = BnReluDilatedConv3(in_channels=self.out_channels_half, out_channels=self.out_channels)    # for PRM-D

        self.smooth = BnReluConv1(in_channels=self.out_channels, out_channels=self.out_channels)                # for PRM-A, B, D
        # self.smooth = BnReluConv1(in_channels=self.out_channels*4, out_channels=self.out_channels)            # for PRM-C

    def forward(self, x):
        identity = self.reo5(x)
        size = identity.size()[-2], identity.size()[-1]

        # multi-scale information
        # scale1 = self.reo1(x)     # for PRM-A
        # scale2 = self.reo2(x)     # for PRM-A
        # scale3 = self.reo3(x)     # for PRM-A
        # scale4 = self.reo4(x)     # for PRM-A
        scale1 = self.reo1(x)       # for PRM-B, C, D
        scale2 = self.reo1(x)       # for PRM-B, C, D
        scale3 = self.reo1(x)       # for PRM-B, C, D
        scale4 = self.reo1(x)       # for PRM-B, C, D
        
        # downsample to multi-scale, scale = pow(2, -M*c/C), where M=1, C=4, c~[0, C]
        ratio1 = F.interpolate(scale1, scale_factor=pow(2, -1))
        ratio2 = F.interpolate(scale2, scale_factor=pow(2, -0.75))
        ratio3 = F.interpolate(scale3, scale_factor=pow(2, -0.5))
        ratio4 = F.interpolate(scale4, scale_factor=pow(2, -0.25))

        # bottle neck convolution
        pyramid1 = F.interpolate(self.ret1(ratio1), size=size)                          # for PRM-A, B, C
        pyramid2 = F.interpolate(self.ret2(ratio2), size=size)                          # for PRM-A, B, C
        pyramid3 = F.interpolate(self.ret3(ratio3), size=size)                          # for PRM-A, B, C
        pyramid4 = F.interpolate(self.ret4(ratio4), size=size)                          # for PRM-A, B, C
        # pyramid1 = self.ret1(scale1)                                                  # for PRM-D
        # pyramid2 = self.ret2(scale2)                                                  # for PRM-D
        # pyramid3 = self.ret3(scale3)                                                  # for PRM-D
        # pyramid4 = self.ret4(scale4)                                                  # for PRM-D
        
        # combine the feature from pyramid
        pyramid_combined = pyramid1 + pyramid2 + pyramid3 + pyramid4                    # for PRM-A, B, D
        # pyramid_combined = torch.cat((pyramid1, pyramid2, pyramid3, pyramid4), 1)     # for PRM-C
        
        # combine the residule part
        smooth = self.smooth(pyramid_combined)
        return identity + smooth


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

class PyraNet(nn.Module):
    def __init__(self, nStack, nModules, nFeats, numOutput):
        super(PyraNet, self).__init__()
        self.nStack = nStack
        self.nModules = nModules
        self.nFeats = nFeats
        self.numOutput = numOutput

        # pyramid hourglass structure
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=2)
        self.prm1 = PRM(64, 64)
        self.ipool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.prm2 = PRM(64, self.nFeats)
        self.relu = nn.ReLU(inplace=True)

        # stacked hourglass                                             # will be removed
        self.relu = nn.ReLU(inplace = True)                             # will be removed
        self.r1 = PRM(64, 128)                                          # will be removed
        # self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)      # will be removed
        self.r4 = PRM(128, 128)                                         # will be removed
        self.r5 = PRM(128, self.nFeats)                                 # will be removed

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
            out.append(tmpOut)
            if i < self.nStack - 1:
                ll_ = self.ll_[i](ll)
                tmpOut_ = self.tmpOut_[i](tmpOut)
                x = x + ll_ + tmpOut_

        return out

