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
        self.branch1 = BnReluConv1(in_channels=self.in_channels, out_channels=self.out_channels_half)                 # for PRM-B, C, D
        self.branch2 = BnReluConv1(in_channels=self.in_channels, out_channels=self.out_channels_half)                 # for PRM-A
        self.branch3 = BnReluConv1(in_channels=self.in_channels, out_channels=self.out_channels_half)                 # for PRM-A
        self.branch4 = BnReluConv1(in_channels=self.in_channels, out_channels=self.out_channels_half)                 # for PRM-A
        self.branch5 = BnReluConv1(in_channels=self.in_channels, out_channels=self.out_channels)                      # identity branch

        self.bottleneck1 = BnReluConv3(in_channels=self.out_channels_half, out_channels=self.out_channels)            # for PRM-A, B, C
        self.bottleneck2 = BnReluConv3(in_channels=self.out_channels_half, out_channels=self.out_channels)            # for PRM-A, B, C
        self.bottleneck3 = BnReluConv3(in_channels=self.out_channels_half, out_channels=self.out_channels)            # for PRM-A, B, C
        self.bottleneck4 = BnReluConv3(in_channels=self.out_channels_half, out_channels=self.out_channels)            # for PRM-A, B, C
        # self.bottleneck1 = BnReluDilatedConv3(in_channels=self.out_channels_half, out_channels=self.out_channels)   # for PRM-D
        # self.bottleneck2 = BnReluDilatedConv3(in_channels=self.out_channels_half, out_channels=self.out_channels)   # for PRM-D
        # self.bottleneck3 = BnReluDilatedConv3(in_channels=self.out_channels_half, out_channels=self.out_channels)   # for PRM-D
        # self.bottleneck4 = BnReluDilatedConv3(in_channels=self.out_channels_half, out_channels=self.out_channels)   # for PRM-D

        self.smooth = BnReluConv1(in_channels=self.out_channels, out_channels=self.out_channels)                      # for PRM-A, B, D
        # self.smooth = BnReluConv1(in_channels=self.out_channels*4, out_channels=self.out_channels)                  # for PRM-C

    def forward(self, x):
        identity = self.branch5(x)
        size = identity.size()[-2], identity.size()[-1]

        # multi-scale information
        # scale1 = self.branch1(x)     # for PRM-A
        # scale2 = self.branch2(x)     # for PRM-A
        # scale3 = self.branch3(x)     # for PRM-A
        # scale4 = self.branch4(x)     # for PRM-A
        scale1 = self.branch1(x)       # for PRM-B, C, D
        scale2 = self.branch1(x)       # for PRM-B, C, D
        scale3 = self.branch1(x)       # for PRM-B, C, D
        scale4 = self.branch1(x)       # for PRM-B, C, D
        
        # downsample to multi-scale, scale = pow(2, -M*c/C), where M=1, C=4, c~[0, C]
        ratio1 = F.interpolate(scale1, scale_factor=pow(2, -1))
        ratio2 = F.interpolate(scale2, scale_factor=pow(2, -0.75))
        ratio3 = F.interpolate(scale3, scale_factor=pow(2, -0.5))
        ratio4 = F.interpolate(scale4, scale_factor=pow(2, -0.25))

        # bottle neck convolution
        pyramid1 = F.interpolate(self.bottleneck1(ratio1), size=size)                          # for PRM-A, B, C
        pyramid2 = F.interpolate(self.bottleneck2(ratio2), size=size)                          # for PRM-A, B, C
        pyramid3 = F.interpolate(self.bottleneck3(ratio3), size=size)                          # for PRM-A, B, C
        pyramid4 = F.interpolate(self.bottleneck4(ratio4), size=size)                          # for PRM-A, B, C
        # pyramid1 = self.bottleneck1(scale1)                                                  # for PRM-D
        # pyramid2 = self.bottleneck2(scale2)                                                  # for PRM-D
        # pyramid3 = self.bottleneck3(scale3)                                                  # for PRM-D
        # pyramid4 = self.bottleneck4(scale4)                                                  # for PRM-D
        
        # combine the feature from pyramid
        pyramid_combined = pyramid1 + pyramid2 + pyramid3 + pyramid4                           # for PRM-A, B, D
        # pyramid_combined = torch.cat((pyramid1, pyramid2, pyramid3, pyramid4), 1)            # for PRM-C
        
        # combine the residule part
        smooth = self.smooth(pyramid_combined)
        return identity + smooth


class Hourglass(nn.Module):
    def __init__(self, n, nModules, nFeats):
        super(Hourglass, self).__init__()
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats
        
        self.up1_ = nn.ModuleList([PRM(self.nFeats, self.nFeats) for _ in range(nModules)])
        self.down1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.down1_ = nn.ModuleList([PRM(self.nFeats, self.nFeats) for _ in range(nModules)])
        
        # recursive structure as hourglass
        if self.n > 1:
            self.down2 = Hourglass(n - 1, self.nModules, self.nFeats)
        else:
            self.down2_ = nn.ModuleList([PRM(self.nFeats, self.nFeats) for _ in range(nModules)])
        
        self.down3_ = nn.ModuleList([PRM(self.nFeats, self.nFeats) for _ in range(nModules)])
        self.up2 = nn.Upsample(scale_factor = 2)
        
    def forward(self, x):
        up1 = self.traverse_modules(x, self.up1_)
        down1 = self.traverse_modules(self.down1(x), self.down1_)
        if self.n > 1:
            down2 = self.down2(down1)
        else:
            down2 = self.traverse_modules(down1, self.down2_)
        down3 = self.traverse_modules(down2, self.down3_)
        up2 = self.up2(down3)
        
        return up1 + up2
    
    def traverse_modules(self, x, module_list):
        for module in module_list:
            x = module(x)
        return x

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
        
        # backbone structure
        self.hourglass = nn.ModuleList([Hourglass(4, self.nModules, self.nFeats) for _ in range(self.nStack)])
        self.Residual = nn.ModuleList([PRM(self.nFeats, self.nFeats) for _ in range(self.nStack * self.nModules)])
        self.lin_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1),
                nn.BatchNorm2d(self.nFeats),
                nn.ReLU(inplace=True)
            ) for _ in range(self.nStack)
        ])
        self.tmpOut = nn.ModuleList([nn.Conv2d(self.nFeats, self.numOutput, bias=True, kernel_size=1, stride=1) for _ in range(self.nStack)])
        self.ll_ = nn.ModuleList([nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1) for _ in range(self.nStack - 1)])
        self.tmpOut_ = nn.ModuleList([nn.Conv2d(self.numOutput, self.nFeats, bias=True, kernel_size=1, stride=1) for _ in range(self.nStack - 1)])

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

