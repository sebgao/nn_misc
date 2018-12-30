import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, 
    out_planes,
    kernel_size=3,
    stride=stride,
    padding=dilation,
    dilation=dilation,
    bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        medium = max(inplanes, planes)//2
        self.conv1 = conv1x1(inplanes, medium)
        self.bn1 = nn.InstanceNorm2d(medium)
        self.conv2 = conv3x3(medium, medium, stride=stride)
        self.bn2 = nn.InstanceNorm2d(medium)
        self.conv3 = conv1x1(medium, planes)
        self.bn3 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if stride == 2 or inplanes != planes:
            self.transition = conv3x3(inplanes, planes, stride=stride)
        else:
            self.transition = None
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)

        if self.transition is not None:
            residual = self.transition(residual)
        
        x += residual
        x = self.bn3(x)
        x = self.relu(x)

        return x

class ResAutoEncoder(nn.Module):
    def __init__(self):
        super(ResAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            BasicBlock(3, 32, stride=2),
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64),
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
            conv1x1(128, 128),
            nn.InstanceNorm2d(128),
        )

        self.decoder = nn.Sequential(
            BasicBlock(128, 64),
            #nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=2),
            BasicBlock(64, 32),
            BasicBlock(32, 32),
            nn.Upsample(scale_factor=2),
            BasicBlock(32, 32),
            BasicBlock(32, 32),
            nn.Upsample(scale_factor=2),
            BasicBlock(32, 32),
            BasicBlock(32, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    x = torch.randn((1, 3, 64, 64))
    net = ResAutoEncoder()

    print(net(x).shape)
