%%writefile xgan.py
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
        medium = max(inplanes, planes)
        self.conv1 = conv1x1(inplanes, medium)
        self.bn1 = nn.BatchNorm2d(medium)
        self.conv2 = conv3x3(medium, medium, stride=stride)
        self.bn2 = nn.BatchNorm2d(medium)
        self.conv3 = conv1x1(medium, planes)
        self.bn3 = nn.BatchNorm2d(planes)
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



class Discriminator(nn.Module):
    def __init__(self, layer=4):
        super(Discriminator, self).__init__()
        main = []
        main += [BasicBlock(64, 64, stride=2), ] * (layer - 1)
        main += [conv3x3(64, 64, stride=2), nn.LeakyReLU(0.1, inplace=True), ]
        main += [conv1x1(64, 1), nn.Sigmoid()]
        self.main = nn.Sequential(
            *main
        )
    def forward(self, x):
        return self.main(x).view(-1)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            BasicBlock(3, 32, stride=2),
            BasicBlock(32, 32, stride=2),
            BasicBlock(32, 64),
            BasicBlock(64, 64),
            conv1x1(64, 64),
            nn.InstanceNorm2d(64),
        )
    
    def forward(self, x):
        return self.main(x)
    
def upsample(c=64):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        conv3x3(c, c),
        nn.BatchNorm2d(c),
        nn.LeakyReLU(0.1, inplace=True),
    )
  
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            #nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            BasicBlock(64, 64),
            upsample(c=64),
            BasicBlock(64, 32),
            upsample(c=32),
            BasicBlock(32, 32),
            conv1x1(32, 3),
            nn.Tanh(),
        )
    
    def forward(self, x):
        return self.main(x)

class XGAN(nn.Module):
    
    def __init__(self, device=torch.device('cpu'), lr_G = 2*1e-4, lr_D = 2*1e-4):
        super(XGAN, self).__init__()
        self.encoders = nn.ModuleList([Encoder(), Encoder()])
        self.decoders = nn.ModuleList([Decoder(), Decoder()])

        self.bceloss = nn.BCELoss()
        self.discriminator = Discriminator()

        self.ttD = Discriminator(layer=7)

        self.optimizerECD = torch.optim.Adam(self.encoders.parameters(), lr_G, betas=(0.5, 0.999))
        self.optimizerDCD = torch.optim.Adam(self.decoders.parameters(), lr_G, betas=(0.5, 0.999))
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr_D, betas=(0.5, 0.999))

        self.optimizerttD = torch.optim.Adam(self.ttD.parameters(), lr_D, betas=(0.5, 0.999))

        self.path = [0, 0]
        self.device = device
        self.to(device)
        self.coeff = [0.05, 0.30, 1.0, 0.2]

    def ad_train(self, real_sample, fake_sample):
        self.train()

        real_B = real_sample.shape[0]
        real_label = torch.full((real_B, ), 1, device=self.device)

        fake_B = fake_sample.shape[0]
        fake_label = torch.full((fake_B, ), 0, device=self.device)

        #discriminator update#

        self.optimizerD.zero_grad()
        
        fake = self.encoders[0](fake_sample)
        err_f = self.coeff[0]*self.bceloss(self.discriminator(fake.detach()), fake_label)
        err_f.backward()
        D_f = err_f.mean().item()

        real = self.encoders[1](real_sample)
        err_r = self.coeff[0]*self.bceloss(self.discriminator(real.detach()), real_label)
        err_r.backward()
        D_r = err_r.mean().item()

        self.optimizerD.step()

        self.optimizerECD.zero_grad()
        self.optimizerDCD.zero_grad()

        fake = self.encoders[0](fake_sample)
        err_f = self.coeff[1]*self.bceloss(self.discriminator(fake), fake_label.fill_(1))
        err_f.backward()
        E_f = err_f.mean().item()

        self.switch_path(0, 0)
        err_f = self.coeff[2]*F.mse_loss(self.forward_inner(fake_sample), fake_sample)
        
        self.switch_path(1, 1)
        err_f += self.coeff[2]*F.mse_loss(self.forward_inner(real_sample), real_sample)

        self.switch_path(0, 1)
        real = self.forward_inner(fake_sample)
        self.switch_path(1, 0)
        err_f += self.coeff[3]*F.mse_loss(self.forward_inner(real), fake_sample)

        self.switch_path(1, 0)
        fake = self.forward_inner(real_sample)
        self.switch_path(0, 1)
        err_f += self.coeff[3]*F.mse_loss(self.forward_inner(fake), real_sample)

        err_f.backward()

        E_r = err_f.item()

        self.optimizerECD.step()
        self.optimizerDCD.step()

        return D_f+D_r, E_f, E_r


    def switch_path(self, i, j):
        self.path = [i, j]

    def forward_inner(self, x):
        latent = self.encoders[self.path[0]](x)
        restore = self.decoders[self.path[1]](latent)
        return restore

    def forward(self, x):
        x = self.encoders[0](x)
        return self.discriminator(x)

if __name__ == "__main__":
    x = torch.randn((1, 3, 64, 64))
    decoder = XGAN()
    print(decoder.ad_train(x, x))