import torch
import torch.nn as nn
import torch.nn.functional as F
from simple_resnet import ResAutoEncoder, conv3x3, conv1x1

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            conv3x3(3, 64, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            conv3x3(64, 64, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            conv3x3(64, 64, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            conv3x3(64, 64, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            conv3x3(64, 64),
            nn.AdaptiveAvgPool2d((1, 1)),
            conv1x1(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x).view(-1)

class DCGAN(nn.Module):
    def __init__(self, device=torch.device('cpu'), lr_G = 1e-3, lr_D = 5*1e-4):
        super(DCGAN, self).__init__()
        self.generator = ResAutoEncoder()
        self.discriminator = Discriminator()
        self.device = device
        self.to(device)
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr_G)
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr_D)
        self.bceloss = nn.BCELoss()

    def flip_grad(self):
        for p in self.generator.parameters():
            if p.requires_grad:
                p.grad.data.mul(-1.0)

    def identity_train(self, real_sample):
        self.train()
        self.generator.zero_grad()
        real = self.generator(real_sample)
        err_mse = F.mse_loss(real, real_sample)
        err_mse.backward()
        G_x = err_mse.item()
        self.optimizerG.step()
        return G_x

    def ad_train(self, real_sample, fake_sample):
        self.train()

        real_B = real_sample.shape[0]
        real_label = torch.full((real_B, ), 1, device=self.device)

        fake_B = fake_sample.shape[0]
        fake_label = torch.full((fake_B, ), 1, device=self.device)

        self.discriminator.zero_grad()
        self.generator.zero_grad()

        output = self.discriminator(real_sample)
        err_real = 0.005*self.bceloss(output, real_label)
        err_real.backward()
        D_r = err_real.mean().item()

        fake = self.generator(fake_sample)
        output = self.discriminator(fake).view(-1)
        err_fake = 0.008*self.bceloss(output, fake_label)
        err_fake.backward()
        D_f = err_fake.mean().item()

        self.flip_grad()

        real = self.generator(real_sample)
        err_mse = F.mse_loss(real, real_sample)
        err_mse.backward()
        G_x = err_mse.item()

        self.optimizerD.step()
        self.optimizerG.step()
        return D_r, D_f, G_x
        
        
    def transform(self, x):
        self.eval()
        return self.generator(x)
        
    def forward(self, x):
        return self.discriminator(self.generator(x))

if __name__ == "__main__":
    X = torch.randn((6, 3, 64, 64))
    gan = DCGAN()
    print(gan(X).shape)
    gan.ad_train(X, X)