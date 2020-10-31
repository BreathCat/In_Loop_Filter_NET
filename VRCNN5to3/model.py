import math
import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        self.conv1 = torch.nn.Sequential(
            #第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )

        self.conv2 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

        )

        self.conv3 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )



        self.conv4 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

        )

        self.conv5 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )



        self.conv6 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=48, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True),

        )

        # self.conv7 = torch.nn.Sequential(
        #     第一层in_channel=1,kenel_size=9,out_channel=64
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),

        # )

        self.shortcut = nn.Sequential()


    def forward(self, x):

        out = self.conv1(x)
        out = torch.cat((self.conv2(out), self.conv3(out)), 1)
        out = torch.cat((self.conv4(out), self.conv5(out)), 1)
        out = self.conv6(out)
        # out = self.conv7(out)
        out = out + self.shortcut(x)
        out = F.relu(out)


        return out

    def weight_init(self, mean, std):
        # for i in self.children():
        #     print('i:',i)
        # 访问 modules
        for m in self.modules():

            # print('m:',m)
            normal_init(m, mean, std)






def normal_init(m, mean, std):
    if  isinstance(m, nn.Conv2d):
        # m.weight.data.normal_(mean, std)
        # m.init.xavier_uniform_(m.weight, gain=1)
        # nn.init.xavier_uniform(m.weight)
        # m.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        # m.bias.data.zero_()
        print('VRCNN5to3正在初始化权重')
        # nn.init.xavier_uniform(m.weight)
        # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        # torch.nn.init.normal_(m.weight, mean=0, std=1)
        # m.bias.data.zero_()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
