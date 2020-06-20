import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * F.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel, out_channels, stride):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x


class UpsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return swish(self.shuffler(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor, num_channel=1, base_filter=64):
        super(Generator, self).__init__()

        self.conv1 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )

        self.conv2 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True),
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

    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)



class Discriminator(nn.Module):
    def __init__(self, num_channel=1, base_filter=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, base_filter, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(base_filter, base_filter, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filter)
        self.conv3 = nn.Conv2d(base_filter, base_filter * 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filter * 2)
        self.conv4 = nn.Conv2d(base_filter * 2, base_filter * 2, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filter * 2)
        self.conv5 = nn.Conv2d(base_filter * 2, base_filter * 4, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(base_filter * 4)
        self.conv6 = nn.Conv2d(base_filter * 4, base_filter * 4, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(base_filter * 4)
        self.conv7 = nn.Conv2d(base_filter * 4, base_filter * 8, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(base_filter * 8)
        self.conv8 = nn.Conv2d(base_filter * 8, base_filter * 8, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(base_filter * 8)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(base_filter * 8, num_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = swish(self.conv1(x))

        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = swish(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)

    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
