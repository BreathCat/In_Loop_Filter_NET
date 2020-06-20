'''

这是VRCNN_ext

'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class VR_Block(nn.Module):
    def __init__(self, stride=1):
        super(VR_Block, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=stride, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),

        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=stride, padding=0, bias=True),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),

        )

        self.shortcut = nn.Sequential()


    def forward(self, x):


        out = torch.cat((self.conv1_1(x), self.conv1_2(x)), 1)
        out = torch.cat((self.conv2_1(out), self.conv2_2(out)), 1)

        out += self.shortcut(x)
        out = F.relu(out)


        return out

class Net(nn.Module):
    def __init__(self, ResidualBlock):
        super(Net, self).__init__()
        # self.inchannel = 64

        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )

        self.layer1 = self.make_layer(ResidualBlock,  num_blocks=10, stride=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            # nn.ReLU(),
        )


        self.shortcut = nn.Sequential()

    def make_layer(self, block, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.conv2(out)


        out += self.shortcut(x)
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
        print('VRCNN_ext  正在初始化权重')
        # nn.init.xavier_uniform(m.weight)
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        # torch.nn.init.normal_(m.weight, mean=0, std=1)
        m.bias.data.zero_()


#
# x=torch.ones(64,1,5,5)
#
# net=Net(VR_Block)
# out=net(x)
# print(out)
# print('net:',net)
# print('out:',out)
# print('out.shape:',out.shape)
# print(out)