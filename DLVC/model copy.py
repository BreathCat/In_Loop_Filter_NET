import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, nChannels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=nChannels, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
        )

        self.conv2 = nn.Sequential(            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64)
        )
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out1 = self.conv1(x) #/64 channel 输入
        out = self.conv2(out1)
        out = out + out1 
        #print (out.size())
        #print('\n')
        #print(x.size())
        out = torch.cat((x, out), 1)
        out = F.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, ResidualBlock):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        
        self.layer1 = self.make_layer(ResidualBlock,  num_blocks = 3, in_channels=64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer2 = self.make_layer(ResidualBlock,  num_blocks = 3, in_channels=64)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer3 = self.make_layer(ResidualBlock,  num_blocks = 3, in_channels=64)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer4 = self.make_layer(ResidualBlock,  num_blocks = 3, in_channels=64)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.shortcut = nn.Sequential()

    def make_layer(self, block, num_blocks,in_channels): # LZH modify this layer
        #strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in range(num_blocks):
            layers.append(block(nChannels = in_channels)
            in_channels += 64

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out) 
        out = self.conv2(out)
        out = self.layer2(out) 
        out = self.conv3(out)
        out = self.layer3(out) 
        out = self.conv4(out)
        out = self.layer4(out) 
        out = self.conv5(out)
        out = self.conv6(out)
        #out += self.shortcut(x)
        #out = F.relu(out)
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
        print('DLVC正在初始化权重')
        # nn.init.xavier_uniform(m.weight)
        # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        # torch.nn.init.normal_(m.weight, mean=0, std=1)
        # m.bias.data.zero_()

