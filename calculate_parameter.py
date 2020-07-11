# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchsummary
from torch.nn import init
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
        out = F.relu(out, inplace=True)
        return out

class Net(nn.Module):
    def __init__(self, ResidualBlock):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        
        self.layer1 = self.make_layer(ResidualBlock,  num_blocks = 3, in_channels=64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer2 = self.make_layer(ResidualBlock,  num_blocks = 3, in_channels=64)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer3 = self.make_layer(ResidualBlock,  num_blocks = 3, in_channels=64)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer4 = self.make_layer(ResidualBlock,  num_blocks = 3, in_channels=64)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.shortcut = nn.Sequential()

    def make_layer(self, block, num_blocks,in_channels): # LZH modify this layer
        #strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in range(num_blocks):
            layers.append(block(nChannels = in_channels))
            in_channels=in_channels+64

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



# class ResidualBlock(nn.Module):
#     def __init__(self, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=stride, padding=1, bias=True),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(64)
#         )
#         self.shortcut = nn.Sequential()

#     def forward(self, x):
#         out = self.conv(x)
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# class Net(nn.Module):
#     def __init__(self, ResidualBlock):
#         super(Net, self).__init__()

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
#         self.layer1 = self.make_layer(ResidualBlock,  num_blocks=16, stride=1)
#         # self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
#         # self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
#         # self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(1),
#             nn.ReLU(),
#         )

#         self.shortcut = nn.Sequential()

#     def make_layer(self, block, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
#         layers = []
#         for stride in strides:
#             layers.append(block(stride))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.layer1(out)
#         out = self.conv2(out)
#         # out += self.shortcut(x)
#         out = F.relu(out)
#         return out

#     def weight_init(self, mean, std):
#         # for i in self.children():
#         #     print('i:',i)
#         # 访问 modules
#         for m in self.modules():

#             # print('m:',m)
#             normal_init(m, mean, std)


class ResidualBlock(nn.Module):
    def __init__(self, nChannels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=nChannels, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64)
        )


    def forward(self, x):
        out1 = self.conv1(x) #/64 channel 输入
        out = self.conv2(out1)
        out = out + out1 
        out = torch.cat((x, out), 1)
        out = F.relu(out, inplace=True)
        return out

class ResidualBlock_0(nn.Module):
    def __init__(self, nChannels):
        super(ResidualBlock_0, self).__init__()

        self.conv = nn.Sequential(            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64)
        )

    def forward(self, x):
        out = self.conv(x) 
        out = torch.cat((x, out), 1)
        out = F.relu(out, inplace=True)
        return out

class Net(nn.Module):
    def __init__(self, ResidualBlock):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        
        self.layer1 = self.make_layer(ResidualBlock,ResidualBlock_0,  num_blocks = 3, in_channels=64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer2 = self.make_layer(ResidualBlock, ResidualBlock_0, num_blocks = 3, in_channels=64)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer3 = self.make_layer(ResidualBlock, ResidualBlock_0,  num_blocks = 3, in_channels=64)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer4 = self.make_layer(ResidualBlock, ResidualBlock_0,  num_blocks = 3, in_channels=64)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True),
        )

        self.shortcut = nn.Sequential()

    def make_layer(self, block,block_0, num_blocks,in_channels): # LZH modify this layer
        #strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        layers.append(block_0(nChannels = in_channels))
        in_channels=in_channels+64

        for stride in range(num_blocks-1):
            layers.append(block(nChannels = in_channels))
            in_channels=in_channels+64

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
        out += x
        # out = F.relu(out)
        return out

    def weight_init(self, mean, std):
        # for i in self.children():
        #     print('i:',i)
        # 访问 modules
        for m in self.modules():

            # print('m:',m)
            normal_init(m, mean, std)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0


model = Net(ResidualBlock).to(device)
torchsummary.summary(model,(1, 28, 28))
print('parameters_count:',count_parameters(model))
