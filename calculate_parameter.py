# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchsummary
from torch.nn import init
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
           
        )
    def forward(self, x):
        out1 = self.conv1(x)
        out = out1 + x
        out2 = self.conv2(out)
        out = out2 + x
        out3 = self.conv3(out)
        out = out3 + x
        out = torch.cat((out, out1), 1)
        out = torch.cat((out, out2), 1)
        out = torch.cat((out, out3), 1)
        out = self.conv4(out)
        out += x
        return out

class Net(nn.Module):
    def __init__(self, ResidualBlock):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock,  num_blocks=6)
       
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.shortcut = nn.Sequential()

    def make_layer(self, res_block, num_blocks):
        num_blocks = [1] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for countnumber in num_blocks:
            layers.append(res_block())

        return nn.Sequential(*layers)

   
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.conv2(out)
        # out += self.shortcut(x)
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
        print('Model7正在初始化权重')
        # nn.init.xavier_uniform(m.weight)
        # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        # torch.nn.init.normal_(m.weight, mean=0, std=1)
        # m.bias.data.zero_()
# class ResidualBlock(nn.Module):
#     def __init__(self, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=stride, padding=1, bias=True),
#             #nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
#             #nn.BatchNorm2d(64)
#         )
#         self.shortcut = nn.Sequential()

#     def forward(self, x):
#         out = self.conv(x)
#         out += x
#         out = F.relu(out)
#         return out

# class Net(nn.Module):
#     def __init__(self, ResidualBlock):
#         super(Net, self).__init__()

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
#             #nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
#         self.layer1 = self.make_layer(ResidualBlock,  num_blocks=16, stride=1)
#         # self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
#         # self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
#         # self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
#             #nn.BatchNorm2d(1),
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






# def normal_init(m, mean, std):
#     if  isinstance(m, nn.Conv2d):
#         # m.weight.data.normal_(mean, std)
#         # m.init.xavier_uniform_(m.weight, gain=1)
#         # nn.init.xavier_uniform(m.weight)
#         # m.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
#         # m.bias.data.zero_()
#         print('DLVC正在初始化权重')
#         # nn.init.xavier_uniform(m.weight)
#         # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
#         torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
#         # torch.nn.init.normal_(m.weight, mean=0, std=1)
#         # m.bias.data.zero_()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0


model = Net(ResidualBlock).to(device)
torchsummary.summary(model,(1, 28, 28))
print('parameters_count:',count_parameters(model))
