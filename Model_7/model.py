import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(64)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
           
        )
    def forward(self, x):
        out = self.conv1(x)
        out1 = out + x
        out = out +x
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
        self.layer1 = self.make_layer(ResidualBlock,  num_blocks=12)
       
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
        nn.init.xavier_uniform(m.weight)
        # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        # torch.nn.init.normal_(m.weight, mean=0, std=1)
        # m.bias.data.zero_()