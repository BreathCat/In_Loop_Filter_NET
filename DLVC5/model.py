import torch
import torch.nn as nn
import torch.nn.functional as F

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

class  Small_ResidualBlockclass(nn.Module):
    def __init__(self,inchannel):
        super(Small_ResidualBlockclass, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1)
        )
        
    def forward(self, x):
        out = self.conv(x)
        out = torch.cat((x, out), 1)
        return out

class Net(nn.Module):
    def __init__(self, Small_ResidualBlockclass):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = Small_ResidualBlockclass(2)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv3 = Small_ResidualBlockclass(3)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv4 = Small_ResidualBlockclass(4)
        self.relu3 = nn.ReLU(inplace=False)
        self.conv5 = Small_ResidualBlockclass(5)
        self.relu4 = nn.ReLU(inplace=False)
        self.conv6 = Small_ResidualBlockclass(6)
        self.relu5 = nn.ReLU(inplace=False)
        self.conv7 = Small_ResidualBlockclass(7)
        self.relu6 = nn.ReLU(inplace=False)
        self.conv8 = Small_ResidualBlockclass(8)
        self.relu7 = nn.ReLU(inplace=False)
        self.conv9 = Small_ResidualBlockclass(9)
        self.relu8 = nn.ReLU(inplace=False)
        self.conv10 = Small_ResidualBlockclass(10)
        self.relu9 = nn.ReLU(inplace=False)
        self.conv11 = Small_ResidualBlockclass(11)
        self.relu10 = nn.ReLU(inplace=False)
        self.conv12 = Small_ResidualBlockclass(12)
        self.relu11 = nn.ReLU(inplace=False)
        self.conv13 = Small_ResidualBlockclass(13)
        self.relu12 = nn.ReLU(inplace=False)
        self.conv14 = Small_ResidualBlockclass(14)
        self.relu13 = nn.ReLU(inplace=False)
        self.conv15 = Small_ResidualBlockclass(15)
        self.relu14 = nn.ReLU(inplace=False)
        self.conv16 = Small_ResidualBlockclass(16)
        self.relu15 = nn.ReLU(inplace=False)
        self.conv17 = Small_ResidualBlockclass(17)
        self.relu16 = nn.ReLU(inplace=False)

        self.conv18 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=False),
        )
        self.relu17 = nn.ReLU(inplace=False)

        self.shortcut = nn.Sequential()


    def forward(self, x):
        out = self.conv1(x)
        out = torch.cat((x, out), 1)
        out = self.conv2(out)
        out += x
        out = F.relu(out)
        out = self.conv3(out)
        out += x
        out = F.relu(out)
        out = self.conv4(out)
        out += x
        out = F.relu(out)
        out = self.conv5(out)
        out += x
        out = F.relu(out)
        out = self.conv6(out)
        out += x
        out = F.relu(out)
        out = self.conv7(out)
        out += x
        out = F.relu(out)
        out = self.conv8(out)
        out += x
        out = F.relu(out)
        out = self.conv9(out)
        out += x
        out = F.relu(out)
        out = self.conv10(out)
        out += x
        out = F.relu(out)
        out = self.conv11(out)
        out += x
        out = F.relu(out)
        out = self.conv12(out)
        out += x
        out = F.relu(out)
        out = self.conv13(out)
        out += x
        out = F.relu(out)
        out = self.conv14(out)
        out += x
        out = F.relu(out)
        out = self.conv15(out)
        out += x
        out = F.relu(out)
        out = self.conv16(out)
        out += x
        out = F.relu(out)
        out = self.conv17(out)
        out += x
        out = F.relu(out)

        out = self.conv18(out)
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
        print('DLVC正在初始化权重')
        # nn.init.xavier_uniform(m.weight)
        # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        # torch.nn.init.normal_(m.weight, mean=0, std=1)
        # m.bias.data.zero_()