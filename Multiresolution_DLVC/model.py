#-*-coding:utf-8-*-
'''

Multiresolution_DLVC

'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class Group2(nn.Module):
    def __init__(self, stride=1):
        super(Group2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64)
        )
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out






class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        ###########           Group1
        self.scale1_group1 = torch.nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )

        self.scale2_group1 = torch.nn.Sequential(

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )

        self.scale3_group1 = torch.nn.Sequential(

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )

        self.scale4_group1 = torch.nn.Sequential(

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )

        self.scale5_group1 = torch.nn.Sequential(

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )

        ###########           Group2

        self.group2 = self.make_layer(Group2, num_blocks=10, stride=1)


        ###########           Group3
        self.scale1_group3 = torch.nn.Sequential(

            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),

        )

        self.scale2_group3 = torch.nn.Sequential(

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),

        )

        self.scale3_group3 = torch.nn.Sequential(

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),

        )

        self.scale4_group3 = torch.nn.Sequential(

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),

        )

        self.scale5_group3 = torch.nn.Sequential(

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),

        )

        self.shortcut = nn.Sequential()






    def make_layer(self, block, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(stride))
        return nn.Sequential(*layers)



    def forward(self, x):

        out1 = self.scale1_group1(x)
        # print('out1:',out1.shape)
        out2 = self.scale2_group1(out1)
        # print('out2:',out2.shape)
        out3 = self.scale3_group1(out2)
        # print('out3:',out3.shape)
        out4 = self.scale4_group1(out3)
        # print('out4:',out4.shape)
        out5 = self.scale5_group1(out4)
        # print('out5:',out5.shape)

        out1 = self.group2(out1)
        # print('out1:', out1.shape)
        out2 = self.group2(out2)
        # print('out2:', out2.shape)
        out3 = self.group2(out3)
        # print('out3:', out3.shape)
        out4 = self.group2(out4)
        # print('out4:', out4.shape)
        out5 = self.group2(out5)
        # print('out5:', out5.shape)


        out1 = self.scale1_group3(out1)
        # print('out1:', out1.shape)
        # print('out1:', out1[0][0][0][0])
        out2 = self.scale2_group3(out2)
        # print('out2:', out2.shape)
        # print('out2:', out2[0][0][0][0])
        out3 = self.scale3_group3(out3)
        # print('out3:', out3.shape)
        # print('out3:', out3[0][0][0][0])
        out4 = self.scale4_group3(out4)
        # print('out4:', out4.shape)
        # print('out4:', out4[0][0][0][0])
        out5 = self.scale5_group3(out5)
        # print('out5:', out5.shape)
        # print('out5:', out5[0][0][0][0])



        out=out1+out2+out3+out4+out5
        # print('out:', out.shape)
        # print('out:', out[0][0][0][0])

        out += self.shortcut(x)
        out = F.relu(out)

        return out



    def weight_init(self, mean, std):
        # for i in self.children():
        #     print('i:',i)
        # 访问 modules
        for m in self.modules():
            pass
            # print('m:',m)
            # normal_init(m, mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        # m.weight.data.normal_(mean, std)
        # m.init.xavier_uniform_(m.weight, gain=1)
        # nn.init.xavier_uniform(m.weight)
        # m.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        # m.bias.data.zero_()
        print(' MULTI 正在初始化权重')
        # nn.init.xavier_uniform(m.weight)
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        # torch.nn.init.normal_(m.weight, mean=0, std=1)

        m.bias.data.zero_()



# x=torch.ones(64,1,128,128)
#
# net=Net()
# out=net(x)
# print(out)
# print(net)
# print('out:',out)
# print(out.shape)
# print(out)
