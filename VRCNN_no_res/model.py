'''

这是第一次训练成功的model,VRCNN没有残差,没有BatchNorm2d
使用Adam,学习率0.001

'''


import torch
import torch.nn as nn


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Sequential(
            #第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )

        self.conv2 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

        )

        self.conv3 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )

        # self.bn_1 = torch.nn.BatchNorm2d(48)

        self.conv4 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

        )

        self.conv5 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )

        # self.bn_2 = torch.nn.BatchNorm2d(48)

        self.conv6 = torch.nn.Sequential(
            # 第一层in_channel=1,kenel_size=9,out_channel=64
            nn.Conv2d(in_channels=48, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),

        )


    def forward(self, x):

        out = self.conv1(x)
        out = torch.cat((self.conv2(out), self.conv3(out)), 1)
        out = torch.cat((self.conv4(out), self.conv5(out)), 1)
        out = self.conv6(out)

        return out



    def weight_init(self, mean, std):
        # for i in self.children():
        #     print('i:',i)
        # 访问 modules
        for m in self.modules():
            # print('m:',m)
            normal_init(m, mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        # m.weight.data.normal_(mean, std)
        # m.init.xavier_uniform_(m.weight, gain=1)
        # nn.init.xavier_uniform(m.weight)
        # m.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        # m.bias.data.zero_()
        print('VRCNN_no_res 正在初始化权重')
        # nn.init.xavier_uniform(m.weight)
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        # torch.nn.init.normal_(m.weight, mean=0, std=1)

        m.bias.data.zero_()


#
# x=torch.ones(64,1,5,5)
#
# net=Net()
# out=net(x)
# # print(out)
# print(net)
# # print('out:',out)
# print(out.shape)
# print(out)
