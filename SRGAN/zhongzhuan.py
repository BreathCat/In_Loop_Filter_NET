#-*-coding:utf-8-*-


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
