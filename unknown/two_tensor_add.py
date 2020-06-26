# -*-coding:utf-8-*-
from torch import nn
import torch

m = nn.BatchNorm2d(48)  # bn设置的参数实际上是channel的参数
input = torch.randn(64, 48, 35, 35)
output = m(input)
# print(output)
# a = (input[0, 0, :, :]+input[1, 0, :, :]+input[2, 0, :, :]+input[3, 0, :, :]).sum()/16
# b = (input[0, 1, :, :]+input[1, 1, :, :]+input[2, 1, :, :]+input[3, 1, :, :]).sum()/16
# c = (input[0, 2, :, :]+input[1, 2, :, :]+input[2, 2, :, :]+input[3, 2, :, :]).sum()/16
# print('The mean value of the first channel is %f' % a.data)
# print('The mean value of the first channel is %f' % b.data)
# print('The mean value of the first channel is %f' % c.data)
# print('The output mean value of the BN layer is %f, %f, %f' % (m.running_mean.data[0],m.running_mean.data[0],m.running_mean.data[0]))
print(m)
print(output)
# print(output.shape)
