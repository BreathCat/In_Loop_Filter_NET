import argparse
import time
import os
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np



model_dir='./epochs'

list_need_retain=[]
for epoch in range(1,161):
    if epoch % 20 == 0 and epoch != 0:
        G_name = 'netG_epoch_1_' + str(epoch) + '.pth'
        G_model_dir = os.path.join(model_dir,G_name)
        # print('test_model_dir:',G_model_dir)
        list_need_retain.append(G_model_dir)

        D_name = 'netD_epoch_1_' + str(epoch) + '.pth'
        D_model_dir = os.path.join(model_dir,D_name)
        print('test_model_dir:',D_model_dir)
        list_need_retain.append(D_model_dir)
# print(len(list_need_retain))


list_all_file=[]
for file in os.listdir(model_dir):
    full_file_name = os.path.join(model_dir, file)
    # print('file:',full_file_name)
    list_all_file.append(full_file_name)
print(len(list_all_file))

list_need_remove=list(set(list_all_file)-set(list_need_retain))
# print('list_need_remove:',len(list_need_remove))

for i in range(len(list_need_remove)):
    # print(list_need_remove[i])
    os.remove(list_need_remove[i])


# list_need_remove=['./epochs/netD_epoch_1_3.pth','./epochs/netD_epoch_1_4.pth']


# list1=['1','2','3']
# list2=['1']
# list3=list(set(list1)-set(list2))
# print(list3)