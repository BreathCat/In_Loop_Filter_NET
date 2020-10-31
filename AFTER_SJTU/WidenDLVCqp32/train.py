import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import ResidualBlock, Net # model change lzh

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=22, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=1, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=160, type=int, help='train epoch number')
parser.add_argument('--resume', default=0, type=int, help='whether load existing model parameters')
parser.add_argument('--widen_factor', default = 4, type=int,help='for Wideres Net')
if __name__ == '__main__':
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    RESUME = opt.resume
    WIDEN_FACTOR = opt.widen_factor
    start_epoch = 1

    # train_set = TrainDatasetFromFolder('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    # val_set = ValDatasetFromFolder('data/DIV2K_valid_HR', upscale_factor=UPSCALE_FACTOR)
    # train_path = '/home/lzh412/Desktop/dataset/Images_for_train/832×480_sub_128×128/train/subim_input'
    # train_path = '/home/lzh412/Desktop/dataset/Images_for_train/DIV2K_input_label/train/subim_input'  # QP=38的数据集
    # train_path = '/home/lzh412/Desktop/dataset/Images_for_train/DIV2K_input_label_QP42/train/subim_input'   # QP=42的数据集
    train_path = '/home/li/Desktop/LZH/CNN_NET/dataset/2020_9/DIV2K_input_label_QP32/train/subim_input'  # QP=32的数据集
    train_set = TrainDatasetFromFolder(train_path, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    # train_set = TrainDatasetFromFolder('data/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('data/val/subim_input', upscale_factor=UPSCALE_FACTOR)

    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    netG = Net(ResidualBlock)# model change lzh
    netG.load_state_dict(torch.load('/home/li/Desktop/LZH/CopyCNN/WidenDLVCqp32/yuxunlian38/DLVC_epoch_1_160.pth'))
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    # netD = Discriminator()
    # print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    generator_criterion = GeneratorLoss()
    # mse_loss = torch.nn.MSELoss()
    print('torch.cuda.is_available() = ',torch.cuda.is_available())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    netG.to(device)
    # 用预训练好的qp=38对网络进行初始化
    
    # netG.load_state_dict(torch.load('/home/li/Desktop/LZH/CopyCNN/Wideres_QP32/yuxunlian38/QP37_Wide_Res_state_dict.pth'))
    # netD.to(device)
    generator_criterion.to(device)

    ''' 
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
    '''

    optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(0.9, 0.999))
    # optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.9, 0.999))
    SchedulerG = optim.lr_scheduler.MultiStepLR(optimizerG,
                    milestones=[40, 80, 120], gamma=0.1)
        

    results = { 'g_loss': [], 'psnr': [], 'ssim': []}
    # Pretrained parameters
    if RESUME:
        path_checkpoint = "Checkpoint/ckpt_best_1.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        netG.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizerG.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        SchedulerG.load_state_dict(checkpoint['lr_schedule'])
    # Start epoch
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'g_loss': 0}

        netG.train()
     
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            
            real_img = Variable(target)
            
            real_img = real_img.to(device)

            z = Variable(data)  
            z = z.to(device)         
            fake_img = netG(z)
            
            # 为了与VGG的输入channel对应, 这里需要把只有一个channel的real_img和fake_img变成channel=3
            real_img_channel_is_3 = torch.cat((real_img, real_img, real_img), 1)
            fake_img_channel_is_3 = torch.cat((fake_img, fake_img, fake_img), 1)
            
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            g_loss = generator_criterion(fake_img_channel_is_3, real_img_channel_is_3)
            g_loss.backward()

            
            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            

            train_bar.set_description(desc='[%d/%d] Loss_G: %.6f' % (
                epoch, NUM_EPOCHS,
                running_results['g_loss'] / running_results['batch_sizes']))

        SchedulerG.step(epoch)
        
        # save model parameters
        torch.save(netG.state_dict(), 'epochs/Widen_DLVC_32_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))# model change lzh
        checkpoint = {
            "net": netG.state_dict(),
            'optimizer': optimizerG.state_dict(),
            "epoch": epoch,
            'lr_schedule': SchedulerG.state_dict()
        }
        if not os.path.isdir("./Checkpoint"):
            os.mkdir("./Checkpoint")
        torch.save(checkpoint, './Checkpoint/Widen_DLVC_32_best_%s.pth' % (str(epoch)))# model change lzh

        # save loss_G
        with open('statistics/train_result_lzh.csv','a') as f: # 'a'表示append,即在原来文件内容后继续写数据（不清楚原有数据）
            f.write('[%d/%d] Loss_G: %.6f\n' % (
                epoch, NUM_EPOCHS,
                running_results['g_loss'] / running_results['batch_sizes']))

            f.write(" lr= %.6f  " %(SchedulerG.get_lr()[0]))

