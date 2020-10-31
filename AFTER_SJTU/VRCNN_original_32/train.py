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
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=22, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=1, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=160, type=int, help='train epoch number')
parser.add_argument('--resume', default=0, type=int, help='whether load existing model parameters')

if __name__ == '__main__':
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    RESUME = opt.resume
    start_epoch = 1

    train_path = '/home/li/Desktop/LZH/CNN_NET/dataset/2020_9/DIV2K_input_label_QP32/train/subim_input'  # QP=32的数据集
    train_set = TrainDatasetFromFolder(train_path, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
  
    val_set = ValDatasetFromFolder('data/val/subim_input', upscale_factor=UPSCALE_FACTOR)

    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR)
    netG.load_state_dict(torch.load('/home/li/Desktop/LZH/CopyCNN/VRCNN_original_QP32/yuxunlian38/netG_epoch_1_160.pth'))
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
   
    generator_criterion = GeneratorLoss()
    # mse_loss = torch.nn.MSELoss()
    print('torch.cuda.is_available() = ',torch.cuda.is_available())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    netG.to(device)
    # 用预训练好的qp=38对网络进行初始化
    # netG.load_state_dict(torch.load('/home/li/Desktop/LZH/CopyCNN/VRCNN_5To3_QP32/yuxunlian32/netG_epoch_1_160.pth'))
  
    generator_criterion.to(device)

    ''' 
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
    '''

    optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.9, 0.999))
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
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        checkpoint = {
            "net": netG.state_dict(),
            'optimizer': optimizerG.state_dict(),
            "epoch": epoch,
            'lr_schedule': SchedulerG.state_dict()
        }
        if not os.path.isdir("./Checkpoint"):
            os.mkdir("./Checkpoint")
        torch.save(checkpoint, './Checkpoint/ckpt_best_%s.pth' % (str(epoch)))

        # save loss_G
        with open('statistics/train_result_lzh.csv','a') as f: # 'a'表示append,即在原来文件内容后继续写数据（不清楚原有数据）
            f.write('[%d/%d] Loss_G: %.6f\n' % (
                epoch, NUM_EPOCHS,
                running_results['g_loss'] / running_results['batch_sizes']))

            f.write(" lr= %.6f  " %(SchedulerG.get_lr()[0]))




        netG.eval()  # 进入eval模式 （测试模式参数固定，只有前向传播）
        out_path = 'training_results/SRF/'     # --ljd
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            i_for_save = 0
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
       

                lr = lr.to(device)
                hr = hr.to(device)
                ''' 
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                '''

                sr = netG(lr)
              


                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

                ''' 
                # 把三张图存起来
                i_for_save = i_for_save +1
                lr_tensor_for_save = lr.data.cpu().squeeze(0)
                lr_img_for_save = ToPILImage()(lr_tensor_for_save)
                lr_img_for_save.save(out_path + 'epoch_%d_lr_image_%d.bmp' % (epoch, i_for_save))
                # lr_img_for_save.show()
                # print('lr_img_for_save：', lr_tensor_for_save.shape)       # lr_img_for_save： torch.Size([1, 480, 832]) --result_ljd

                sr_tensor_for_save = sr.data.cpu().squeeze(0)
                sr_img_for_save = ToPILImage()(sr_tensor_for_save)
                sr_img_for_save.save(out_path + 'epoch_%d_sr_image_%d.bmp' % (epoch, i_for_save))

                hr_tensor_for_save = hr.data.cpu().squeeze(0)
                hr_img_for_save = ToPILImage()(hr_tensor_for_save)
                hr_img_for_save.save(out_path + 'epoch_%d_hr_image_%d.bmp' % (epoch, i_for_save))
                '''



                # 通过extend把三张图连在一起
                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
                # print('val_images：', val_images[0].shape,val_images[1].shape,val_images[2].shape,val_images[3].shape)       # val_hr_restore： torch.Size([1, 3, 348, 348]) --result_ljd

         
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 3)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1

        
        # save loss\scores\psnr\ssim
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={ 'Loss_G': results['g_loss'],'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
