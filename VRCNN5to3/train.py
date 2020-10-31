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

if __name__ == '__main__':
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs


    train_path = '/home/lzh412/Desktop/dataset/Images_for_train/832×480_sub_128×128/train/subim_input'
    # train_path = '/home/lzh412/Desktop/dataset/Images_for_train/DIV2K_input_label/train/subim_input'
    train_set = TrainDatasetFromFolder(train_path, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    # train_set = TrainDatasetFromFolder('data/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('data/val/subim_input', upscale_factor=UPSCALE_FACTOR)

    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    # netD = Discriminator()
    # print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    generator_criterion = GeneratorLoss()
    # mse_loss = torch.nn.MSELoss()
    print('torch.cuda.is_available() = ',torch.cuda.is_available())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    netG.to(device)
    # netD.to(device)
    generator_criterion.to(device)

    ''' 
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
    '''

    optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.9, 0.999))
    # optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.9, 0.999))

    results = { 'g_loss': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'g_loss': 0}

        netG.train()
        # netD.train()
        for data, target in train_bar:
            g_update_first = True
            # print('data：',data.shape)   # data： torch.Size([64, 3, 22, 22])  --result_ljd
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            # print('target：', target.shape)      # target： torch.Size([64, 3, 88, 88])    --result_ljd
            # print('real_img：', real_img.shape)  # real_img： torch.Size([64, 3, 88, 88])  --result_ljd
            real_img = real_img.to(device)

            # if torch.cuda.is_available():
            #     real_img = real_img.cuda()


            z = Variable(data)
            # print('data：', data.shape)    # data： torch.Size([64, 3, 22, 22])    --result_ljd
            # print('z：', z.shape)          # z： torch.Size([64, 3, 22, 22])    --result_ljd
            z = z.to(device)

            # if torch.cuda.is_available():
            #     z = z.cuda()

            fake_img = netG(z)
            # print('fake_img：', fake_img.shape)    # fake_img： torch.Size([64, 3, 88, 88])    --result_ljd

            # netD.zero_grad()
            # real_out = netD(real_img).mean()
            # fake_out = netD(fake_img).mean()
            # print('real_img：', real_img.shape)                      # real_img： torch.Size([64, 3, 88, 88])  --result_ljd
            # print('fake_img：', fake_img.shape)                      # real_img： torch.Size([64, 3, 88, 88])  --result_ljd
            # print('netD(real_img)：', netD(real_img).shape)    # netD(real_img)： torch.Size([64])   --result_ljd
            # print('netD(fake_img)：', netD(fake_img).shape)    # netD(fake_img)： torch.Size([64])    --result_ljd
            # print('real_out = netD(real_img).mean()：', real_out.shape)    # nreal_out = netD(real_img).mean()： torch.Size([])  第一次值为 0.5066  --result_ljd
            # print('fake_out = netD(fake_img).mean()：', fake_out.shape)    # fake_out = netD(fake_img).mean()： torch.Size([])   第一次值为 0.5016    --result_ljd
            # ljd_loss = mse_loss(fake_img, real_img)
            # print('ljd_loss = ',ljd_loss)

            # 为了与VGG的输入channel对应, 这里需要把只有一个channel的real_img和fake_img变成channel=3
            real_img_channel_is_3 = torch.cat((real_img, real_img, real_img), 1)
            fake_img_channel_is_3 = torch.cat((fake_img, fake_img, fake_img), 1)
            # print('real_img_channel_is_3：', real_img_channel_is_3.shape)                      # real_img： torch.Size([64, 3, 88, 88])  --result_ljd
            # print('fake_img_channel_is_3：', fake_img_channel_is_3.shape)                      # real_img： torch.Size([64, 3, 88, 88])  --result_ljd
            # ljd_loss_channel_is_3 = mse_loss(fake_img_channel_is_3, real_img_channel_is_3)
            # print('ljd_loss = ',ljd_loss_channel_is_3)


            # d_loss = 1 - real_out + fake_out   # 第一次值为1.0004 这就是D的训练目标，最小化d_loss
            # d_loss.backward(retain_graph=True)
            # optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            g_loss = generator_criterion(fake_img_channel_is_3, real_img_channel_is_3)
            g_loss.backward()

            # fake_img = netG(z)
            # fake_out = netD(fake_img).mean()

            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            # running_results['d_loss'] += d_loss.item() * batch_size
            # running_results['d_score'] += real_out.item() * batch_size
            # running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_G: %.6f' % (
                epoch, NUM_EPOCHS,
                running_results['g_loss'] / running_results['batch_sizes']))

        netG.eval()  # 进入eval模式 （测试模式参数固定，只有前向传播）
        # out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
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
                # print('lr = val_lr：', lr.shape)       # lr = val_lr： torch.Size([1, 3, 93, 93])  --result_ljd
                # print('hr = val_hr：', hr.shape)       # hr = val_hr： torch.Size([1, 3, 372, 372])  --result_ljd

                lr = lr.to(device)
                hr = hr.to(device)
                ''' 
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                '''

                sr = netG(lr)
                # print('sr = netG(lr)：', sr.shape)       # sr = netG(lr)： torch.Size([1, 3, 372, 372])  --result_ljd


                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

                # print('val_hr_restore：', val_hr_restore.shape)       # val_hr_restore： torch.Size([1, 3, 348, 348]) --result_ljd
                # print('val_hr_restore.squeeze(0)：', val_hr_restore.squeeze(0).shape)       # val_hr_restore.squeeze(0)： torch.Size([3, 348, 348])  --result_ljd

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

            # print('val_images：', val_images[0].shape, val_images[1].shape, val_images[2].shape,
            #       val_images[3].shape)  # val_hr_restore： torch.Size([1, 3, 348, 348]) --result_ljd
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 3)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1

        # save model parameters
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
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
