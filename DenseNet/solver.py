from __future__ import print_function

from math import log10
from os.path import exists, join, basename
import torch
import torch.backends.cudnn as cudnn

import DenseNet.model as DENSE


# from progress_bar import progress_bar


# def get_parameters(model, bias=False):
#     import torch.nn as nn
#     # modules_skipped = (
#     #     nn.ReLU,
#     #     nn.Sequential,
#     #
#     # )
#     for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()

    




class DENSETrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(DENSETrainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        # self.CUDA = torch.cuda.set_device(1)
        self.device = torch.device('cuda:0' if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader

    def build_model(self):
        self.model = DENSE.DenseNet(growthRate=12, depth=40, reduction=0.5,
                            bottleneck=True, nClasses=10)
        
          
        # self.model = torch.nn.DataParallel(Net, device_ids=[0, 1])
        self.model.weight_init(mean=0.0, std=0.01)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()
            self.model.cuda()

        # TODO 可能可以用原文的下面这个梯度下降函数
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-1,betas=(0.9, 0.999), momentum=0.9, weight_decay=1e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,betas=(0.9, 0.999), eps=1e-8,weight_decay=0.0001)
    
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)
        # self.optimizer = torch.optim.SGD([{'params': get_parameters(self.model, bias=False)},
        #                                   {'params': get_parameters(self.model, bias=True), 'lr': 0.01}],
        #                                  lr=self.lr, momentum=0.9, weight_decay=0.0001)
        # self.optimizer = torch.optim.ASGD(self.model.parameters(),lr=self.lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
        # self.optimizer = torch.optim.ASGD(self.model.parameters(),lr=self.lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)



        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[40, 80, 120], gamma=0.1)#设置每个参数组的学习率

    def save_model(self):
        root_dir='./model_save'
        model_out_path = "model_path.pth"
        model_out_path = join(root_dir, epo_ch,model_out_path)
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            # print('data:', data.shape)
            # print('target:',target.shape)
            # print('self.model(data):', self.model(data).shape)
            # print('prediction:',prediction.shape)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            # progress_bar(batch_num, len(self.training_loader), '    Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average train_Loss: {:.6f}".format(train_loss / len(self.training_loader)),end='')
        # print('len(self.training_loader):',len(self.training_loader),end='')

    def test(self):
        self.model.eval()
        avg_psnr = 0
        test_loss=0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                # print('data:', data.shape)
                # print('target:',target.shape)
                # print('prediction:',prediction.shape)
                mse = self.criterion(prediction, target)
                test_loss += mse.item()
                # print('mse type:',type(mse.item()))
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                # progress_bar(batch_num, len(self.testing_loader), '    PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        print("    Average test_Loss: {:.6f}".format(test_loss / len(self.testing_loader)), end='')
        print("    Average PSNR: {:.8f} dB".format(avg_psnr / len(self.testing_loader)),end='')

    def run(self):
        self.build_model()
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [4, 8,12], 0.1)
        for epoch in range(1, self.nEpochs + 1):
            # scheduler.step()
            print("\nEpoch {} starts:".format(epoch),end='')
            # print("lr= %g" % self.scheduler.get_lr()[0],end='')
            print(" lr= {:.10f} dB".format(self.scheduler.get_lr()[0]), end='')
            # print('lr= ',self.lr)
            self.train()
            self.test()
            self.scheduler.step(epoch)
            # if epoch == self.nEpochs:
            #     self.save_model()
            if epoch%20 == 0:
                global epo_ch
                epo_ch = str(epoch)
                self.save_model()

