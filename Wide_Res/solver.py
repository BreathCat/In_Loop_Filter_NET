from __future__ import print_function

from math import log10
from os.path import exists, join, basename
import torch
import torch.backends.cudnn as cudnn

from Wide_Res.model import WideResNet,BasicBlock,NetworkBlock
# from progress_bar import progress_bar


def get_parameters(model, bias=False):
    import torch.nn as nn
    # modules_skipped = (
    #     nn.ReLU,
    #     nn.Sequential,
    #
    # )
    for m in model.modules():
        # print(m)

        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, nn.Conv2d)==False:
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))

def weight_init(self, mean, std):
        # for i in self.children():
        #     print('i:',i)
        # 访问 modules
        for m in self.modules():
            # print('m:',m)
            if  isinstance(m, nn.Conv2d):
                print('Wide_Res正在初始化权重')
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')


class Wide_ResTrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(Wide_ResTrainer, self).__init__()
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
        self.widen_factor = config.widen_factor
        self.QP = config.qp
    def build_model(self):
        self.model = WideResNet(28, self.widen_factor, dropRate=0).to(self.device)
        print(self.widen_factor)
        print(WideResNet)
        # self.model.weight_init(mean=0.0, std=0.01)
        ######## below is loading existing model
        self.model.load_state_dict(torch.load('/home/li/Desktop/LZH/CNN_NET/QP37_Wide_Res_state_dict.pth'))
        ######## above is loading existing model
        
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)
        # self.model = torch.nn.DataParallel(Net, device_ids=[0, 1])
        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

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
        model_out_path = "Wide_Res.pth"
        model_out_path = join(root_dir, epo_ch,"LZHdataset_epoch"+epo_ch+"_QP"+str(self.QP)+"__reinitial_"+model_out_path)
        torch.save(self.model, model_out_path)
        torch.save(self.model.state_dict(), root_dir+'/'+epo_ch+"/QP"+str(self.QP)+"__reinitial_"+ "Wide_Res_state_dict.pth")
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
            if epoch%20 == 0 or epoch == 1:
                global epo_ch
                epo_ch = str(epoch)
                self.save_model()

