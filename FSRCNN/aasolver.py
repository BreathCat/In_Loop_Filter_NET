from __future__ import print_function
from math import log10
from os.path import exists, join, basename
import torch
import torch.backends.cudnn as cudnn
from FSRCNN.model import Net
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

class FSRCNNTrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(FSRCNNTrainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
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
        self.model = Net(num_channels=1, upscale_factor=self.upscale_factor).to(self.device)
        self.model.weight_init(mean=0.0, std=0.2)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)  # lr decay

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
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def test(self):
        self.model.eval()
        avg_psnr = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                mse = self.criterion(prediction, target)
                test_loss += mse.item()
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                # progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))

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
