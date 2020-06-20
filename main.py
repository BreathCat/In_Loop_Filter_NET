# coding=utf-8
from __future__ import print_function

import argparse

from torch.utils.data import DataLoader

from VRCNN.solver import VRCNNTrainer
from VRCNN_no_res.solver import VRCNN_no_resTrainer
from VRCNN_ext.solver import VRCNN_extTrainer
from DLVC.solver import DLVCTrainer
from VRCNN_ext_1conv.solver import VRCNN_ext_1convTrainer
from Multiresolution_CNN.solver import Multiresolution_CNNTrainer
from VRCNN_1conv.solver import VRCNN_1convTrainer
from Multiresolution_VRCNN.solver import Multiresolution_VRCNNTrainer
from Multiresolution_DLVC.solver import Multiresolution_DLVCTrainer
from SRGAN.solver import SRGANTrainer




from dataset.data import get_training_set, get_test_set
from torchsummary import summary

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=128, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=64, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=160, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='DLVC', help='choose which model is going to use')

args = parser.parse_args()


def main():
    # ===========================================================
    # Set train dataset & test dataset
    # ===========================================================
    # print('===> Loading datasets')
    train_set = get_training_set(args.upscale_factor)
    # print('train_set:',type(train_set))
    test_set = get_test_set(args.upscale_factor)
    # print('test_set:',type(test_set))
    training_data_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    dataiter = iter(training_data_loader)
    imgs, labels = next(dataiter)
    print('imgs.size():',imgs.size())  # batch_size, channel, height, weight
    # print('imgs type:',imgs[0,0,0:10,0:10])
    # print('labels type:',labels[0,0,0:10,0:10])
    print('labels.size():',labels.size())  # batch_size, channel, height, weight


    testing_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    # dataiter = iter(testing_data_loader)
    # imgs, labels = next(dataiter)
    # print('imgs.size():',imgs.size())  # batch_size, channel, height, weight
    # print('imgs type:',labels[0,0,0,:10])
    # print('labels.size():',labels.size())  # batch_size, channel, height, weight




    if args.model == 'VRCNN':
        model = VRCNNTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'VRCNN_no_res':
        model = VRCNN_no_resTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'VRCNN_ext':
        model = VRCNN_extTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'DLVC':
        model = DLVCTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'VRCNN_ext_1conv':
        model = VRCNN_ext_1convTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'Multiresolution_CNN':
        model = Multiresolution_CNNTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'VRCNN_1conv':
        model = VRCNN_1convTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'Multiresolution_VRCNN':
        model = Multiresolution_VRCNNTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'Multiresolution_DLVC':
        model = Multiresolution_DLVCTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'SRGAN':
        model = SRGANTrainer(args, training_data_loader, testing_data_loader)

    else:
        raise Exception("the model does not exist")

    model.run()





if __name__ == '__main__':
    main()
