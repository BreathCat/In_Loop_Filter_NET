from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np

# ===========================================================
# Argument settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# print('parser:',type(parser))
parser.add_argument('--input', type=str, required=False, default='./test_model/low_input/not_in_train_and_test/total_PartyScene_832x480_50_off8.bmp', help='input image to use')
parser.add_argument('--model', type=str, default='./model_save/40/model_path.pth', help='model file to use')
parser.add_argument('--output', type=str, default='test.bmp', help='where to save the output image')
args = parser.parse_args()
# print(args)


# ===========================================================
# input image setting
# ===========================================================
GPU_IN_USE = torch.cuda.is_available()
# print('args.input type:',type(args.input))
# img = Image.open(args.input).convert('YCbCr')
# print('img type:',img)
# img.show()

# y, cb, cr = img.split()
# y.show()
# cb.show()
# cr.show()

img = Image.open(args.input)
y=img
print('img type:',img)
# img.show()



# ===========================================================
# model import & setting
# ===========================================================
device = torch.device('cuda' if GPU_IN_USE else 'cpu')
print('device:',device)
print('args.model:',type(args.model))
model = torch.load(args.model, map_location=lambda storage, loc: storage)
model = model.to(device)
data = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
print('data type:',data.shape)

data = data.to(device)
print('data.to(device) type:',data.shape)

if GPU_IN_USE:
    cudnn.benchmark = True


# ===========================================================
# output and save image
# ===========================================================

out = model(data)
print('out type:',out.shape)
out = out.cpu()
out_img_y = out.data[0].numpy()
print('out_img_y:',type(out_img_y))
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
print('out_img_y:',type(out_img_y))

#
# out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
# out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
# out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
out_img=out_img_y
print('out_img:',out_img.size, out_img.format, out_img.mode)
out_img.save(args.output)
print('output image saved to ', args.output)
