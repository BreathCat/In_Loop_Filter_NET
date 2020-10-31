#-*-coding:utf-8-*-
from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms import ToTensor
import os
import numpy as np

low_input='./test_model_Class_ABCDE/low_input/'
model_dir='./model_save'



for file in os.listdir(low_input):
    low_input_dir=os.path.join(low_input,file)#low_input_dir='./test_model_Class_ABCDE/low_input/ClassA'
                                              #low_input_dir=' ./test_model_Class_ABCDE/low_input/ClassB'
                                              #low_input_dir='./test_model_Class_ABCDE/low_input/ClassC'
                                              # low_input_dir='./test_model_Class_ABCDE/low_input/ClassD'
                                              # low_input_dir='./test_model_Class_ABCDE/low_input/ClassE'

    for low_input_image in os.listdir(low_input_dir):
        # low_input_image_name=low_input_image
        low_input_image = os.path.join(low_input_dir,low_input_image) #将路径与文件名结合起来就是每个文件的完整路径
        # print('file names: ',low_input_image)

        # ===========================================================
        # Argument settings
        # ===========================================================
        for model_epoch in os.listdir(model_dir):
            test_model_dir = os.path.join(model_dir, model_epoch, 'model_path.pth')



            parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
            # print('parser:',type(parser))
            output_names="high_output"+"_"+model_epoch

            high_output_image=low_input_image.replace("low_input", output_names).replace(".bmp","_model.bmp")
            # high_output_image_name=low_input_image_name.replace(".bmp","_model.bmp")
            print('low_input_image:      ',low_input_image)
            parser.add_argument('--input', type=str, required=False, default=low_input_image, help='input image to use')
            parser.add_argument('--model', type=str, default=test_model_dir, help='model file to use')
            parser.add_argument('--output', type=str, default=high_output_image, help='where to save the output image')
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
            # print('img type:',img)
            # img.show()



            # ===========================================================
            # model import & setting
            # ===========================================================
            device = torch.device('cuda:1' if GPU_IN_USE else 'cpu')
            # print('device:',device)
            # print('args.model:',type(args.model))
            model = torch.load(args.model, map_location=lambda storage, loc: storage)
            model = model.to(device)
            data = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
            # print('data type:',data.shape)

            data = data.to(device)
            # print('data.to(device) type:',data.shape)

            if GPU_IN_USE:
                cudnn.benchmark = True


            # ===========================================================
            # output and save image
            # ===========================================================

            with torch.no_grad():
                out = model(data)
                # out = model(out0)
            # print('out type:',out.shape)
            out = out.cpu()
            out_img_y = out.data[0].numpy()
            # print('out_img_y:',type(out_img_y))
            out_img_y *= 255.0
            out_img_y = out_img_y.clip(0, 255)
            out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
            # print('out_img_y:',type(out_img_y))

            #
            # out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
            # out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
            # out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
            out_img=out_img_y
            # print('out_img:',out_img.size, out_img.format, out_img.mode)
            out_img.save(args.output)
            print('output image saved to ', args.output)
