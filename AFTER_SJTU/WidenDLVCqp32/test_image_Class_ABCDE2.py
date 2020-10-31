import argparse
import time
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
from model import Generator

low_input='./test_model_Class_ABCDE/low_input/'
model_dir='./epochs'

for file in os.listdir(low_input):
    low_input_dir=os.path.join(low_input,file)#low_input_dir='./test_model_Class_ABCDE/low_input/ClassA'
                                              #low_input_dir=' ./test_model_Class_ABCDE/low_input/ClassB'
                                              #low_input_dir='./test_model_Class_ABCDE/low_input/ClassC'
                                              # low_input_dir='./test_model_Class_ABCDE/low_input/ClassD'
                                              # low_input_dir='./test_model_Class_ABCDE/low_input/ClassE'

    for low_input_image in os.listdir(low_input_dir):
        # low_input_image_name=low_input_image
        low_input_image = os.path.join(low_input_dir,low_input_image) #将路径与文件名结合起来就是每个文件的完整路径
        print('low_input_image: ',low_input_image)

        # ===========================================================
        # Argument settings
        # ===========================================================
        for epoch in range(1,161):
            if epoch % 20 == 0 and epoch != 0:
                test_model_name = 'netG_epoch_1_' + str(epoch) + '.pth'
                test_model_dir = os.path.join(model_dir,test_model_name)
                print('test_model_dir:',test_model_dir)

                parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
                output_names="high_output"+"_"+str(epoch)
                # print('output_names:',output_names)


                high_output_image=low_input_image.replace("low_input", output_names).replace(".bmp","_model.bmp")
                print('high_output_image:',high_output_image)


                parser = argparse.ArgumentParser(description='Test Single Image')
                parser.add_argument('--upscale_factor', default=1, type=int, help='super resolution upscale factor')
                parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
                parser.add_argument('--image_name', default=low_input_image, type=str, help='test low resolution image name')
                parser.add_argument('--model_name', default=test_model_dir, type=str, help='generator model epoch name')
                opt = parser.parse_args()

                UPSCALE_FACTOR = opt.upscale_factor
                TEST_MODE = True if opt.test_mode == 'GPU' else False
                IMAGE_NAME = opt.image_name
                MODEL_NAME = opt.model_name

                model = Generator(UPSCALE_FACTOR).eval()
                if TEST_MODE:
                    model.cuda()
                    model.load_state_dict(torch.load(MODEL_NAME))
                else:
                    model.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage))

                image = Image.open(IMAGE_NAME)
                # print('/////////////////////////////////')
                # print('pixel:',image.getpixel((1, 4)))
                data = (ToTensor()(image)).view(1, -1, image.size[1], image.size[0])
                # print('image:',image[0][0][1][4])
                if TEST_MODE:
                    data = data.cuda()

                # start = time.clock()
                with torch.no_grad():
                    out = model(data)
                out = out.cpu()
                out_img_y = out.data[0].numpy()
                out_img_y *= 255.0
                out_img_y = out_img_y.clip(0, 255)
                out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
                # print('out_pixel:',out_img.getpixel((1, 4)))
                out_img_y.save(high_output_image)
