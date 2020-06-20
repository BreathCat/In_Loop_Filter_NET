from os import listdir
from os.path import join

import torch.utils.data as data
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".jpg", ".jpeg"])


def load_img(filepath):
    # img = Image.open(filepath).convert('YCbCr')
    img = Image.open(filepath)
    y=img
    # print('img:',img.size, img.format, img.mode)
    # print('rgb:',Image.open(filepath).size, Image.open(filepath).format, Image.open(filepath).mode)
    # print('img size:',img.size)
    # img.show()

    # y, _, _ = img.split()
    # print('y:', y.size, y.format, y.mode)
    # print('y :',y.size)
    # print('y type:',type(y))
    # y.show()
    return y
#
# def is_input_label_match(image_input_dir,image_label_dir):
#     for x in listdir(image_input_dir) :
#         sub_input=join(image_input_dir, x)
#         matched_sub_label=join(image_label_dir,x)+
#         return sub_input,matched_sub_label

    # return sub_input,matched_sub_label





class DatasetFromFolder(data.Dataset):
    def __init__(self, image_input_dir,input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_input_dir, x) for x in listdir(image_input_dir) if is_image_file(x)]
        # self.image_label_filenames = [join(image_label_dir, x) for x in listdir(image_input_dir) if is_image_file(x)]
        # self.image_input_filenames,self.image_label_filenames=is_input_label_match(image_input_dir, image_label_dir)

        # print('join(image_dir, x):',join(image_dir, '07.bmp'))
        # print('listdir(image_dir) : ',listdir(image_input_dir))
        # print('is_image_file(x)',type(is_image_file(x)))
        self.input_transform = input_transform
        self.target_transform = target_transform





    def __getitem__(self, index):
        #'input_image' is the Y_channel of the image
        input_image = load_img(self.image_filenames[index])
        # input_image.show()
        matched_label = self.image_filenames[index].replace("subim_input", "subim_label")
        # print('self.image_filenames[index]:',self.image_filenames[index])
        # print('matched_label:',matched_label)
        # target = input_image.copy()
        target =load_img(matched_label)
        # print(type(target))
        # target.show()
        if self.input_transform:
            # print('input_image:',type(input_image))
            input_image = self.input_transform(input_image)
            # print('input_image:',input_image.shape)
            # print(self.image_filenames[index],':',input_image)
        if self.target_transform:
            target = self.target_transform(target)

        # print('input_image:',input_image[:,0:1,3:16])
        # print('input_image:',input_image.shape)
        # print('target:', target.shape)
        #the type of 'input_image, target' is torch.FloatTensor
        #input_image: torch.Size([1, 64, 64])  value of input_image is [0,1]
        #target: torch.Size([1, 256, 256])
        return input_image, target

    def __len__(self):
        return len(self.image_filenames)
