import tarfile
from os import remove
from os.path import exists, join, basename

from six.moves import urllib
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from .dataset import DatasetFromFolder

output_image_dir='./dataset/Images_for_train/DIV2K_input_label'   #QP=37的９万对图像
# output_image_dir='./dataset/Images_for_train/input_label'     #QP=37的4万对图像
# output_image_dir='./dataset/Images_for_train/DIV2K_input_label_QP22'
#output_image_dir='./dataset/Images_for_train/DIV2K_input_label'#使用的数据集，９万对
#output_image_dir='./dataset/Images_for_train/input_label'#4万对
# output_image_dir='./dataset/Images_for_train/ultra_high_resolution'

# output_image_dir='./dataset/Images_for_train/ultra_high_resolution'
# output_image_dir='./dataset/Images_for_train/832×480_sub_35×35'
# output_image_dir='./dataset/Images_for_train/832×480_sub_128×128'
# output_image_dir='./dataset/BSDS300/832_780_286'
# output_image_dir='./dataset/BSDS300/cut_videos_subimages_2_except_trfficflow_Intersection_Mainroad'
# output_image_dir='./dataset/BSDS300/yizhang'


#
# def download_bsd300(dest="./dataset"):
#     output_image_dir = join(dest, "BSDS300/images")
#     print(output_image_dir)
#
#     if not exists(output_image_dir):
#         url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
#         print("downloading url ", url)
#
#         data = urllib.request.urlopen(url)
#
#         file_path = join(dest, basename(url))
#         with open(file_path, 'wb') as f:
#             f.write(data.read())
#
#         print("Extracting data")
#         with tarfile.open(file_path) as tar:
#             for item in tar:
#                 tar.extract(item, dest)
#
#         remove(file_path)
#
#     return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(35),
        # Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(35),
        ToTensor(),
    ])


def get_training_set(upscale_factor):
    root_dir = output_image_dir
    # train_input_dir = join(root_dir, "train")
    train_input_dir = join(root_dir, "train","subim_input")
    # train_label_dir = join(root_dir, "train","subim_label")
    crop_size = calculate_valid_crop_size(140, upscale_factor)#crop_size=256
    # print('crop_size:',crop_size)
    # print('upscale_factor:', upscale_factor)
    # print('input_transform:',input_transform(crop_size, upscale_factor))

    return DatasetFromFolder(train_input_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))



def get_test_set(upscale_factor):
    root_dir = output_image_dir
    # test_dir = join(root_dir, "test")
    # test_input_dir = join(root_dir, "test")
    test_input_dir = join(root_dir, "test", "subim_input")
    # test_label_dir = join(root_dir, "test", "subim_label")
    crop_size = calculate_valid_crop_size(128, upscale_factor)

    return DatasetFromFolder(test_input_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))



