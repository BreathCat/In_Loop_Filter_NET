from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.bmp'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        CenterCrop(35),
        # ToPILImage(),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(35),
        # ToPILImage(),
        # Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        # print('image_filenames:',self.image_filenames[index])
        # matched_label = self.image_filenames[index].replace("subim_input", "subim_label")
        # print('matched_label:',matched_label)

        lr_image = self.lr_transform(Image.open(self.image_filenames[index]))
        hr_image = self.hr_transform(Image.open(self.image_filenames[index].replace("subim_input", "subim_label")))

        # diff = hr_image-lr_image

        # input_image = load_img(self.image_filenames[index])
        # input_image.show()
        # print('hr_image：', hr_image.shape)    # hr_image： torch.Size([3, 88, 88])    --result_ljd
        # print('lr_image：', lr_image.shape)    # lr_image： torch.Size([3, 22, 22])    --result_ljd

        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        lr_image = Image.open(self.image_filenames[index])
        # print('image_filenames:',self.image_filenames[index])
        matched_hr = self.image_filenames[index].replace("subim_input", "subim_label")
        matched_hr = matched_hr[:-9] + str(1) +matched_hr[-4:]
        # print('matched_hr:',matched_hr)
        hr_image = Image.open(matched_hr)

        # w, h = hr_image.size
        # crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        # lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        # hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        # hr_image = CenterCrop(crop_size)(hr_image)
        # lr_image = lr_scale(hr_image)
        hr_restore_img = lr_image
        # print('lr_image:',lr_image.size)
        # print('hr_image:',hr_image.size)
        # print('hr_restore_img:',hr_restore_img.size)
        # diff1 = ToTensor()(lr_image) - ToTensor()(hr_restore_img)
        # diff2 = ToTensor()(lr_image) - ToTensor()(hr_image)
        # diff3 = ToTensor()(hr_restore_img) - ToTensor()(hr_image)

        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)

''' 
class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
'''