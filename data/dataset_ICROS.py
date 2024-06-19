import os
import random
import copy

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from data.image_utils import random_augmentation, crop_img
from data.degradation_utils import Degradation


class NYUV2Dataset(Dataset):

    def __init__(self, args, is_train=True):
        super(NYUV2Dataset, self).__init__()
        self.args = args
        self.is_train = is_train
        if is_train:
            with open('/home/jnu/Project/dataset/Depth_Estimation/NYU_depth_V2/nyudepthv2_train_files_with_gt.txt', 'r') as f:
                self.filenames = f.readlines()

        else:
            with open('/home/jnu/Project/dataset/Depth_Estimation/NYU_depth_V2/nyudepthv2_test_files_with_gt.txt', 'r') as f:
                self.filenames = f.readlines()
        self.toTensor = ToTensor()

    def __getitem__(self, idx):
        sample_path = self.filenames[idx] # (image, depth, focal)
        # focal = float(sample_path.split()[2])
        focal = 518.8579
        if self.is_train: # for Training
            image_path = os.path.join('/home/jnu/Project/dataset/Depth_Estimation/NYU_depth_V2', 'train' + sample_path.split()[0])
            depth_path = os.path.join('/home/jnu/Project/dataset/Depth_Estimation/NYU_depth_V2', 'train' + sample_path.split()[1])
            # image_path = os.path.join('/home/jnu/Project/dataset/Depth_Estimation/NYU_depth_V2/train', sample_path.split()[0])
            # depth_path = os.path.join('/home/jnu/Project/dataset/Depth_Estimation/NYU_depth_V2/train', sample_path.split()[1])

            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)
            # print(image, depth_gt)
            if self.args.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            # To avoid blank boundaries due to pixel registration
            if self.args.dataset_name == 'nyu':
                if self.args.input_height == 480:
                    depth_gt = np.array(depth_gt)
                    valid_mask = np.zeros_like(depth_gt)
                    valid_mask[45:472, 43:608] = 1
                    depth_gt[valid_mask == 0] = 0
                    depth_gt = Image.fromarray(depth_gt)
                else:
                    depth_gt = depth_gt.crop((43, 45, 608, 472)) # for NYU
                    image = image.crop((43, 45, 608, 472)) # for NYU
                    # image, depth_gt = image.resize((640, 480), Image.BICUBIC), depth_gt.resize((640, 480), Image.BICUBIC)

            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = image.rotate(random_angle, resample=Image.BILINEAR)
                depth_gt = depth_gt.rotate(random_angle, resample=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / 1000.0 # for nyu

            if image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width:
                image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
            image, depth_gt = self.train_preprocess(image, depth_gt)

            sample = {'image': image, 'depth': depth_gt, 'focal': focal}

        else: # for valid test
            # image_path = os.path.join('/data1/Depth_Estimation/NYU_depth_V2/test', sample_path.split()[0])
            image_path = os.path.join('/home/jnu/Project/dataset/Depth_Estimation/NYU_depth_V2/test_hazy', sample_path.split()[0])
            # depth_path = os.path.join('/data1/Depth_Estimation/NYU_depth_V2/test', sample_path.split()[1])
            # print("A")
            image = Image.open(image_path).convert('RGB')
            # plt.imshow(image)
            # plt.show()
            # image = image.crop((43, 45, 608, 472))  # for NYU
            # plt.imshow(image)
            # plt.show()
            # image = image.resize((640, 480), Image.BICUBIC)
            # plt.imshow(image)
            # plt.show()
            # image = np.array(image)
            # print(image.shape)
            # plt.imshow(image)
            # plt.show()
            # plt.imshow(image)
            # image = self._add_gaussian_noise(image, sigma=25)
            # plt.imshow(image)
            # plt.show()
            image = self.toTensor(image)
            return image, sample_path.split()[0]

        return sample

    def _add_gaussian_noise(self, clean_patch, sigma):
        # noise = torch.randn(*(clean_patch.shape))
        # clean_patch = self.toTensor(clean_patch)
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        # noisy_patch = torch.clamp(clean_patch + noise * sigma, 0, 255).type(torch.int32)
        return noisy_patch

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset_name == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

