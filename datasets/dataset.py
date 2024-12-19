import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image

# def label_to_multiclass(label):
#     LIVER_RANGE = (55, 70)
#     RIGHT_KIDNEY_RANGE = (110, 135)
#     LEFT_KIDNEY_RANGE = (175, 200)
#     SPLEEN_RANGE = (240, 255)

#     liver_mask = (label >= LIVER_RANGE[0]) & (label <= LIVER_RANGE[1])
#     right_kidney_mask = (label >= RIGHT_KIDNEY_RANGE[0]) & (label <= RIGHT_KIDNEY_RANGE[1])
#     left_kidney_mask = (label >= LEFT_KIDNEY_RANGE[0]) & (label <= LEFT_KIDNEY_RANGE[1])
#     spleen_mask = (label >= SPLEEN_RANGE[0]) & (label <= SPLEEN_RANGE[1])

#     multi_label = np.zeros_like(label)
#     multi_label[liver_mask] = 1
#     multi_label[right_kidney_mask] = 2
#     multi_label[left_kidney_mask] = 3
#     multi_label[spleen_mask] = 4

#     return multi_label

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def binarize_label(label, threshold=127):
    return (label > threshold).astype(np.float32)

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # label = binarize_label(label)  # 이 부분을 주석 처리 또는 삭제

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class MRT1_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, output_size=(256, 256)):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.output_size = output_size

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')
        data_path = os.path.join(self.data_dir, slice_name+'.npz')
        data = np.load(data_path)
        image, label = data['image'], data['label']

        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = Image.fromarray(image)
            image = image.convert('L')
            image = np.array(image)
        
        if image.shape[0] != self.output_size[0] or image.shape[1] != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / image.shape[0], self.output_size[1] / image.shape[1]), order=3)
            label = zoom(label, (self.output_size[0] / label.shape[0], self.output_size[1] / label.shape[1]), order=0)
        # label = label_to_multiclass(label)
        label[label > 0] = 1

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = slice_name
        return sample