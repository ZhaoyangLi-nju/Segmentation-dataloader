import os
import random

import h5py
import numpy as np
import scipy.io
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import numbers
from util.utils import color_label_np
import PIL.ImageEnhance as ImageEnhance
from collections import Counter
from skimage import color
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.datasets as dataset
import skimage.transform
from pycocotools.coco import COCO
from pycocotools import mask
from tqdm import trange
from data import custom_transforms as tr

class Seg_VOC(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self, cfg=None, data_dir=None, phase_train=True):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = data_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')
        self.class_weights=None
        self.ignore_label=255
        self.cfg=cfg
        if phase_train:
            split='train'
        else:
            split='val'
        self.split = split
        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.ms_targets=[]
        with open(os.path.join(os.path.join(_splits_dir, split + '.txt')), "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            _image = os.path.join(self._image_dir, line + ".jpg")
            _cat = os.path.join(self._cat_dir, line + ".png")
            assert os.path.isfile(_image)
            assert os.path.isfile(_cat)
            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target
    def __getitem__(self, index):
        image, label = self._make_img_gt_point_pair(index)
        seg = None

        if self.cfg.NO_TRANS == False:
            if 'seg' == self.cfg.TARGET_MODAL:
                seg = Image.fromarray((color_label_np(label_copy, ignore=self.ignore_label).astype(np.uint8)), mode='RGB')
                # seg = np.load(seg_path)
                seg = Image.fromarray(seg.astype(np.uint8), mode='RGB')
            sample = {'image': image, 'label': label, 'seg': seg}
        else:
            # print(image.size)
            sample = {'image': image, 'label': label}
        for key in list(sample.keys()):
            if sample[key] is None:
                sample.pop(key)
        # if self.transform:
        #     sample = self.transform(sample)

        # return sample

        # for split in self.split:
        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)


    def transform_tr(self, sample):
        train_transforms = list()
        train_transforms.append(tr.RandomHorizontalFlip())
        train_transforms.append(tr.RandomScaleCrop(base_size=self.cfg.LOAD_SIZE, crop_size=self.cfg.FINE_SIZE))
        train_transforms.append(tr.RandomGaussianBlur())
        train_transforms.append(tr.ToTensor())
        train_transforms.append(tr.Normalize(mean=self.cfg.MEAN, std=self.cfg.STD, ms_targets=self.ms_targets))
        composed_transforms = transforms.Compose(train_transforms)
        return composed_transforms(sample)

    def transform_val(self, sample):
        val_transforms=list()
        val_transforms.append(tr.FixScaleCrop(crop_size=self.cfg.FINE_SIZE))
        val_transforms.append(tr.ToTensor())
        val_transforms.append(tr.Normalize(mean=self.cfg.MEAN, std=self.cfg.STD, ms_targets=self.ms_targets))
        composed_transforms = transforms.Compose(val_transforms)

        return composed_transforms(sample)

    # def __str__(self):
    #     return 'VOC2012(split=' + str(self.split) + ')'