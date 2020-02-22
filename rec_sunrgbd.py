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

class Rec_SUNRGBD:

    def __init__(self, cfg, data_dir=None, transform=None, phase_train=None):
        self.cfg = cfg
        self.data_dir = data_dir
        self.labeled = not cfg.UNLABELED
        self.ignore_label = -100
        self.phase_train=phase_train
        self.ms_targets=[]
        if self.labeled:
            self.classes, self.class_to_idx = self.find_classes(self.data_dir)
            self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
            self.imgs = self.make_dataset()
            self.class_weights = list(Counter([i[1] for i in self.imgs]).values())
        else:
            self.imgs = self.get_images()
            self.class_weights = None

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self):
        images = []
        dir = os.path.expanduser(self.data_dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_idx[target])
                    images.append(item)
        return images

    def get_images(self):
        images = []
        dir = os.path.expanduser(self.data_dir)
        image_names = [d for d in os.listdir(dir)]
        for image_name in image_names:
            file = os.path.join(dir, image_name)
            images.append(file)
        return images

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.labeled:
            img_path, label = self.imgs[index]
        else:
            img_path = self.imgs[index]

        img_name = os.path.basename(img_path)
        AB_conc = Image.open(img_path).convert('RGB')

        # split RGB and Depth as A and B
        w, h = AB_conc.size
        w2 = int(w / 2)
        if w2 > self.cfg.FINE_SIZE[0]:
            A = AB_conc.crop((0, 0, w2, h)).resize(self.cfg.LOAD_SIZE, Image.BICUBIC)
            B = AB_conc.crop((w2, 0, w, h)).resize(self.cfg.LOAD_SIZE, Image.BICUBIC)
        else:
            A = AB_conc.crop((0, 0, w2, h))
            B = AB_conc.crop((w2, 0, w, h))

        if self.labeled:
            sample = {'image': A, 'label': label}
        else:
            sample = {'image': A}

        if self.cfg.TARGET_MODAL == 'depth':
            sample['depth'] = B

        if self.cfg.WHICH_DIRECTION == 'BtoA':
            sample['image'], sample['depth'] = sample['depth'], sample['image']

        if self.phase_train:
            return self.transform_tr(sample)
        elif not self.phase_train:
            return self.transform_val(sample)


        # if self.transform:
        #     sample = self.transform(sample)

        # return sample
    def transform_tr(self, sample):
        train_transforms = list()
        if self.cfg.TASK_TYPE!='infomax':
            train_transforms.append(tr.RandomScale(self.cfg.RANDOM_SCALE_SIZE))
        train_transforms.append(tr.RandomRotate())
        train_transforms.append(tr.RandomCrop_Unaligned(self.cfg.FINE_SIZE, pad_if_needed=True, fill=0))
        train_transforms.append(tr.RandomHorizontalFlip())
        if self.cfg.TARGET_MODAL == 'lab':
            train_transforms.append(tr.RGB2Lab())
        if self.cfg.MULTI_SCALE:
            for item in self.cfg.MULTI_TARGETS:
                self.ms_targets.append(item)
            train_transforms.append(tr.MultiScale(size=cfg.FINE_SIZE,scale_times=cfg.MULTI_SCALE_NUM, ms_targets=self.ms_targets))
        train_transforms.append(tr.ToTensor())
        train_transforms.append(tr.Normalize(mean=self.cfg.MEAN, std=self.cfg.STD, ms_targets=self.ms_targets))
        composed_transforms = transforms.Compose(train_transforms)
        return composed_transforms(sample)

    def transform_val(self, sample):
        val_transforms = list()
        val_transforms.append(tr.Resize(self.cfg.LOAD_SIZE))
        if self.cfg.MULTI_SCALE:
            val_transforms.append(tr.MultiScale(size=self.cfg.FINE_SIZE,scale_times=self.cfg.MULTI_SCALE_NUM, ms_targets=self.ms_targets))
        val_transforms.append(tr.ToTensor())
        val_transforms.append(tr.Normalize(mean=self.cfg.MEAN, std=self.cfg.STD, ms_targets=self.ms_targets))
        composed_transforms = transforms.Compose(val_transforms)