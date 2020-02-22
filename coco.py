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

class Seg_COCO(Dataset):
    NUM_CLASSES = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
        1, 64, 20, 63, 7, 72]

    def __init__(self,cfg=None, data_dir=None, phase_train=True,year='2017'):
        super().__init__()
        if phase_train:
            split='train'
        else:
            split='val'
        ann_file = os.path.join(data_dir, 'annotations/instances_{}{}.json'.format(split, year))
        ids_file = os.path.join(data_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(data_dir, 'images/{}{}'.format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        self.cfg=cfg
        self.ignore_label=255
        self.class_weights=None
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.ms_targets=[]
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
        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        _target = Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']))

        return _img, _target

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask
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

    def __len__(self):
        return len(self.ids)