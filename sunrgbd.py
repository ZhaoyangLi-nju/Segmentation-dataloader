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

class Seg_SUNRGBD(Dataset):

    def __init__(self, cfg=None, data_dir=None, phase_train=None):

        self.cfg = cfg
        self.phase_train = phase_train
        self.ignore_label = 255
        self.ms_targets=[]
        self.id_to_trainid = {-1: self.ignore_label, 0: self.ignore_label, 1: 0, 2: 1,
                              3: 2, 4: 3, 5: 4, 6: 5,
                              7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12,
                              14: 13, 15: 14, 16: 15, 17: 16,
                              18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 26: 25, 27: 26,
                              28: 27, 29: 28, 30: 29, 31: 30, 32: 31, 33: 32, 34: 33, 35: 34, 36: 35, 37: 36}

        self.img_dir_train_file = './sunrgbd_seg/img_dir_train.txt'
        self.depth_dir_train_file = './sunrgbd_seg/depth_dir_train.txt'
        self.label_dir_train_file = './sunrgbd_seg/label_train.txt'
        self.img_dir_test_file = './sunrgbd_seg/img_dir_test.txt'
        self.depth_dir_test_file = './sunrgbd_seg/depth_dir_test.txt'
        self.label_dir_test_file = './sunrgbd_seg/label_test.txt'
        self.class_weights = [0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
                              0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
                              2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
                              0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
                              1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
                              4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
                              3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
                              0.750738, 4.040773]
        self.img_dir_train = []
        self.depth_dir_train = []
        self.label_dir_train = []
        self.img_dir_test = []
        self.depth_dir_test = []
        self.label_dir_test = []

        try:
            with open(self.img_dir_train_file, 'r') as f:
                for i in f.read().splitlines():
                    self.img_dir_train.append(os.path.join(data_dir,i))
                # self.img_dir_train = os.path.join(data_dir, f.read().splitlines())
            with open(self.depth_dir_train_file, 'r') as f:
                for i in f.read().splitlines():
                    self.depth_dir_train.append(os.path.join(data_dir,i))
                # self.depth_dir_train = os.path.join(data_dir, f.read().splitlines())
            with open(self.label_dir_train_file, 'r') as f:
                for i in f.read().splitlines():
                    self.label_dir_train.append(os.path.join(data_dir,i))
	            # self.label_dir_train = os.path.join(data_dir, f.read().splitlines())
            with open(self.img_dir_test_file, 'r') as f:
                for i in f.read().splitlines():
                    self.img_dir_test.append(os.path.join(data_dir,i))
	            # self.img_dir_test = os.path.join(data_dir, f.read().splitlines())
            with open(self.depth_dir_test_file, 'r') as f:
                for i in f.read().splitlines():
                    self.depth_dir_test.append(os.path.join(data_dir,i))
	            # self.depth_dir_test = os.path.join(data_dir, f.read().splitlines())
            with open(self.label_dir_test_file, 'r') as f:
                for i in f.read().splitlines():
                    self.label_dir_test.append(os.path.join(data_dir,i))
	            # self.label_dir_test = os.path.join(data_dir, f.read().splitlines())
        except:

            SUNRGBDMeta_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat')
            allsplit_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
            SUNRGBD2Dseg_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat')
            self.img_dir_train = []
            self.depth_dir_train = []
            self.label_dir_train = []
            self.img_dir_test = []
            self.depth_dir_test = []
            self.label_dir_test = []
            self.SUNRGBD2Dseg = h5py.File(SUNRGBD2Dseg_dir, mode='r', libver='latest')

            SUNRGBDMeta = scipy.io.loadmat(SUNRGBDMeta_dir, squeeze_me=True,
                                           struct_as_record=False)['SUNRGBDMeta']
            split = scipy.io.loadmat(allsplit_dir, squeeze_me=True, struct_as_record=False)
            split_train = split['alltrain']

            seglabel = self.SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']

            for i, meta in enumerate(SUNRGBDMeta):
                meta_dir = '/'.join(meta.rgbpath.split('/')[:-2])
                real_dir = meta_dir.replace('/n/fs/sun3d/data', data_dir)
                depth_bfx_path = os.path.join(real_dir, 'hha/' + meta.depthname)
                rgb_path = os.path.join(real_dir, 'image/' + meta.rgbname)

                label_path = os.path.join(real_dir, 'label/label.npy')

                if not os.path.exists(label_path):
                    os.makedirs(os.path.join(real_dir, 'label'), exist_ok=True)
                    label = np.array(self.SUNRGBD2Dseg[seglabel.value[i][0]].value.transpose(1, 0))
                    np.save(label_path, label)

                if meta_dir in split_train:
                    self.img_dir_train = np.append(self.img_dir_train, rgb_path)
                    self.depth_dir_train = np.append(self.depth_dir_train, depth_bfx_path)
                    self.label_dir_train = np.append(self.label_dir_train, label_path)
                else:
                    self.img_dir_test = np.append(self.img_dir_test, rgb_path)
                    self.depth_dir_test = np.append(self.depth_dir_test, depth_bfx_path)
                    self.label_dir_test = np.append(self.label_dir_test, label_path)

            local_file_dir = '/'.join(self.img_dir_train_file.split('/')[:-1])
            if not os.path.exists(local_file_dir):
                os.mkdir(local_file_dir)
            with open(self.img_dir_train_file, 'w') as f:
                f.write('\n'.join(self.img_dir_train))
            with open(self.depth_dir_train_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_train))
            with open(self.label_dir_train_file, 'w') as f:
                f.write('\n'.join(self.label_dir_train))
            with open(self.img_dir_test_file, 'w') as f:
                f.write('\n'.join(self.img_dir_test))
            with open(self.depth_dir_test_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_test))
            with open(self.label_dir_test_file, 'w') as f:
                f.write('\n'.join(self.label_dir_test))

        self.seg_dir_train = []
        self.seg_dir_test = []
        try:
            with open(self.img_dir_train_file, 'r') as f:
                self.img_dir_train = f.read().splitlines()
            for i in range(len(self.img_dir_train)):
                self.img_dir_train[i] = self.img_dir_train[i].split(".")[0] + ".npy"

            with open(self.depth_dir_train_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
            for i in range(len(self.depth_dir_train)):
                self.depth_dir_train[i] = self.depth_dir_train[i].split(".")[0] + ".npy"

            with open(self.label_dir_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()
            for i in range(len(self.label_dir_train)):
                self.seg_dir_train.append(self.label_dir_train[i].split(".")[0] + "seg.npy")
            with open(self.label_dir_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()
            for i in range(len(self.label_dir_train)):
                self.label_dir_train.append(self.label_dir_train[i].split(".")[0] + ".npy")

            with open(self.label_dir_test_file, 'r') as f:
                self.label_dir_test = f.read().splitlines()
            for i in range(len(self.label_dir_train)):
                self.label_dir_test.append(self.label_dir_test[i].split(".")[0] + ".npy")
            with open(self.img_dir_test_file, 'r') as f:
                self.img_dir_test = f.read().splitlines()
            for i in range(len(self.img_dir_test)):
                self.img_dir_test[i] = self.img_dir_test[i].split(".")[0] + ".npy"

            with open(self.depth_dir_test_file, 'r') as f:
                self.depth_dir_test = f.read().splitlines()
            for i in range(len(self.depth_dir_test)):
                self.depth_dir_test[i] = self.depth_dir_test[i].split(".")[0] + ".npy"

            with open(self.label_dir_test_file, 'r') as f:
                self.label_dir_test = f.read().splitlines()
            for i in range(len(self.label_dir_test)):
                self.seg_dir_test.append(self.label_dir_test[i].split(".")[0] + "seg.npy")

        except:

            SUNRGBDMeta_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat')
            allsplit_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
            SUNRGBD2Dseg_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat')
            self.img_dir_train = []
            self.depth_dir_train = []
            self.label_dir_train = []
            self.img_dir_test = []
            self.depth_dir_test = []
            self.label_dir_test = []
            self.SUNRGBD2Dseg = h5py.File(SUNRGBD2Dseg_dir, mode='r', libver='latest')

            SUNRGBDMeta = scipy.io.loadmat(SUNRGBDMeta_dir, squeeze_me=True,
                                           struct_as_record=False)['SUNRGBDMeta']
            split = scipy.io.loadmat(allsplit_dir, squeeze_me=True, struct_as_record=False)
            split_train = split['alltrain']

            seglabel = self.SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']

            for i, meta in enumerate(SUNRGBDMeta):
                meta_dir = '/'.join(meta.rgbpath.split('/')[:-2])
                real_dir = meta_dir.replace('/n/fs/sun3d/data', data_dir)
                depth_bfx_path = os.path.join(real_dir, 'hha/' + meta.depthname)
                rgb_path = os.path.join(real_dir, 'image/' + meta.rgbname)

                label_path = os.path.join(real_dir, 'label/label.npy')

                if not os.path.exists(label_path):
                    os.makedirs(os.path.join(real_dir, 'label'), exist_ok=True)
                    label = np.array(self.SUNRGBD2Dseg[seglabel.value[i][0]].value.transpose(1, 0))
                    np.save(label_path, label)

                if meta_dir in split_train:
                    self.img_dir_train = np.append(self.img_dir_train, rgb_path)
                    self.depth_dir_train = np.append(self.depth_dir_train, depth_bfx_path)
                    self.label_dir_train = np.append(self.label_dir_train, label_path)
                else:
                    self.img_dir_test = np.append(self.img_dir_test, rgb_path)
                    self.depth_dir_test = np.append(self.depth_dir_test, depth_bfx_path)
                    self.label_dir_test = np.append(self.label_dir_test, label_path)

            local_file_dir = '/'.join(self.img_dir_train_file.split('/')[:-1])
            if not os.path.exists(local_file_dir):
                os.mkdir(local_file_dir)
            with open(self.img_dir_train_file, 'w') as f:
                f.write('\n'.join(self.img_dir_train))
            with open(self.depth_dir_train_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_train))
            with open(self.label_dir_train_file, 'w') as f:
                f.write('\n'.join(self.label_dir_train))
            with open(self.img_dir_test_file, 'w') as f:
                f.write('\n'.join(self.img_dir_test))
            with open(self.depth_dir_test_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_test))
            with open(self.label_dir_test_file, 'w') as f:
                f.write('\n'.join(self.label_dir_test))

    def __len__(self):
        if self.phase_train:
            return len(self.img_dir_train)
        else:
            return len(self.img_dir_test)

    def __getitem__(self, idx):
        if self.phase_train:
            img_dir = self.img_dir_train
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
            seg_dir = self.seg_dir_train
        else:
            img_dir = self.img_dir_test
            depth_dir = self.depth_dir_test
            label_dir = self.label_dir_test
            seg_dir = self.seg_dir_test

        # image = Image.open(img_dir[idx]).convert('RGB')
        # _label = np.load(label_dir[idx])
        # #
        # # image = self.examples[idx]['image']
        # # _label = self.examples[idx]['label']
        # _label_copy = _label.copy()
        # for k, v in self.id_to_trainid.items():
        #     _label_copy[_label == k] = v
        # label = Image.fromarray(_label_copy.astype(np.uint8))
        #
        # depth = None
        # seg = None
        #
        # if self.cfg.MULTI_MODAL:
        #     depth = Image.open(depth_dir[idx]).convert('RGB')
        #     seg = Image.fromarray((color_label_np(_label_copy, ignore=self.ignore_label).astype(np.uint8)), mode='RGB')
        # elif 'depth' == self.cfg.TARGET_MODAL:
        #     depth = Image.open(depth_dir[idx]).convert('RGB')
        # elif 'seg' == self.cfg.TARGET_MODAL:
        #     seg = Image.fromarray((color_label_np(_label_copy, ignore=self.ignore_label).astype(np.uint8)), mode='RGB')
        #
        # sample = {'image': image, 'depth': depth, 'label': label, 'seg': seg}
        # for key in list(sample.keys()):
        #     if sample[key] is None:
        #         sample.pop(key)
        #
        # if self.transform:
        #     sample = self.transform(sample)

        image = np.load(img_dir[idx])
        _label = np.load(label_dir[idx])
        _label_copy = _label.copy()
        for k, v in self.id_to_trainid.items():
            _label_copy[_label == k] = v

        depth = None
        seg = None
        lab = None

        if self.cfg.MULTI_MODAL:
            # seg = Image.fromarray((color_label_np(_label_copy, ignore=self.ignore_label).astype(np.uint8)), mode='RGB')
            depth_array = np.load(depth_dir[idx])
            depth = Image.fromarray(depth_array, mode='RGB')
            seg_array = np.load(seg_dir[idx])
            seg = Image.fromarray(seg_array, mode='RGB')
        elif 'depth' == self.cfg.TARGET_MODAL:
            depth_array = np.load(depth_dir[idx])
            depth = Image.fromarray(depth_array, mode='RGB')
        elif 'seg' == self.cfg.TARGET_MODAL:
            seg_array = np.load(seg_dir[idx])
            seg = Image.fromarray(seg_array, mode='RGB')
        elif 'lab' == self.cfg.TARGET_MODAL:
            lab = color.rgb2lab(image)

        image = Image.fromarray(image, mode='RGB')
        label = Image.fromarray(_label_copy.astype(np.uint8))

        sample = {'image': image, 'depth': depth, 'label': label, 'seg': seg, 'lab': lab}
        for key in list(sample.keys()):
            if sample[key] is None:
                sample.pop(key)
        if self.phase_train:
            return self.transform_tr(sample)
        elif not self.phase_train:
            return self.transform_val(sample)


        # if self.transform:
        #     sample = self.transform(sample)

        # return sample
    def transform_tr(self, sample):
        train_transforms = list()
        train_transforms.append(tr.Resize(self.cfg.LOAD_SIZE))
        train_transforms.append(tr.RandomCrop(self.cfg.FINE_SIZE, pad_if_needed=True, fill=0))
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
        if not self.cfg.SLIDE_WINDOWS:
            val_transforms.append(tr.CenterCrop(self.cfg.FINE_SIZE))
        if self.cfg.MULTI_SCALE:
            val_transforms.append(tr.MultiScale(size=self.cfg.FINE_SIZE,scale_times=self.cfg.MULTI_SCALE_NUM, ms_targets=self.ms_targets))
        val_transforms.append(tr.ToTensor())
        val_transforms.append(tr.Normalize(mean=self.cfg.MEAN, std=self.cfg.STD, ms_targets=self.ms_targets))
        composed_transforms = transforms.Compose(val_transforms)


        return composed_transforms(sample)