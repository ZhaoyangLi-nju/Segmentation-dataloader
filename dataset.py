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
class Seg_Cityscapes(torch.utils.data.Dataset):
    def __init__(self, cfg=None, data_dir=None, phase_train=True):
        train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
                      "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
                      "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
                      "bremen/", "bochum/", "aachen/"]
        val_dirs = ["frankfurt/", "munster/", "lindau/"]
        test_dirs = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]
        self.id_to_trainid = {-1: -1, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1,
                              9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255,
                              30: 255, 31: 16, 32: 17, 33: 18}
        # self.class_weights = [2.5965083, 6.7422495, 3.5350077, 9.866795, 9.691752,
        #                       9.369563, 10.289785, 9.954636, 4.308077, 9.491024, 7.6707582, 9.395554,
        #                       10.3475065, 6.3950195, 10.226835, 10.241277, 10.280692, 10.396961, 10.05563]
        self.class_weights = None
        self.cfg = cfg
        self.ignore_label = 255
        self.ms_targets=[]
        if phase_train:
            self.split = "train"
            file_dir = train_dirs
        else:
            self.split = "val"
            file_dir = val_dirs
        self.img_dir = data_dir + "/leftImg8bit/" + self.split + "/"
        self.label_dir = data_dir + "/gtFine/" + self.split + "/"
        self.examples = []
        # count=0
        for train_dir in file_dir:
            train_img_dir_path = self.img_dir + train_dir
            file_names = os.listdir(train_img_dir_path)
            for file_name in file_names:
                if 't.npy' not in file_name:
                    continue
                img_id = file_name.split("_leftImg8bit.npy")[0]
                img_path = train_img_dir_path + file_name
                label_img_path = self.label_dir + train_dir + img_id + "_gtFine_labelIds.npy"
                example = {}
                seg_path_npy = label_img_path.split("_gtFine_labelIds.npy")[0] + "_seg.npy"

                example["img_path"] = img_path
                example["label_path"] = label_img_path
                example["seg_path"] = seg_path_npy

                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)
        # if phase_train and self.class_weights == None:
        #     self.class_weights = self.getClassWeight(self.examples)
        #     print(self.class_weights)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        example = self.examples[index]
        img_path = example["img_path"]
        label_path = example["label_path"]
        seg_path = example["seg_path"]

        # image = Image.open(img_path).convert('RGB')
        image = np.load(img_path)
        label = np.load(label_path)

        label_copy = label.copy()
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = Image.fromarray(label_copy.astype(np.uint8))
        image = Image.fromarray(image.astype(np.uint8), mode='RGB')

        seg = None

        if self.cfg.NO_TRANS == False:
            if 'seg' == self.cfg.TARGET_MODAL:
                # seg = Image.fromarray((color_label_np(label_copy, ignore=self.ignore_label).astype(np.uint8)), mode='RGB')
                seg = np.load(seg_path)
                seg = Image.fromarray(seg.astype(np.uint8), mode='RGB')
            sample = {'image': image, 'label': label, 'seg': seg}
        else:
            # print(image.size)
            sample = {'image': image, 'label': label}
        for key in list(sample.keys()):
            if sample[key] is None:
                sample.pop(key)
        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)


    def transform_tr(self, sample):
        train_transforms = list()
        train_transforms.append(tr.RandomScale(self.cfg.RANDOM_SCALE_SIZE))
        train_transforms.append(tr.RandomRotate())
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
        if not self.cfg.SLIDE_WINDOWS:
            val_transforms.append(tr.CenterCrop(self.cfg.FINE_SIZE))
        if self.cfg.MULTI_SCALE:
            val_transforms.append(tr.MultiScale(size=self.cfg.FINE_SIZE,scale_times=self.cfg.MULTI_SCALE_NUM, ms_targets=self.ms_targets))
        val_transforms.append(tr.ToTensor())
        val_transforms.append(tr.Normalize(mean=self.cfg.MEAN, std=self.cfg.STD, ms_targets=self.ms_targets))
        composed_transforms = transforms.Compose(val_transforms)

        return composed_transforms(sample)
    # def getClassWeight(self, example):
    #     trainId_to_count = {}
    #     class_weights = []
    #     num_classes = 19
    #     for trainId in range(num_classes):
    #         trainId_to_count[trainId] = 0
    #     for step, samples in enumerate(example):
    #         if step % 100 == 0:
    #             print(step)
    #
    #         # label_img = cv2.imread(label_img_path, -1)
    #         label = np.load(samples["label_img_path"])
    #         label_copy = label.copy()
    #         for k, v in self.id_to_trainid.items():
    #             label_copy[label == k] = v
    #         label_img = Image.fromarray(label_copy.astype(np.uint8))
    #         for trainId in range(num_classes):
    #             # count how many pixels in label_img which are of object class trainId:
    #             trainId_mask = np.equal(label_img, trainId)
    #             trainId_count = np.sum(trainId_mask)
    #             # add to the total count:
    #             trainId_to_count[trainId] += trainId_count
    #     total_count = sum(trainId_to_count.values())
    #     for trainId, count in trainId_to_count.items():
    #         trainId_prob = float(count) / float(total_count)
    #         trainId_weight = 1 / np.log(1.02 + trainId_prob)
    #         class_weights.append(trainId_weight * 0.1)
    #     return class_weights
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

class Rec_NYUD2(Rec_SUNRGBD):
    def __init__(self, cfg, data_dir=None, transform=None, phase_train=None):
        super().__init__(cfg, data_dir, transform, phase_train)

# class Rec_MIT67(Rec_SUNRGBD):
#     def __init__(self, cfg, data_dir=None, transform=None, phase_train=None):
#         super().__init__(cfg, data_dir, transform, phase_train)

class Rec_MIT67(Rec_SUNRGBD):

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        sample = {'image': img, 'label': label}
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


class Rec_PLACES(Rec_SUNRGBD):

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        sample = {'image': img, 'label': label}
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

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)
    return images
