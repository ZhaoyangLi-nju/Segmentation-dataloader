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