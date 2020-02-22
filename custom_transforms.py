import torch
import random
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import PIL.ImageEnhance as ImageEnhance
from skimage import color
import torchvision.datasets as dataset
import skimage.transform
from tqdm import trange
from PIL import Image, ImageOps, ImageFilter

# class Normalize(object):
#     """Normalize a tensor image with mean and standard deviation.
#     Args:
#         mean (tuple): means for each channel.
#         std (tuple): standard deviations for each channel.
#     """
#     def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
#         self.mean = mean
#         self.std = std

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         img = np.array(img).astype(np.float32)
#         mask = np.array(mask).astype(np.float32)
#         img /= 255.0
#         img -= self.mean
#         img /= self.std

#         return {'image': img,
#                 'label': mask}


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         img = sample['image']
#         mask = sample['label']
#         img = np.array(img).astype(np.float32).transpose((2, 0, 1))
#         mask = np.array(mask).astype(np.float32)

#         img = torch.from_numpy(img).float()
#         mask = torch.from_numpy(mask).float()

#         return {'image': img,
                # 'label': mask} 


# class RandomHorizontalFlip(object):
#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         if random.random() < 0.5:
#             img = img.transpose(Image.FLIP_LEFT_RIGHT)
#             mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

#         return {'image': img,
#                 'label': mask}


# class RandomRotate(object):
#     def __init__(self, degree):
#         self.degree = degree

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         rotate_degree = random.uniform(-1*self.degree, self.degree)
#         img = img.rotate(rotate_degree, Image.BILINEAR)
#         mask = mask.rotate(rotate_degree, Image.NEAREST)

#         return {'image': img,
#                 'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        sample={'image': image,'label': label}
        return sample


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size[0] * 0.5), int(self.base_size[1] * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < min(self.crop_size[0],self.crop_size[1]):
            padh = self.crop_size[0] - oh if oh < self.crop_size[0] else 0
            padw = self.crop_size[1] - ow if ow < self.crop_size[1] else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size[0])
        y1 = random.randint(0, h - self.crop_size[1])
        img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        mask = mask.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        sample={'image': img,'label': mask}
        return sample


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size[0]
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size[1]
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size[0]) / 2.))
        y1 = int(round((h - self.crop_size[1]) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        mask = mask.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        sample={'image': img, 'label': mask}
        return sample

# class FixedResize(object):
#     def __init__(self, size):
#         self.size = (size, size)  # size: (h, w)

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']

#         assert img.size == mask.size 

#         img = img.resize(self.size, Image.BILINEAR)
#         mask = mask.resize(self.size, Image.NEAREST)

#         return {'image': img,
#                 'label': mask}
#######################################################################
#######################################################################

class Resize(transforms.Resize):

    def __call__(self, sample):

        for key in sample.keys():
            if not isinstance(sample[key], Image.Image):
                continue
            if key == 'label':
                sample[key] = F.resize(sample[key], self.size, interpolation=Image.NEAREST)
                continue 
            sample[key] = F.resize(sample[key], self.size, interpolation=Image.BICUBIC)

        return sample


class RandomCrop(transforms.RandomCrop):

    def __call__(self, sample):
        img = sample['image']

        # i, j, h, w = self.get_params(image, self.size)

        # if self.padding > 0:
        #     img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        for key in sample.keys():
            # sample[key] = super().__call__(sample[key])
            if not isinstance(sample[key], Image.Image):
                continue
            sample[key] = F.crop(sample[key], i, j, h, w)

        return sample


class RandomCrop_Unaligned(transforms.RandomCrop):

    def __call__(self, sample):
        img = sample['image']

        # i, j, h, w = self.get_params(image, self.size)

        # if self.padding > 0:
        #     img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        for key in sample.keys():
            i, j, h, w = self.get_params(img, self.size)
            # sample[key] = super().__call__(sample[key])
            if not isinstance(sample[key], Image.Image):
                continue
            sample[key] = F.crop(sample[key], i, j, h, w)

        return sample


class CenterCrop(transforms.CenterCrop):

    def __call__(self, sample):
        for key in sample.keys():
            if not isinstance(sample[key], Image.Image):
                continue
            sample[key] = F.center_crop(sample[key], self.size)

        return sample


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __call__(self, sample):
        if random.random() > 0.5:
            for key in sample.keys():
                if not isinstance(sample[key], Image.Image):
                    continue
                sample[key] = F.hflip(sample[key])

        return sample


class RandomScale(object):

    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image = sample['image']
        target_scale = random.uniform(self.scale_low, self.scale_high)
        target_h = int(round(target_scale * image.size[1]))
        target_w = int(round(target_scale * image.size[0]))

        for key in sample.keys():
            if not isinstance(sample[key], Image.Image):
                continue
            if key == 'label':
                sample['label'] = F.resize(sample['label'], (target_h, target_w), interpolation=Image.NEAREST)
                continue
            sample[key] = F.resize(sample[key], (target_h, target_w), interpolation=Image.BICUBIC)
        return sample


class MultiScale(object):

    def __init__(self, size, scale_times=5, ms_targets=[]):
        assert ms_targets
        self.ms_targets = ms_targets
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale_times = scale_times

    def __call__(self, sample):
        h = self.size[0]
        w = self.size[1]

        for key in self.ms_targets:
            if key not in sample.keys():
                raise ValueError('multiscale keys not in sample keys!!!')
            item = sample[key]
            sample[key] = [F.resize(item, (int(h / pow(2, i)), int(w / pow(2, i))), interpolation=Image.BICUBIC) for i in range(self.scale_times)]

        return sample


class RandomRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate=[-10, 10]):
        self.rotate = rotate
        self.p = 0.5

    def __call__(self, sample):
        if random.random() < self.p:
            image, label = sample['image'], sample['label']
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            for key in sample.keys():
                if not isinstance(sample[key], Image.Image):
                    continue
                if key == 'label':
                    sample['label'] = label.rotate(angle, Image.NEAREST)
                    continue
                sample[key] = sample[key].rotate(angle, Image.BILINEAR)
        return sample


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness > 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if not contrast is None and contrast > 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if not saturation is None and saturation > 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, sample):
        image = sample['image']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        image = ImageEnhance.Brightness(image).enhance(r_brightness)
        image = ImageEnhance.Contrast(image).enhance(r_contrast)
        image = ImageEnhance.Color(image).enhance(r_saturation)
        sample['image'] = image
        return sample


class ToTensor(object):
    def __init__(self, ms_targets=[]):
        self.ms_targets = ms_targets

    def __call__(self, sample):

        single_targets = list(set(sample.keys()) ^ set(self.ms_targets))

        for key in self.ms_targets:
            sample[key] = [F.to_tensor(item) for item in sample[key]]
        for key in single_targets:
            if not isinstance(sample[key], Image.Image) and key != 'lab':
                continue
            if key == 'label':
                label = sample['label']
                _label = np.maximum(np.array(label, dtype=np.int32), 0)
                sample['label'] = torch.from_numpy(_label).long()
                # label = sample['label']
                #
                # sample['label2'] = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2),
                #                                   order=0, mode='reflect', preserve_range=True)
                # sample['label3'] = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4),
                #                                   order=0, mode='reflect', preserve_range=True)
                # sample['label4'] = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
                #                                   order=0, mode='reflect', preserve_range=True)
                # sample['label5'] = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16),
                #                                   order=0, mode='reflect', preserve_range=True)
                continue
            sample[key] = F.to_tensor(sample[key])

        return sample


class Normalize(transforms.Normalize):

    def __init__(self, mean, std, ms_targets=[]):
        super().__init__(mean, std)
        self.ms_targets = ms_targets

    def __call__(self, sample):

        single_targets = list(set(sample.keys()) ^ set(self.ms_targets))

        for key in self.ms_targets:
            sample[key] = [F.normalize(item, self.mean, self.std) for item in sample[key]]
        for key in single_targets:
            if key == 'label':
                continue
            sample[key] = F.normalize(sample[key], self.mean, self.std)

        return sample


class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""

    def __call__(self, sample):
        image = sample['image']
        img = np.asarray(image, np.uint8)
        lab = color.rgb2lab(img)
        sample['lab'] = lab

        return sample
