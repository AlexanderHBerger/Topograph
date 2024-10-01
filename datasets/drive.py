import os
from PIL import Image
import torch
import torchvision
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TVF
from torch.utils.data import Dataset
import random

class Drive(Dataset):
    def __init__(self, data_path, crop_size=100, max_samples=-1, five_crop=True, rescale=1, augment=False):
        img_dir = os.path.join(data_path, "images")
        label_dir = os.path.join(data_path, "1st_manual")
        mask_dir = os.path.join(data_path, "mask")
        self.crop_size = crop_size
        self.five_crop = five_crop
        self.augment = augment

        self.img_names = list(sorted(os.listdir(img_dir)))
        self.label_names = list(sorted(os.listdir(label_dir)))
        self.mask_names = list(sorted(os.listdir(mask_dir)))

        if max_samples > 0:
            self.img_names = self.img_names[:max_samples]
            self.label_names = self.label_names[:max_samples]
            self.mask_names = self.mask_names[:max_samples]
        
        transform_list = [v2.ToDtype(torch.float32, scale=True)]
        target_transform_list = [v2.ToDtype(torch.float32, scale=False)]
        
        if five_crop:
            transform_list.append(v2.FiveCrop(crop_size))
            target_transform_list.append(v2.FiveCrop(crop_size))

        self.transforms = v2.Compose(transform_list)
        self.target_transforms = v2.Compose(target_transform_list)
        
        self.images = []
        self.labels = []
        self.masks = []
        for img_name, label_name, mask_name in zip(self.img_names, self.label_names, self.mask_names):
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, label_name)
            mask_path = os.path.join(mask_dir, mask_name)
            
            image = torchvision.io.read_image(img_path)
            label = (torchvision.io.read_image(label_path, mode=torchvision.io.ImageReadMode.GRAY) > 127) * 1
            mask = (torchvision.io.read_image(mask_path, mode=torchvision.io.ImageReadMode.GRAY) > 127) * 1
            
            h, w = image.shape[-2:]

            if rescale != 1 and (h//(rescale+1) < crop_size or w//(rescale+1) < crop_size):
                continue

            image = TVF.resize(image, min(h, w) // rescale, antialias=True)
            label = TVF.resize(label, min(h, w) // rescale, antialias=True)

            self.images.append(image)
            self.labels.append(torch.stack((1-label, label), dim=0).squeeze(dim=1))
            self.masks.append(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        mask = self.masks[idx]

        im_h, im_w = image.shape[-2:]

        # RandomCrop
        if self.augment:
            crop_size = self.crop_size
            if self.five_crop:
                crop_size *= 2
            i, j, h, w = v2.RandomCrop.get_params(
                image, output_size=(min(crop_size, im_h), min(crop_size, im_w)))
            image = TVF.crop(image, i, j, h, w)
            label = TVF.crop(label, i, j, h, w)
            mask = TVF.crop(mask, i, j, h, w)
            
            #rotate
            angle = random.choice([0, 90, 180, 270])
            image = TVF.rotate(image, angle)
            label = TVF.rotate(label, angle)
            mask = TVF.rotate(mask, angle)
        elif not self.five_crop:
            image = TVF.center_crop(image, self.crop_size)
            label = TVF.center_crop(label, self.crop_size)
            mask = TVF.center_crop(mask, self.crop_size)

        image = self.transforms(image)
        label = self.target_transforms(label)
        mask = self.target_transforms(mask)
        if self.five_crop:
            image = torch.stack(image, dim=0)
            label = torch.stack(label, dim=0)
            mask = torch.stack(mask, dim=0)
        return {"img": image, "seg": label, "mask": mask}