import os
from PIL import Image
import torch
import torchvision
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TVF
from torch.utils.data import Dataset
import random

class Roads(Dataset):
    def __init__(self, data_path, crop_size=100, max_samples=-1, normalize=True, five_crop=True, rescale=1, augment=False):
        img_dir = os.path.join(data_path, "images")
        label_dir = os.path.join(data_path, "labels")
        self.crop_size = crop_size
        self.five_crop = five_crop
        self.augment = augment

        self.img_names = list(sorted(os.listdir(img_dir)))
        self.label_names = list(sorted(os.listdir(label_dir)))

        if max_samples > 0:
            self.img_names = self.img_names[:max_samples]
            self.label_names = self.label_names[:max_samples]
        
        transform_list = [v2.ToDtype(torch.float32, scale=True)]
        target_transform_list = [v2.ToDtype(torch.float32, scale=False)]

        if normalize:
            transform_list.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        if five_crop:
            transform_list.append(v2.FiveCrop(crop_size))
            target_transform_list.append(v2.FiveCrop(crop_size))

        self.transforms = v2.Compose(transform_list)
        self.target_transforms = v2.Compose(target_transform_list)
        
        self.images = []
        self.labels = []
        for img_name, label_name in zip(self.img_names, self.label_names):
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, label_name)
            
            image = torchvision.io.read_image(img_path)
            label = (torchvision.io.read_image(label_path, mode=torchvision.io.ImageReadMode.GRAY) > 127)*1
            
            h, w = image.shape[-2:]

            if rescale != 1 and (h//(rescale+1) < crop_size or w//(rescale+1) < crop_size):
                continue

            image = TVF.resize(image, min(h, w) // rescale, antialias=True)
            label = TVF.resize(label, min(h, w) // rescale, antialias=True)

            im_h, im_w = image.shape[-2:]

            if not self.augment:
                # transform the image into 4 parts (for each corner) and add them to the list
                # left top corner
                image1 = TVF.crop(image, 0, 0, crop_size, crop_size)
                label1 = TVF.crop(label, 0, 0, crop_size, crop_size)
                # right top corner
                image2 = TVF.crop(image, 0, im_w - crop_size, crop_size, crop_size)
                label2 = TVF.crop(label, 0, im_w - crop_size, crop_size, crop_size)
                # left bottom corner
                image3 = TVF.crop(image, im_h - crop_size, 0, crop_size, crop_size)
                label3 = TVF.crop(label, im_h - crop_size, 0, crop_size, crop_size)
                # right bottom corner
                image4 = TVF.crop(image, im_h - crop_size, im_w - crop_size, crop_size, crop_size)
                label4 = TVF.crop(label, im_h - crop_size, im_w - crop_size, crop_size, crop_size)

                self.images.extend([image1, image2, image3, image4])
                self.labels.extend([torch.stack((1-label1, label1), dim=0).squeeze(dim=1), torch.stack((1-label2, label2), dim=0).squeeze(dim=1), torch.stack((1-label3, label3), dim=0).squeeze(dim=1), torch.stack((1-label4, label4), dim=0).squeeze(dim=1)])


            else:
                self.images.append(image)
                self.labels.append(torch.stack((1-label, label), dim=0).squeeze(dim=1))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

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
            
            #rotate
            angle = random.choice([0, 90, 180, 270])
            image = TVF.rotate(image, angle)
            label = TVF.rotate(label, angle)
        elif not self.five_crop:
            image = TVF.center_crop(image, self.crop_size)
            label = TVF.center_crop(label, self.crop_size)

        image = self.transforms(image)
        label = self.target_transforms(label)
        if self.five_crop:
            image = torch.stack(image, dim=0)
            label = torch.stack(label, dim=0)
        return {"img": image, "seg": label}