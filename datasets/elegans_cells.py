import os
import torch
import torchvision
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TVF
from torch.utils.data import Dataset
import random 
import torch.nn.functional as F

class Elegans(Dataset):
    def __init__(self, data_path, crop_size=48, max_samples=-1, rescale=1, augment=False):
        # set: "train" or "val"
        img_dir = os.path.join(data_path, "images")
        label_dir = os.path.join(data_path, "labels")
        self.crop_size = crop_size
        self.augment = augment

        self.img_names = list(sorted(os.listdir(img_dir)))
        self.label_names = list(sorted(os.listdir(label_dir)))

        if max_samples > 0:
            self.img_names = self.img_names[:max_samples]
            self.label_names = self.label_names[:max_samples]

        transform_list = [v2.ToDtype(torch.float32, scale=True)]
        target_transform_list = [v2.ToDtype(torch.float32, scale=False)]

        self.transforms = v2.Compose(transform_list)
        self.target_transforms = v2.Compose(target_transform_list)
        
        self.images = []
        self.labels = []
        for img_name, label_name in zip(self.img_names, self.label_names):
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, label_name)
            
            image = torchvision.io.read_image(img_path)
            label = (torchvision.io.read_image(label_path, mode=torchvision.io.ImageReadMode.GRAY) > 127)*1
            
            label = self.preprocess_label(label)

            h, w = image.shape[-2:]

            image = TVF.resize(image, min(h, w), antialias=True)
            label = TVF.resize(label, min(h, w), antialias=True)

            self.images.append(image)
            self.labels.append(torch.stack((1-label, label), dim=0).squeeze(dim=1))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        im_h, im_w = image.shape[-2:]

        if self.augment:
            # Random Rotate
            angle = random.choice([0, 90, 180, 270])
            image = TVF.rotate(image, angle)
            label = TVF.rotate(label, angle)

            if self.crop_size < im_h:
                # RandomCrop
                i, j, h, w = v2.RandomCrop.get_params(
                    image, output_size=(self.crop_size, self.crop_size))
                image = TVF.crop(image, i, j, h, w)
                label = TVF.crop(label, i, j, h, w)

        image = self.transforms(image)
        label = self.target_transforms(label)
        
        return {"img": image, "seg": label}
    

    def preprocess_label(self, label):
        # Define a 3x3 kernel to check the surrounding pixels
        kernel = torch.tensor([[0, 1, 0],
                               [1, 0, 1],
                               [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Apply the kernel to the label image
        neighbor_count = F.conv2d(label.float(), kernel, padding=1)
        
        # Identify isolated background pixels (value 0) surrounded by 8 foreground pixels (value 8)
        isolated_background = (label == 0) & (neighbor_count == 4)
        
        # Change isolated background pixels to foreground pixels
        label[isolated_background] = 1
        
        return label