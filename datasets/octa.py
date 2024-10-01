from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate
import os
import numpy as np
from PIL import Image
from monai.transforms import RandSpatialCropd

import typing
if typing.TYPE_CHECKING:
    from jaxtyping import Float

class Octa500Dataset(Dataset):
    def __init__(self, img_dir, artery_dir, vein_dir, rotation_correction=True, augmentation=False, max_samples=-1):
        self.augmentation = augmentation
        self.crop = RandSpatialCropd(keys=["img", "seg"], roi_size=(128, 128), random_size=False, random_center=True)

        imgs = []
        labels = []

        # For each image in image folder
        for img_file in os.listdir(img_dir):
            # Load image
            img = Image.open(img_dir + "/" + img_file)
            # Convert to numpy array
            img = np.array(img)
            # Rotate image by 90 degrees
            if rotation_correction:
                img = np.rot90(img, 1)
            # Convert to torch tensor
            img = torch.from_numpy(img.copy()).float()
            # Normalize image
            img = img / 255 - 0.5
            # Append to list
            imgs.append(img.unsqueeze(0))

            # Load artery label
            artery_label = Image.open(artery_dir + "/" + img_file)
            # Load vein label
            vein_label = Image.open(vein_dir + "/" + img_file)

            # Convert both to numpy
            artery_label = np.array(artery_label)
            vein_label = np.array(vein_label)

            # Combine both labels to single numpy array with background in 0, artery in 1 and vein in 2
            label = torch.zeros(artery_label.shape)
            label[artery_label == 255] = 1
            label[vein_label == 255] = 2

            # Convert to one-hot encoding
            label = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=3)

            # Convert to channel first
            label = label.permute(2, 0, 1)

            # Append to list
            labels.append(label)

        # Convert lists to torch tensors
        self.imgs = torch.stack(imgs)
        self.labels = torch.stack(labels)

        if max_samples > 0:
            self.imgs = self.imgs[:max_samples]
            self.labels = self.labels[:max_samples]
        
    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.imgs)
        
    def __getitem__(self, idx) -> dict[str, Float[torch.Tensor, "channel *spatial_dimensions"]]:
        # Retrieve and preprocess the sample at the given index
        # Return the preprocessed sample and its corresponding label
        img = self.imgs[idx]
        seg = self.labels[idx]
        data = {"img": img, "seg": seg}

        # Random crop 128x128
        #data = self.crop(data)
        
        if self.augmentation:
            # Generate random rotation angle from [90, 180, 270]
            angle = torch.randint(1, 4, (1,)).item() * 90
            data["img"] = rotate(data["img"], angle)
            data["seg"] = rotate(data["seg"], angle)
        
        return data
