from __future__ import annotations

import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image, ImageSequence


import typing
if typing.TYPE_CHECKING:
    from jaxtyping import Float

class PlateletDataset(Dataset):
    def __init__(
            self, 
            img_file: str, 
            label_file: str, 
            frame_ids: list[int], 
            augmentation: bool=False, 
            max_samples: int=-1,
            patch_width: int=200,
            patch_height: int=200
        ):
        self.augmentation = augmentation

        imgs: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        img_slices = Image.open(img_file)
        label_slices = Image.open(label_file)

        # Read all frames
        img_slices = [np.array(frame) for frame in ImageSequence.Iterator(img_slices)]
        seg_slices = [np.array(frame) for frame in ImageSequence.Iterator(label_slices)]

        # Min-Max Normalization for each slice
        img_slices = [(slice - np.min(slice)) / (np.max(slice) - np.min(slice)) - 0.5 for slice in img_slices]

        id = 0
        
        # Iterate over all frames
        for img_slice, seg_slice in zip(img_slices, seg_slices):
            # Skip frames not in frame_ids
            if len(frame_ids) > 0 and id not in frame_ids:
                id += 1
                continue

            # Create overlapping patches of specified size from the images and labels with a stride of 30
            for i in range(0, img_slice.shape[0] - patch_width, 100):
                for j in range(0, img_slice.shape[1] - patch_height, 100):
                    img_patch = img_slice[i:i+patch_width, j:j+patch_height]
                    seg_patch = seg_slice[i:i+patch_width, j:j+patch_height]

                    # Convert to one-hot encoding with class dim first
                    seg_patch = torch.nn.functional.one_hot(torch.from_numpy(seg_patch.astype(np.int64)), num_classes=7)
                    seg_patch = seg_patch.permute(2, 0, 1)

                    # Add to list
                    imgs.append(torch.from_numpy(img_patch).float())
                    labels.append(seg_patch)

            id += 1

        # Limit the number of samples
        if max_samples > 0:
            imgs = imgs[:max_samples]
            labels = labels[:max_samples]

        # Convert to torch tensor
        self.imgs = torch.stack(imgs)
        self.labels = torch.stack(labels)

        # Add channel dim
        self.imgs = self.imgs.unsqueeze(1)

                        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx) -> dict[str, Float[torch.Tensor, "channel *spatial_dimensions"]]:
        img = self.imgs[idx]
        seg = self.labels[idx]
        if self.augmentation:
            # rotate image
            k = np.random.randint(4)
            img = torch.rot90(img, k=k, dims=(1, 2))
            seg = torch.rot90(seg, k=k, dims=(1, 2))
        
        pair = {"img": img, "seg": seg}
        
        return pair
