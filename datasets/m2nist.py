from __future__ import annotations

from monai.data import Dataset
import numpy as np
import torch
from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    EnsureTyped,
    EnsureType,
)

import typing
if typing.TYPE_CHECKING:
    from jaxtyping import Float

class M2NIST(Dataset):
    def __init__(self, data_path, augmentation=False, max_samples=-1):
        self.augmentation = augmentation
        self.transform = Compose(
            [
                AddChanneld(keys=["img"]),
                ScaleIntensityd(keys=["img", "seg"]), # doing normalisation here :)
                # RandCropByPosNegLabeld(
                #     keys=["img", "seg"],
                #     label_key="seg",
                #     spatial_size=[-1, config.DATA.IMG_SIZE[0], config.DATA.IMG_SIZE[1]],
                #     pos=1,
                #     neg=1,
                #     num_samples=config.DATA.NUM_PATCH,
                # ),
                #RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
                EnsureTyped(keys=["img", "seg"]),
            ]
        )

        # Load image and ground truth:
        img = np.load(data_path + 'combined.npy')
        img = np.array(img, dtype=float)

        seg = np.load(data_path + 'segmented.npy')
        seg = np.array(seg, dtype=float)

        # Reorder dims such that channel is first
        seg = np.moveaxis(seg, -1, 1)

        self.pairs = [{"img": img[i], "seg": seg[i]} for i in range(len(img))]

        if max_samples > 0:
            self.pairs = self.pairs[:max_samples]
        
    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.pairs)
        
    def __getitem__(self, idx: int) -> dict[str, Float[torch.Tensor, "channel *spatial_dimensions"]]:
        # Retrieve and preprocess the sample at the given index
        # Return the preprocessed sample and its corresponding label
        if self.augmentation:
            # Fail with not implemented
            raise NotImplementedError
        
        pair = self.transform(self.pairs[idx])
        
        return pair