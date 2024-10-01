from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip
import os
import nibabel as nib


import typing
if typing.TYPE_CHECKING:
    from jaxtyping import Float

class TopCowDataset(Dataset):
    def __init__(
            self, 
            img_dir: str, 
            label_dir: str,
            augmentation: bool=False, 
            max_samples: int=-1,
            width: int=100,
            height: int=80
        ):
        """
        Initializes a dataset for TopCow images and labels.

        Args:
            img_dir (str): The directory path containing the image files.
            label_dir (str): The directory path containing the label files.
            augmentation (bool, optional): Whether to apply data augmentation. Defaults to False.
            max_samples (int, optional): The maximum number of samples to include in the dataset. 
                Defaults to -1, which includes all samples.
            width (int, optional): The desired width of the images. Defaults to 100.
            height (int, optional): The desired height of the images. Defaults to 80.
        """
        self.augmentation = augmentation

        imgs: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        # iterate over all files in img_dir
        for img_file in os.listdir(img_dir):
            img = nib.load(os.path.join(img_dir, img_file))
            label = nib.load(os.path.join(label_dir, img_file.replace("img", "seg")))
            img = img.get_fdata()
            label = label.get_fdata()

            # Convert to torch
            img = torch.tensor(img).float()
            label = torch.tensor(label)

            # Rescale to width and height
            img = torch.nn.functional.interpolate(img.unsqueeze(0).unsqueeze(0), size=(width, height)).squeeze()    
            label = torch.nn.functional.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=(width, height)).squeeze().long()

            # Convert to one-hot encoding with class dim first
            label = torch.nn.functional.one_hot(label, num_classes=16).permute(2, 0, 1).float()

            # Add to list
            imgs.append(img)
            labels.append(label)

        # Limit the number of samples
        if max_samples > 0:
            imgs = imgs[:max_samples]
            labels = labels[:max_samples]

        # Convert to torch tensor
        self.imgs = torch.stack(imgs)
        self.labels = torch.stack(labels)

        # Normalize
        self.imgs = (self.imgs - self.imgs.min()) / (self.imgs.max() - self.imgs.min()) - 0.5

        # Add channel dim
        self.imgs = self.imgs.unsqueeze(1)

                        
    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.imgs)

    def __getitem__(self, idx) -> dict[str, Float[torch.Tensor, "channel *spatial_dimensions"]]:
        """
        Returns the image and label pair at the given index.

        Args:
            idx (int): The index of the image and label pair.

        Returns:
            dict[str, Float[torch.Tensor, "channel *spatial_dimensions"]]: A dictionary containing the image and label pair.
        """
        pair = {"img": self.imgs[idx], "seg": self.labels[idx]}
        if self.augmentation:
            # with 50% chance, flip the image horizontally
            if torch.rand(1) > 0.5:
                pair["img"] = hflip(pair["img"])
                pair["seg"] = hflip(pair["seg"])
        
        return pair
