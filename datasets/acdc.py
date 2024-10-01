from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import nibabel as nib

from monai.transforms import RandCropByLabelClassesd, CenterSpatialCropd


import typing
if typing.TYPE_CHECKING:
    from jaxtyping import Float

class ACDC_ShortAxisDataset(Dataset):
    def __init__(
            self, 
            img_dir: str, 
            patient_ids: list[int], 
            mean: float = -1, 
            std: float = -1,
            rand_crop: bool = False,
            augmentation: bool=False, 
            max_samples: int=-1, 
            resize=False):
        """
        ACDC_ShortAxis dataset class for loading and preprocessing ACDC short-axis cardiac MRI images and labels.

        Args:
            img_dir (str): The directory path where the image files are stored.
            patient_ids (list[int]): List of patient IDs to include in the dataset.
            mean (float, optional): Mean value for image normalization. If not provided, it will be calculated from the images. Must be provided for validation and test. Defaults to -1.
            std (float, optional): Standard deviation value for image normalization. If not provided, it will be calculated from the images. Must be provided for validation and test. Defaults to -1.
            augmentation (bool, optional): Flag indicating whether to apply data augmentation. Defaults to False.
            max_samples (int, optional): Maximum number of samples to include in the dataset. Defaults to -1 (include all samples).
            resize (bool, optional): Flag indicating whether to resize the images to 154x154. Defaults to False.

        Raises:
            NotImplementedError: Raised if augmentation is set to True (not implemented yet).
        """
        self.augmentation = augmentation
        self.resize = resize

        imgs = []
        labels = []



        for i, id in enumerate(patient_ids):
            # convert id to string with three digits
            id = str(id).zfill(3)

            # iterate through every image in image directory
            for img_file in os.listdir(os.path.join(img_dir, f"patient{id}")):
                # if name ends with .nii.gz and does not contain _gt and does not contain 4d, it is an image
                if img_file.endswith(".nii.gz") and "_gt" not in img_file and "4d" not in img_file:
                    # Load image
                    img = nib.load(f"{img_dir}/patient{id}/{img_file}").get_fdata()
                    # Load label
                    label = nib.load(f"{img_dir}/patient{id}/{img_file.split('.')[0]}_gt.nii.gz").get_fdata()
                    # Convert to torch tensor
                    img = torch.from_numpy(img).float()
                    label = torch.from_numpy(label).float()

                    # Crop image and label to 154x154xnum_slices
                    if rand_crop:
                        crop = RandCropByLabelClassesd(
                            keys=["img", "label"],
                            label_key="label",
                            spatial_size=(154, 154, img.shape[2]),
                            ratios=[0, 1, 1, 1],
                            num_classes=4,
                            num_samples=3,
                        )
                        data = crop({"img": img.unsqueeze(0), "label": label.unsqueeze(0)})
                    else:
                        crop = CenterSpatialCropd(
                            keys=["img", "label"], 
                            roi_size=(154, 154, img.shape[2])
                        )
                        data = [crop({"img": img.unsqueeze(0), "label": label.unsqueeze(0)})]


                    # Append each crop and slice to list
                    for crop in data:
                        img = crop["img"][0]
                        label = crop["label"][0]

                        # Move slices dim to the front
                        img = img.permute(2, 0, 1)
                        label = label.permute(2, 0, 1)

                        # convert to 3d tensor to list of 2d slices
                        for i in range(img.shape[0]):
                            img_slice = img[i]
                            label_slice = label[i]

                            # Convert label to int
                            label_slice = label_slice.long()

                            # Convert to one-hot encoding with class dim first
                            label_slice = torch.nn.functional.one_hot(label_slice, num_classes=4)
                            label_slice = label_slice.permute(2, 0, 1)

                            # Put into dataset
                            imgs.append(img_slice)
                            labels.append(label_slice)

        # Convert lists to torch tensors
        self.imgs = torch.stack(imgs)
        self.labels = torch.stack(labels)

        if max_samples > 0:
            self.imgs = self.imgs[:max_samples]
            self.labels = self.labels[:max_samples]

        # Normalize images (if mean and std are not given, calculate them from the images. This should only be done for the training set)
        if mean == -1:
            self.mean = self.imgs.mean().item()
        else:
            self.mean = mean
        if std == -1:
            self.std = self.imgs.std().item()
        else:
            self.std = std

        self.imgs = (self.imgs - self.mean) / self.std

        # Add channel dim
        self.imgs = self.imgs.unsqueeze(1)
        
    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.imgs)
    
    def get_mean_std(self) -> tuple[float, float]:
            """
            Returns the mean and standard deviation of the dataset.

            Returns:
                tuple[float, float]: A tuple containing the mean and standard deviation.
            """
            return self.mean, self.std
        
    def __getitem__(self, idx) -> dict[str, Float[torch.Tensor, "channel *spatial_dimensions"]]:
        """
        Retrieve and preprocess the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict[str, Float[torch.Tensor, "channel *spatial_dimensions"]]: A dictionary containing the preprocessed sample and its corresponding label.
        """
        pair = {"img": self.imgs[idx], "seg": self.labels[idx]}
        if self.augmentation:
            raise NotImplementedError("Data augmentation is not implemented yet.")
        
        # resize by padding the borders to 160x160
        if self.resize:
            pair["img"] = torch.nn.functional.pad(pair["img"], pad=(3, 3, 3, 3), mode="constant", value=0)
            pair["seg"] = torch.nn.functional.pad(pair["seg"], pad=(3, 3, 3, 3), mode="constant", value=0)
            pair["seg"][0,-3:3, -3:3] = 1
        
        return pair
