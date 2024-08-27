import albumentations as A
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import glob

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir_rgbs, root_dir_masks, transform=None):
        # Get the path for the images and for the masks
        rgbs = Path(root_dir_rgbs)
        masks = Path(root_dir_masks)
        # Create a list for all images and a list for all masks
        self.image_list = list(rgbs.glob('*'))
        self.mask_list = list(masks.glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        # Get the image path and mask path
        img_path = self.image_list[index]
        mask_path = self.mask_list[index]

        # Read the image and mask
        image = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path))  # Load mask without grayscale conversion

        # Convert a numpy array into a torch tensor for the image and for the mask
        image = torch.from_numpy(image).to(torch.float32) / 255.0  # Normalize the image
        mask = torch.from_numpy(mask).to(torch.float32)  # Convert mask to float tensor

        mask = (mask > (128/255)).float()
        mask = torch.round(mask)

        # Permute the dimensions of the image and for the mask from (0, 1, 2) to (2, 0, 1)
        #                                                           (H, W, C) to (C, H, W)
        # H = height, W = width, C = channels
        image = image.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)

        # Transform the image and the mask if a transform is necessary
        if self.transform:
            augmented = self.transform(image=image.numpy(), mask=mask.numpy())
            image = torch.from_numpy(augmented['image']).permute(2, 0, 1).to(torch.float32)
            mask = torch.from_numpy(augmented['mask']).permute(2, 0, 1).to(torch.float32)

        return image, mask

    def __len__(self):
        return len(self.image_list)
