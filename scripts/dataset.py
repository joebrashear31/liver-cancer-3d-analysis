import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.ndimage import find_objects

class LiverTumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_shape=(96, 96, 96), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.target_shape = target_shape
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        image = np.clip(image, 0, 400) / 400.0  # normalize
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)

        # Find tumor bounding box
        bbox = find_objects((mask > 0).numpy())
        if not bbox:
            center = [s // 2 for s in mask.shape]  # fallback: center
        else:
            bbox = bbox[0]
            center = [(s.start + s.stop) // 2 for s in bbox]

        # Define crop coordinates
        D, H, W = self.target_shape
        start = [max(0, c - s // 2) for c, s in zip(center, self.target_shape)]
        end = [min(start[i] + self.target_shape[i], mask.shape[i]) for i in range(3)]
        start = [max(0, end[i] - self.target_shape[i]) for i in range(3)]

        # Crop
        image = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        mask = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        # Pad if crop is smaller than target shape
        pad_D = max(0, D - image.shape[0])
        pad_H = max(0, H - image.shape[1])
        pad_W = max(0, W - image.shape[2])

        pad_dims = (0, pad_W, 0, pad_H, 0, pad_D)  # (W_left, W_right, H_left, H_right, D_left, D_right)
        image = F.pad(image, pad_dims, mode='constant', value=0)
        mask = F.pad(mask, pad_dims, mode='constant', value=0)

        # Add channel dimension to image
        image = image.unsqueeze(0)  # shape: (1, D, H, W)

        return image, mask
