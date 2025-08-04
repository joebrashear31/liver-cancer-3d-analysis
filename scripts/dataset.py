import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

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

        # Load volumes
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Normalize intensity
        image = np.clip(image, 0, 400) / 400.0

        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)       # (1, D, H, W)

        # Downsample
        image = F.interpolate(image.unsqueeze(0), size=self.target_shape, mode='trilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.float().unsqueeze(0), size=self.target_shape, mode='nearest').squeeze(0)

        # Remove channel dim from mask to make it (D, H, W)
        mask = mask.squeeze(0).long()

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
