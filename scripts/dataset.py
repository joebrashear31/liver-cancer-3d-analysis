import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class LiverTumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Load image and mask
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Normalize image
        image = np.clip(image, 0, 400) / 400.0

        # Convert to tensors and add channel dimension
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
        mask = torch.tensor(mask, dtype=torch.long)  # (D, H, W)

        # Resize both to smallest manageable 3D shape
        target_shape = (64, 64, 64)

        image = F.interpolate(image.unsqueeze(0), size=target_shape, mode='trilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=target_shape, mode='nearest')
        mask = mask.squeeze(0).squeeze(0).long()

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
