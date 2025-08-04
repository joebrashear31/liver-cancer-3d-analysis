
import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

# Importing dataset.
# Inherits Dataset object from pytorch utils data
class LiverTumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Load image and mask using nibabel
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Normalize image intensities (clip to [0, 400] and scale to [0, 1])
        image = np.clip(image, 0, 400) / 400.0

        # Convert to torch tensors and add channel dimensions
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
        mask = torch.tensor(mask, dtype=torch.long)                   # (D, H, W)

        # Resize both to the same fixed shape (e.g., 128x128x128)
        target_shape = (128, 128, 128)

        # Resize image using trilinear interpolation
        image = F.interpolate(image.unsqueeze(0), size=target_shape, mode='trilinear', align_corners=False).squeeze(0)

        # Resize mask using nearest-neighbor to preserve class labels
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=target_shape, mode='nearest').squeeze(0).long()

        # Apply optional transforms
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
