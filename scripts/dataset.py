
import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

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

        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Normalize image to [0, 1]
        image = np.clip(image, 0, 400) / 400.0

        # Convert to torch tensors and add channel dimension
        # Tensor in this case is a 3D vector way to 
        # represent the image
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # shape: (1, D, H, W)
        mask = torch.tensor(mask, dtype=torch.long)                   # shape: (D, H, W)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
