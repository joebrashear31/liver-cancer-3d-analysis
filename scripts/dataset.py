import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.ndimage import find_objects

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

        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Normalize image
        image = np.clip(image, 0, 400) / 400.0

        image = torch.tensor(image, dtype=torch.float32)  # (D, H, W)
        mask = torch.tensor(mask, dtype=torch.long)       # (D, H, W)

        # Target shape
        target_shape = (96, 96, 96)
        D, H, W = target_shape

        # Find tumor bounding box
        bbox = find_objects((mask > 0).numpy())
        if not bbox:
            # fallback: center crop (or skip by raising error)
            center = [s // 2 for s in mask.shape]
        else:
            bbox = bbox[0]
            center = [(s.start + s.stop) // 2 for s in bbox]

        # Compute start and end slices
        start = [max(0, c - t // 2) for c, t in zip(center, target_shape)]
        end = [min(start[i] + target_shape[i], mask.shape[i]) for i in range(3)]
        start = [end[i] - target_shape[i] for i in range(3)]  # adjust start if near edge

        # Crop
        image = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        mask = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        # Add channel dimension to image
        image = image.unsqueeze(0)  # shape: (1, D, H, W)

        return image, mask
