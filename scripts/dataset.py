import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random

class TumorOnlyPatchDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=(32, 32, 32)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Normalize intensity
        image = np.clip(image, 0, 400) / 400.0
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)

        # Binary mask: tumor=1, everything else=0
        mask = torch.where(mask == 2, torch.tensor(1), torch.tensor(0))

        tumor_voxels = torch.nonzero(mask == 1, as_tuple=False)
        if len(tumor_voxels) > 0:
            center = tumor_voxels[random.randint(0, len(tumor_voxels) - 1)]
        else:
            all_voxels = torch.nonzero(mask >= 0, as_tuple=False)
            center = all_voxels[random.randint(0, len(all_voxels) - 1)]

        # Extract patch
        ps = self.patch_size
        start = [max(0, center[d].item() - ps[d] // 2) for d in range(3)]
        end = [start[d] + ps[d] for d in range(3)]

        # Pad if needed
        pad_D = max(0, end[0] - image.shape[0])
        pad_H = max(0, end[1] - image.shape[1])
        pad_W = max(0, end[2] - image.shape[2])
        if pad_D > 0 or pad_H > 0 or pad_W > 0:
            image = F.pad(image, (0, pad_W, 0, pad_H, 0, pad_D))
            mask = F.pad(mask, (0, pad_W, 0, pad_H, 0, pad_D))

        patch_img = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        patch_mask = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        # Add channel dimension to image
        patch_img = patch_img.unsqueeze(0)

        print(f"Image shape: {patch_img.shape}")       # (1, D, H, W)
        print(f"Mask shape: {patch_mask.shape}")       # (D, H, W)
        print(f"Unique values in mask: {patch_mask.unique()}")  # Should show 0 and 1

        return patch_img, patch_mask


class LiverTumorPatchDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=(64, 64, 64), tumor_patch_ratio=0.7, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
        self.patch_size = patch_size
        self.tumor_patch_ratio = tumor_patch_ratio
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Normalize intensity
        image = np.clip(image, 0, 400) / 400.0
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)

        # Get voxel coordinates
        tumor_voxels = torch.nonzero(mask == 2, as_tuple=False)
        liver_voxels = torch.nonzero(mask == 1, as_tuple=False)
        all_voxels = torch.nonzero(mask >= 0, as_tuple=False)

        # Decide patch center
        if len(tumor_voxels) > 0 and random.random() < self.tumor_patch_ratio:
            center = tumor_voxels[random.randint(0, len(tumor_voxels) - 1)]
        elif len(liver_voxels) > 0:
            center = liver_voxels[random.randint(0, len(liver_voxels) - 1)]
        else:
            center = all_voxels[random.randint(0, len(all_voxels) - 1)]

        # Extract patch
        ps = self.patch_size
        start = [max(0, center[d].item() - ps[d] // 2) for d in range(3)]
        end = [start[d] + ps[d] for d in range(3)]

        # Pad if needed
        pad_D = max(0, end[0] - image.shape[0])
        pad_H = max(0, end[1] - image.shape[1])
        pad_W = max(0, end[2] - image.shape[2])
        if pad_D > 0 or pad_H > 0 or pad_W > 0:
            image = F.pad(image, (0, pad_W, 0, pad_H, 0, pad_D))
            mask = F.pad(mask, (0, pad_W, 0, pad_H, 0, pad_D))

        patch_img = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        patch_mask = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        # Add channel dimension to image
        patch_img = patch_img.unsqueeze(0)

        if self.transform:
            patch_img, patch_mask = self.transform(patch_img, patch_mask)
        print(f"Image shape: {patch_img.shape}")       # (1, D, H, W)
        print(f"Mask shape: {patch_mask.shape}")       # (D, H, W)
        print(f"Unique values in mask: {patch_mask.unique()}")  # Should show 0 and 1

        return patch_img, patch_mask
