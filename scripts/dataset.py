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
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
        self.target_shape = target_shape
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Normalize image intensity
        image = np.clip(image, 0, 400) / 400.0

        # Find bounding box around any nonzero region (liver + tumor)
        bbox = find_objects(mask > 0)
        if not bbox:
            print(f"[Warning] Empty mask for file: {self.mask_files[idx]} — using full volume")
            cropped_image = image
            cropped_mask = mask
        else:
            z_slice, y_slice, x_slice = bbox[0]
            cropped_image = image[z_slice, y_slice, x_slice]
            cropped_mask = mask[z_slice, y_slice, x_slice]

        # Convert to torch tensors and add channel dimension
        cropped_image = torch.tensor(cropped_image, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
        cropped_mask = torch.tensor(cropped_mask, dtype=torch.long).unsqueeze(0)       # (1, D, H, W)

        # Pad to at least target shape before downsampling
        pad_D = max(0, self.target_shape[0] - cropped_image.shape[1])
        pad_H = max(0, self.target_shape[1] - cropped_image.shape[2])
        pad_W = max(0, self.target_shape[2] - cropped_image.shape[3])

        if pad_D > 0 or pad_H > 0 or pad_W > 0:
            pad_dims = (0, pad_W, 0, pad_H, 0, pad_D)  # (W_left, W_right, H_left, H_right, D_left, D_right)
            cropped_image = F.pad(cropped_image, pad_dims, mode='constant', value=0)
            cropped_mask = F.pad(cropped_mask, pad_dims, mode='constant', value=0)

        # Downsample crop to target shape
        cropped_image = F.interpolate(cropped_image.unsqueeze(0), size=self.target_shape, mode='trilinear', align_corners=False).squeeze(0)
        cropped_mask = F.interpolate(cropped_mask.float().unsqueeze(0), size=self.target_shape, mode='nearest').squeeze(0).long()

        # Remove channel from mask
        cropped_mask = cropped_mask.squeeze(0)

        print(f"Mask values for {self.mask_files[idx]}: {torch.unique(cropped_mask)}")

        unique_vals, counts = torch.unique(cropped_mask, return_counts=True)
        print(f"Mask stats for {self.mask_files[idx]}:")
        for val, count in zip(unique_vals.tolist(), counts.tolist()):
            print(f"  Value {val}: {count} voxels")


        if self.transform:
            cropped_image, cropped_mask = self.transform(cropped_image, cropped_mask)

        return cropped_image, cropped_mask
