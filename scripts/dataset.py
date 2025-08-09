import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import os
import random

class LiverTumorPatchDataset(Dataset):
    # reduce the patch size around a tumor area
    def __init__(self, image_dir, mask_dir, patch_size=(32, 64, 64), num_patches=20):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.image_filenames = [f for f in os.listdir(image_dir) if f.startswith("volume") and f.endswith(".nii.gz")]
        self.image_filenames.sort()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        mask_filename = image_filename.replace("volume", "segmentation")

        image_path = os.path.join(self.image_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        patch_img, patch_mask = self._random_patch(image, mask)

        patch_img = torch.from_numpy(patch_img).float().unsqueeze(0)  # [1, D, H, W], shape of 3d Image for model
        patch_mask = torch.from_numpy(patch_mask).long()
        patch_mask = torch.clamp(patch_mask, min=0, max=2)  # ensure class range [0,1,2]

        return patch_img, patch_mask

    def _random_patch(self, image, mask):
        d, h, w = image.shape
        pd, ph, pw = self.patch_size
        d_start = random.randint(0, d - pd)
        h_start = random.randint(0, h - ph)
        w_start = random.randint(0, w - pw)

        patch_img = image[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        patch_mask = mask[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        return patch_img, patch_mask
