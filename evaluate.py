import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import matplotlib.pyplot as plt

from scripts.unet3d import UNet3D  # your 3D U-Net
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Deterministic eval dataset (no randomness) ----------
class EvalLiverTumorDataset(Dataset):
    """
    Deterministic crop for evaluation:
      - Find tumor bbox (mask==2). If none, use liver (mask==1). If none, center.
      - Crop around bbox with margin and pad as needed.
      - Optionally resize to target_shape.
    """
    def __init__(self, image_dir, mask_dir, target_shape=(96,96,96), margin=10):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
        self.mask_files  = sorted([f for f in os.listdir(mask_dir)  if f.endswith(".nii") or f.endswith(".nii.gz")])
        self.target_shape = target_shape
        self.margin = margin
        assert len(self.image_files) == len(self.mask_files), "Image/mask count mismatch"

    def __len__(self):
        return len(self.image_files)

    def _deterministic_crop(self, img, msk):
        # priority: tumor (2) -> liver (1) -> center
        def bbox_from_label(label_val):
            idx = np.where(msk == label_val)
            if len(idx[0]) == 0:
                return None
            zmin, zmax = idx[0].min(), idx[0].max()
            ymin, ymax = idx[1].min(), idx[1].max()
            xmin, xmax = idx[2].min(), idx[2].max()
            return [zmin, zmax, ymin, ymax, xmin, xmax]

        bbox = bbox_from_label(2)
        if bbox is None:
            bbox = bbox_from_label(1)

        D, H, W = img.shape
        if bbox is None:
            # center crop
            center = (D//2, H//2, W//2)
            tz, ty, tx = self.target_shape
            z0 = max(0, center[0] - tz//2); z1 = min(D, z0 + tz)
            y0 = max(0, center[1] - ty//2); y1 = min(H, y0 + ty)
            x0 = max(0, center[2] - tx//2); x1 = min(W, x0 + tx)
        else:
            zmin, zmax, ymin, ymax, xmin, xmax = bbox
            # expand by margin
            z0 = max(0, zmin - self.margin); z1 = min(D, zmax + self.margin + 1)
            y0 = max(0, ymin - self.margin); y1 = min(H, ymax + self.margin + 1)
            x0 = max(0, xmin - self.margin); x1 = min(W, xmax + self.margin + 1)

        img_c = img[z0:z1, y0:y1, x0:x1]
        msk_c = msk[z0:z1, y0:y1, x0:x1]

        # pad to at least target_shape, then resize exactly to target_shape
        tz, ty, tx = self.target_shape
        pad_z = max(0, tz - img_c.shape[0])
        pad_y = max(0, ty - img_c.shape[1])
        pad_x = max(0, tx - img_c.shape[2])
        if pad_z or pad_y or pad_x:
            pad = ((0,pad_z),(0,pad_y),(0,pad_x))
            img_c = np.pad(img_c, pad, mode='constant', constant_values=0)
            msk_c = np.pad(msk_c, pad, mode='constant', constant_values=0)

        # to tensors (C,D,H,W) for image, (D,H,W) for mask
        img_t = torch.from_numpy(np.clip(img_c,0,400)/400.0).float().unsqueeze(0)  # (1,D,H,W)
        msk_t = torch.from_numpy(msk_c.astype(np.int64))                           # (D,H,W)

        # resize to exact target (trilinear for image, nearest for mask)
        img_t = F.interpolate(img_t.unsqueeze(0), size=self.target_shape, mode='trilinear', align_corners=False).squeeze(0)
        msk_t = F.interpolate(msk_t.unsqueeze(0).unsqueeze(0).float(), size=self.target_shape, mode='nearest').squeeze(0).squeeze(0).long()

        return img_t, msk_t

    def __getitem__(self, idx):
        img = nib.load(os.path.join(self.image_dir, self.image_files[idx])).get_fdata()
        msk = nib.load(os.path.join(self.mask_dir,  self.mask_files[idx])).get_fdata()
        img_t, msk_t = self._deterministic_crop(img, msk)
        return {"image": img_t, "mask": msk_t, "name": self.image_files[idx]}

# ---------- Dice per class ----------
def dice_per_class(logits, target, num_classes=3):
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=num_classes).permute(0,4,1,2,3).float()
    dims = (0,2,3,4)
    inter = torch.sum(probs * one_hot, dims)
    denom = torch.sum(probs + one_hot, dims) + 1e-5
    return (2*inter/denom).cpu().numpy()

# ---------- Evaluate + visualize ----------
def evaluate_and_visualize(
    image_dir="data/images",
    mask_dir="data/masks",
    checkpoint="checkpoints/unet3d_stage2_best.pth",
    target_shape=(96,96,96),
    margin=10,
    max_cases=8
):
    ds = EvalLiverTumorDataset(image_dir, mask_dir, target_shape=target_shape, margin=margin)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model = UNet3D(in_channels=1, out_channels=3).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    model.eval()

    dice_sum = np.zeros(3)
    n = 0

    with torch.no_grad():
        for i, batch in enumerate(dl):
            img = batch["image"].to(device)       # (1,1,D,H,W)
            msk = batch["mask"].to(device)        # (1,D,H,W)
            name = batch["name"][0]

            logits = model(img)
            dice_c = dice_per_class(logits, msk, num_classes=3)
            dice_sum += dice_c
            n += 1

            # ----- Visualization (choose a slice with signal) -----
            # pick slice with max tumor voxels; if none, max liver; else middle
            m = msk[0].cpu().numpy()
            tumor_area = (m==2).sum(axis=(1,2))
            if tumor_area.max() > 0:
                z = int(tumor_area.argmax())
            else:
                liver_area = (m==1).sum(axis=(1,2))
                z = int(liver_area.argmax()) if liver_area.max() > 0 else m.shape[0]//2

            img_np  = img[0,0].cpu().numpy()
            pred_np = torch.argmax(logits, dim=1)[0].cpu().numpy()

            if i < max_cases:
                fig, axs = plt.subplots(1,3, figsize=(12,4))
                axs[0].imshow(img_np[z], cmap="gray"); axs[0].set_title("CT Slice"); axs[0].axis("off")
                axs[1].imshow(m[z], vmin=0, vmax=2);   axs[1].set_title("Ground Truth Mask"); axs[1].axis("off")
                axs[2].imshow(pred_np[z], vmin=0, vmax=2); axs[2].set_title("Predicted Mask"); axs[2].axis("off")
                fig.suptitle(f"{name} | Dice (bg, liver, tumor): {dice_c.round(3)}")
                plt.tight_layout(); plt.show()

    mean_dice = dice_sum / max(n,1)
    print(f"\nMean Dice over {n} cases -> Background: {mean_dice[0]:.3f}, Liver: {mean_dice[1]:.3f}, Tumor: {mean_dice[2]:.3f}")

if __name__ == "__main__":
    evaluate_and_visualize()
