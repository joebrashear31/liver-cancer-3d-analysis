import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scripts.unet3d import UNet3D
from scripts.dataset import LiverTumorPatchDataset
import matplotlib.pyplot as plt

# Metric used to predict the mask based on ground truth
# Basically testing if the model generated
# accurate masks to separate tumors and liver tissue
# formula - Dice = 2 * |A inter B| / (|A| + |B|)
# finding the TP intersects divided by the total
def dice_score(pred, target, num_classes=3):
    dice = []
    pred = F.one_hot(pred, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    target = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    for c in range(1, num_classes):  # Skip background
        inter = (pred[:, c] * target[:, c]).sum()
        union = pred[:, c].sum() + target[:, c].sum()
        score = (2. * inter) / (union + 1e-8)
        dice.append(score.item())
    return dice

# --- Config ---
image_dir = 'data/images'
mask_dir = 'data/masks'
checkpoint_path = 'models/unet3d_checkpoint.pth'
batch_size = 1

# --- Data Loading ---
dataset = LiverTumorPatchDataset(image_dir=image_dir, mask_dir=mask_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# --- Load Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3D(in_channels=1, out_channels=3).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# --- Evaluation ---
all_dice = []

with torch.no_grad():
    for i, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        dice = dice_score(preds, masks)
        all_dice.append(dice)

        # Visualization: slice from center
        slice_idx = images.shape[2] // 2
        img_slice = images[0, 0, :, :, slice_idx].cpu().numpy()
        mask_slice = masks[0, :, :, slice_idx].cpu().numpy()
        pred_slice = preds[0, :, :, slice_idx].cpu().numpy()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img_slice, cmap='gray')
        plt.title('CT Slice')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask_slice, cmap='jet', alpha=0.6)
        plt.title('Ground Truth Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pred_slice, cmap='jet', alpha=0.6)
        plt.title('Predicted Mask')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# --- Results ---
all_dice = torch.tensor(all_dice)
mean_dice = all_dice.mean(dim=0)
for i, score in enumerate(mean_dice):
    print(f"Mean Dice Score - Class {i+1}: {score:.4f}")
