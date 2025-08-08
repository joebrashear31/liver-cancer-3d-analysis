import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scripts.unet3d import UNet3D
from scripts.dataset import LiverTumorPatchDataset
import matplotlib.pyplot as plt

def compute_class_weights(dataset, num_classes=3, max_samples=1000):
    counter = Counter()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, (image, mask) in enumerate(loader):
        counter.update(mask.flatten().tolist())
        if i >= max_samples:
            break

    total = sum(counter.values())
    weights = []
    for i in range(num_classes):
        freq = counter[i] / total if i in counter else 1e-6
        weights.append(1.0 / (freq + 1e-6))
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights / weights.sum()


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

        intersection = torch.sum(inputs * targets_one_hot)
        union = torch.sum(inputs + targets_one_hot)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class CombinedLoss(torch.nn.Module):
    def __init__(self, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss()
        self.ce = torch.nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, inputs, targets):
        loss_dice = self.dice(inputs, targets)
        loss_ce = self.ce(inputs, targets)
        return loss_dice + loss_ce

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
