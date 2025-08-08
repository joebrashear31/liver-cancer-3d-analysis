import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import nibabel as nib
from scripts.unet3d import UNet3D
from scripts.dataset import TumorOnlyPatchDataset, LiverTumorPatchDataset

# ================= CONFIG ================= #
train_val_split = 0.8

# Stage 1: Tumor-only
stage1_epochs = 50
stage1_patch_size = (32, 32, 32)
stage1_batch_size = 2
stage1_lr = 1e-4

# Stage 2: Full segmentation
stage2_epochs = 100
stage2_patch_size = (64, 64, 64)
stage2_batch_size = 1
stage2_lr = 1e-4
tumor_patch_ratio = 0.7

# Early stopping
early_stopping_patience = 10  # epochs without improvement before stopping

image_dir = "data/images"
mask_dir = "data/masks"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Calculate the loss by finding the intersectioin
# between ground truth and prediction.
# ================= LOSSES ================= #
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        probs = torch.softmax(logits, dim=1)
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice_per_class = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice_per_class.mean()

# Calculate the combined loss
class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, dice_weight=0.5, ce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
    def forward(self, logits, targets):
        return self.dice_weight * self.dice(logits, targets) + self.ce_weight * self.ce(logits, targets)

# ================= UTILS ================= #
def compute_class_weights(dataset, num_classes):
    counts = np.zeros(num_classes)
    for _, mask in dataset:
        vals, freqs = np.unique(mask.numpy(), return_counts=True)
        for v, f in zip(vals, freqs):
            counts[v] += f
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float)

def dice_score_per_class(logits, targets, num_classes):
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    probs = torch.softmax(logits, dim=1)
    dims = (0, 2, 3, 4)
    intersection = torch.sum(probs * targets_one_hot, dims)
    cardinality = torch.sum(probs + targets_one_hot, dims)
    return ((2. * intersection) / (cardinality + 1e-5)).cpu().numpy()

def evaluate_model(model, loader, num_classes):
    model.eval()
    dice_scores = np.zeros(num_classes)
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice_scores += dice_score_per_class(outputs, masks, num_classes)
    return dice_scores / len(loader)

def train_stage(model, train_loader, val_loader, loss_fn, optimizer, scaler, epochs, num_classes, stage_name, csv_path, ckpt_path):
    best_dice = 0
    best_epoch = 0
    epochs_no_improve = 0

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['epoch', 'train_loss'] + [f'dice_class_{i}' for i in range(num_classes)]
        writer.writerow(header)

        for epoch in range(epochs):
            model.train()
            running_loss = 0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                with autocast(enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = loss_fn(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            val_dice = evaluate_model(model, val_loader, num_classes)

            writer.writerow([epoch+1, avg_loss] + list(val_dice))
            print(f"{stage_name} Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Dice: {val_dice}")

            mean_dice = np.mean(val_dice)
            if mean_dice > best_dice:
                best_dice = mean_dice
                best_epoch = epoch + 1
                torch.save(model.state_dict(), ckpt_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1} for {stage_name}")
                break

    print(f"Best {stage_name} Epoch: {best_epoch} | Best Val Dice: {best_dice:.4f}")

# ================= MAIN ================= #
if __name__ == "__main__":
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ---- Stage 1: Tumor-only ----
    stage1_dataset = TumorOnlyPatchDataset(image_dir, mask_dir, patch_size=stage1_patch_size)
    train_len = int(len(stage1_dataset) * train_val_split)
    val_len = len(stage1_dataset) - train_len
    stage1_train, stage1_val = random_split(stage1_dataset, [train_len, val_len])

    stage1_train_loader = DataLoader(stage1_train, batch_size=stage1_batch_size, shuffle=True)
    stage1_val_loader = DataLoader(stage1_val, batch_size=1)

    class_weights_stage1 = compute_class_weights(stage1_train, num_classes=2).to(device)
    model = UNet3D(in_channels=1, out_channels=2).to(device)
    loss_fn_stage1 = CombinedLoss(class_weights=class_weights_stage1)
    optimizer_stage1 = optim.Adam(model.parameters(), lr=stage1_lr)

    print("Starting Stage 1 (Tumor-only)...")
    train_stage(model, stage1_train_loader, stage1_val_loader, loss_fn_stage1, optimizer_stage1, scaler,
                stage1_epochs, 2, "Stage 1", "logs/metrics_stage1.csv", "checkpoints/unet3d_stage1_best.pth")

# ---- Stage 2: Full 3-class ----
stage2_dataset = LiverTumorPatchDataset(
    image_dir, mask_dir,
    patch_size=stage2_patch_size,
    tumor_patch_ratio=tumor_patch_ratio
)
train_len = int(len(stage2_dataset) * train_val_split)
val_len = len(stage2_dataset) - train_len
stage2_train, stage2_val = random_split(stage2_dataset, [train_len, val_len])

stage2_train_loader = DataLoader(stage2_train, batch_size=stage2_batch_size, shuffle=True)
stage2_val_loader = DataLoader(stage2_val, batch_size=1)

class_weights_stage2 = compute_class_weights(stage2_train, num_classes=3).to(device)
class_weights_stage2[2] *= 3.0  # Boost tumor weight

model = UNet3D(in_channels=1, out_channels=3).to(device)

# Load Stage 1 weights except the final out_conv
checkpoint = torch.load("checkpoints/unet3d_stage1_best.pth", map_location=device)
if "out_conv.weight" in checkpoint:
    del checkpoint["out_conv.weight"]
if "out_conv.bias" in checkpoint:
    del checkpoint["out_conv.bias"]

model.load_state_dict(checkpoint, strict=False)

loss_fn_stage2 = CombinedLoss(class_weights=class_weights_stage2)
optimizer_stage2 = optim.Adam(model.parameters(), lr=stage2_lr)

print("Starting Stage 2 (Full 3-class)...")
train_stage(
    model, stage2_train_loader, stage2_val_loader,
    loss_fn_stage2, optimizer_stage2, scaler,
    stage2_epochs, 3, "Stage 2",
    "logs/metrics_stage2.csv",
    "checkpoints/unet3d_stage2_best.pth"
)
   