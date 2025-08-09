import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from scripts.dataset import LiverTumorPatchDataset
from scripts.unet3d import UNet3D  # adjust this import based on your model file
import os

# Patch of random numbers used for randomness in the model (like weights)
def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Find the overlap of GT and Predicted, then calculate loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        targets = torch.clamp(targets, 0, num_classes - 1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        probs = torch.softmax(logits, dim=1)
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

def dice_score_per_class(logits, targets, num_classes):
    targets = torch.clamp(targets, 0, num_classes - 1)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    probs = torch.softmax(logits, dim=1)
    dims = (0, 2, 3, 4)
    intersection = torch.sum(probs * targets_one_hot, dims)
    cardinality = torch.sum(probs + targets_one_hot, dims)
    return ((2. * intersection) / (cardinality + 1e-5)).cpu().numpy()

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2
NUM_EPOCHS = 100
LR = 1e-4

# Set seed
seed_everything()

# Load data
train_dataset = LiverTumorPatchDataset("data/images", "data/masks")
val_dataset = LiverTumorPatchDataset("data/images", "data/masks")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Load model
model = UNet3D(in_channels=1, out_channels=3).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = DiceLoss()

# Training loop
# Calculates dice loss, then adjusts weights in the model.
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    dice_total = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = model(images)
            dice_total += dice_score_per_class(outputs, masks, num_classes=3).mean()

    avg_dice = dice_total / len(val_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Dice: {avg_dice:.4f}")
