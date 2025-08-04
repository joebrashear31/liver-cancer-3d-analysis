# import libraries
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scripts.unet3d import UNet3D
from scripts.dataset import LiverTumorDataset

# --- Configuration ---
image_dir = 'data/images'
mask_dir = 'data/masks'
batch_size = 2
epochs = 10
learning_rate = 1e-4
checkpoint_path = 'models/unet3d_checkpoint.pth'

# --- Data Loading ---
dataset = LiverTumorDataset(image_dir=image_dir, mask_dir=mask_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Model Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3D(in_channels=1, out_channels=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss() # To caculate loss

# --- Training Loop ---
model.train()
for epoch in range(epochs):
    running_loss = 0.0
for i, (images, masks) in enumerate(dataloader):
    images, masks = images.to(device), masks.to(device)

    outputs = model(images)
    loss = criterion(outputs, masks)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

avg_loss = running_loss / len(dataloader)
print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

# Save checkpoint
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
torch.save(model.state_dict(), checkpoint_path)
print(f"Checkpoint saved to: {checkpoint_path}")
