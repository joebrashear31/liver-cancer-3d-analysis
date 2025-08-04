# Import libs
import os
import torch
import nibabel as nib
import numpy as np
from scripts.unet3d import UNet3D

# --- Configuration ---
input_path = 'data/inference/sample_volume'
output_path = 'outputs/sample_prediction.nii.gz'
checkpoint_path = 'models/unet3d_checkpoint.pth'
output_classes = 3  # background, liver, tumor

# --- Load and preprocess image ---
img = nib.load(input_path)
img_data = img.get_fdata()

# Normalize and reshape
img_data = np.clip(img_data, 0, 400) / 400.0
img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

# --- Load model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3D(in_channels=1, out_channels=output_classes).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# --- Inference ---
with torch.no_grad():
    img_tensor = img_tensor.to(device)
    output = model(img_tensor)
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# --- Save prediction as NIfTI ---
pred_nifti = nib.Nifti1Image(pred.astype(np.uint8), affine=img.affine)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
nib.save(pred_nifti, output_path)

print(f"Saved predicted mask to: {output_path}")
