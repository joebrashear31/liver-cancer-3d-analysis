import os
import nibabel as nib
import matplotlib.pyplot as plt

image_path = "data/images/volume-0.nii.gz"
mask_path = "data/masks/segmentation-0.nii.gz"

# Load the volumes
image = nib.load(image_path).get_fdata()
mask = nib.load(mask_path).get_fdata()

# Choose a slice index (e.g., middle of the volume)
slice_idx = image.shape[2] // 2
img_slice = image[:, :, slice_idx]
mask_slice = mask[:, :, slice_idx]

# Plot side-by-side
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_slice, cmap='gray')
plt.title('CT Slice')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_slice, cmap='gray')
plt.imshow(mask_slice, cmap='jet', alpha=0.4)
plt.title('CT + Segmentation Mask Overlay')
plt.axis('off')

plt.tight_layout()
plt.show()
