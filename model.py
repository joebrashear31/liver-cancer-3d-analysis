import nibabel as nib
import matplotlib.pyplot as plt

# Load a NIfTI file - load the file
img = nib.load("data/raw/volume-10.nii")
img_data = img.get_fdata()

# Plot a middle slice - plot it
slice_idx = img_data.shape[2] // 2
plt.imshow(img_data[:, :, slice_idx], cmap='gray')
plt.title(f"Slice {slice_idx}")
plt.axis('off')
plt.show()
