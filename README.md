---

## 🚀 How to Use This Project

### 1️⃣ Train the Model
Make sure your NIfTI volumes and masks are in the correct folders:
```
data/
├── images/        # CT volumes (e.g., volume-0.nii.gz)
└── masks/         # Ground truth masks (e.g., segmentation-0.nii.gz)
```

Then run:
```bash
python train.py
```

### 2️⃣ Evaluate Model Performance
Evaluate the model on test or validation volumes:
```bash
python evaluate.py
```
This computes Dice scores and shows visual slice comparisons.

### 3️⃣ Run Inference on New Volumes
To generate predictions for a new CT volume:
- Place it at: `data/inference/sample_volume.nii.gz`
- Then run:
```bash
python inference.py
```
Predicted mask will be saved at: `outputs/sample_prediction.nii.gz`

---