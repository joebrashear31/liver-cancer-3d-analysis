# 🧠 3D Liver Tumor Segmentation with Deep Learning

## Overview
This project implements a 3D U-Net architecture for automatic segmentation of liver and liver tumors from volumetric CT scans. It is designed for medical imaging research applications, particularly in cancer diagnosis and treatment planning. The model is trained and evaluated using a subset of the [LiTS (Liver Tumor Segmentation Challenge)](https://competitions.codalab.org/competitions/17094) dataset.

---

## 🚀 Objectives
- Develop a robust 3D deep learning pipeline for volumetric image segmentation
- Preprocess, normalize, and visualize CT volumes in NIfTI format
- Train a 3D U-Net model with Dice and CrossEntropy loss functions
- Evaluate model performance with Dice coefficient and overlay visualizations
- Enable future integration with clinical tools and research workflows

---

## 🏗️ Architecture
- **Model**: 3D U-Net with skip connections for high-resolution segmentation
- **Input**: 3D CT scans (`128×128×128` voxels)
- **Output**: Voxel-wise labels (background, liver, tumor)
- **Framework**: PyTorch + MONAI for medical deep learning

---

## 📁 Project Structure
```
3D_Liver_Tumor_Segmentation/
├── data/            # Raw and processed CT/NIfTI files
├── scripts/         # Preprocessing, model, and training scripts
├── models/          # Saved model checkpoints
├── outputs/         # Predictions, logs, and metrics
├── train.py         # Main training script
├── inference.py     # Run inference on new scans
└── README.md        # Project documentation
```

---

## 🧪 Key Features
- Efficient preprocessing of volumetric CT scans using `nibabel` and `SimpleITK`
- Medical-specific augmentations using `MONAI`
- Real-time visualization of predictions using `matplotlib` and `napari`
- Modular design for rapid experimentation and reproducibility

---

## 🔧 Tech Stack
- Python, PyTorch, MONAI, SimpleITK, NiBabel, matplotlib
- Hardware: NVIDIA GPU recommended for training 3D volumes

---

## 📚 Future Work
- Add vessel segmentation and spatial lesion analysis
- Integrate model into a research pipeline or clinical viewer (3D Slicer, OHIF)
- Explore self-supervised pretraining for medical data

---

## 👨‍🔬 About Me
I’m a software engineer transitioning into cancer research and AI imaging. This project demonstrates my hands-on ability to build medical AI tools that contribute to real-world oncology applications. I’m particularly interested in applying ML to radiology, surgery, and clinical decision support.

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

