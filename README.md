# CAVIT-Lung: Explainable CNN–Transformer for Lung Cancer Classification

An explainable deep learning framework for multi-class lung cancer classification using CT scan images. This project combines EfficientNetB0, CBAM Attention, and Vision Transformer (ViT) to extract and learn both local and global image features.

## Overview

CAVIT-Lung is a hybrid deep learning model designed to classify lung CT scan images into three categories:

- **Benign**
- **Malignant**
- **Normal**

The model integrates CNN-based feature extraction with Transformer-based global feature learning to improve classification and provide visual explanations of its predictions.

## Model Architecture

The proposed architecture consists of:

1. **EfficientNetB0** – Extracts deep visual features from CT scan images.
2. **CBAM Attention** – Focuses on important channel-wise and spatial features.
3. **Vision Transformer (ViT)** – Captures global relationships between image regions.
4. **Feature Fusion** – Combines CNN and Transformer representations.
5. **Classification Head** – Predicts one of the three lung CT categories.

## Dataset

**Dataset:** IQ-OTHNCCD Lung Cancer Dataset

The dataset contains CT scan images categorized into:

- Benign cases
- Malignant cases
- Normal cases

Images are resized to 224 × 224 pixels before being passed to the model.

## Data Preprocessing

- Image resizing and normalization
- Data augmentation
- CLAHE contrast enhancement
- Random rotation and flipping
- Noise augmentation
- Class weighting
- Weighted Random Sampling

## Training Strategy

The model uses a two-phase training approach:

- **Phase 1:** Training the classification head while freezing the EfficientNetB0 backbone.
- **Phase 2:** Fine-tuning selected backbone layers along with the remaining model components.

Additional techniques include:

- MixUp
- Label Smoothing
- Exponential Moving Average (EMA)
- Early Stopping
- Test-Time Augmentation (TTA)

## Explainability Features

CAVIT-Lung incorporates multiple explainability techniques:

- **Grad-CAM:** Highlights important image regions influencing predictions.
- **ViT Attention Maps:** Visualizes spatial attention patterns.
- **Attention Consistency Score (ACS):** An additional framework for analyzing prediction consistency.
- **Uncertainty Analysis:** Combines prediction entropy, MC Dropout, and ACS.

## Evaluation Metrics

The model evaluation includes:

- Accuracy
- Precision
- Recall
- Macro F1-Score
- Balanced Accuracy
- Confusion Matrix
- ROC Curves
- Precision-Recall Curves
- Cohen's Kappa
- Matthews Correlation Coefficient (MCC)

## Technologies Used

- Python
- PyTorch
- TensorFlow-independent Deep Learning Pipeline
- OpenCV
- Albumentations
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- TIMM

## Project Structure

```text
CAVIT-Lung/
│
├── checkpoints/
│   └── lung_cancer_hybrid_final.pt
│
├── output_folder/
│   └── The IQ-OTHNCCD Lung Cancer Augmented Dataset/
│
├── Lunge-Cancer-Detection-404_Error.ipynb
│
└── README.md
```

## How to Run

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/CAVIT-Lung.git
```

**2. Navigate to the project directory**

```bash
cd CAVIT-Lung
```

**3. Install dependencies**

```bash
pip install torch torchvision timm
pip install opencv-python albumentations
pip install numpy pandas scikit-learn
pip install matplotlib seaborn scipy
```

**4. Prepare the dataset**

Download the IQ-OTHNCCD Lung Cancer Dataset and organize it according to the directory structure used in the notebook.

**5. Run the notebook**

Open the Jupyter Notebook or Google Colab and execute the cells sequentially.

## Disclaimer

This project is developed for academic and research purposes only. It is not intended to replace professional medical diagnosis or clinical decision-making.

## Author

**Bijoy Sen**

Computer Science and Engineering Graduate

GitHub: https://github.com/bijoy1295

---

⭐ If you find this project interesting, consider giving the repository a star!
