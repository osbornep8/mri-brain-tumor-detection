# Brain Tumor Detection from MRI Images using Deep Learning

A deep learning model for detecting brain tumors in MRI scans using PyTorch and transfer learning with ResNet50.

## Requirements

> See requirements.txt

## Project Structure

brain-tumor-detection/
│
├── data/ # Data directory (not included in repo)
├── models/ # Saved model weights and results
├── notebooks/ # Jupyter notebooks
├── src/ # Source code
│ ├── dataset.py # Dataset class
│ ├── model.py # Model architecture
│ ├── train.py # Training functions
│ ├── evaluate.py # Evaluation functions
│ ├── transforms.py # Data transformations
│ └── utils.py # Utility functions
├── requirements.txt
└── README.md

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/osbornep8/mri-brain-tumor-detection.git
cd mri-brain-tumor-detection
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate
```

3. Install requirements:

```bash
pip install -r requirements.txt
```

## Usage

Just run:

```bash
python src/train.py
```

## Project Overview

This project implements a binary classification model to detect the presence of brain tumors in MRI scans. Using transfer learning with a ResNet50 architecture, the model achieves high precision in tumor detection, making it a potentially valuable tool for medical image analysis.

The model achieves:

- Precision: 0.8571
- Recall: 0.7059
- F1 Score: 0.7742
- ROC AUC Score: 0.7721

## Dataset

The dataset consists of brain MRI images divided into two categories:

- Tumor (155 images)
- No Tumor (98 images)

Source: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

## Model Architecture

- Base Model: ResNet50 (pretrained on ImageNet)
- Custom classification head with dropout for regularization
- Binary classification output (tumor/no-tumor)
- Training with weighted BCE loss to handle class imbalance

## Data Preprocessing & Augmentation

- Resize to 224x224
- Random rotations (±43 degrees)
- Random horizontal and vertical flips
- Shear transformations
- Color jitter (brightness and contrast)
- Normalization using the MRI dataset statistics
- Gaussian noise addition

## Training Process

The model was trained with the following specifications:

- Optimizer: SGD with momentum (0.9) and weight decay (1e-4)
- Learning rate scheduler: ReduceLROnPlateau
- Early stopping with patience of 10 epochs
- Train/Validation/Test split: 80/10/10

## Results

The model achieves promising results in detecting brain tumors from MRI images:

### Confusion Matrix

```
                 Predicted No Tumor  Predicted Tumor
Actual No Tumor      6                2
Actual Tumor         5                12
```

### Performance Metrics

- Precision: 0.8571 (85.71% of tumor predictions were correct)
- Recall: 0.7059 (70.59% of actual tumors were detected)
- F1 Score: 0.7742 (harmonic mean of precision and recall)
- ROC AUC Score: 0.7721 (model's discriminative ability)

### Training Progress

[Training history plot description]

## Model Weights

Trained model weights can be downloaded from the releases section of this repository.

## Contributing

Feel free to open issues or submit pull requests with improvements.

## Acknowledgments

- Dataset provided by Kaggle
- Base implementation using PyTorch