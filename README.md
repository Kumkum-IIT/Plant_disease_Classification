# Plant Disease Classification using EfficientNet-B3

## Overview

This project implements a deep learning-based plant disease classification system using **EfficientNet-B3** and **PyTorch**. The model is trained on a dataset containing 38 plant disease categories and healthy plant classes, with the objective of accurately identifying diseases from leaf images.

The solution employs transfer learning, mixed-precision training, extensive data augmentation, and a two-stage fine-tuning strategy to achieve high classification accuracy while maintaining efficient training times.

---

## Features

* Multi-class plant disease classification (38 classes)
* Transfer learning with EfficientNet-B3
* Two-stage training strategy
* Mixed Precision Training (AMP)
* Advanced image augmentation using RandAugment
* Efficient inference pipeline
---

## Dataset

The dataset contains images of plant leaves categorized into 38 classes representing various diseases and healthy conditions.

### Dataset Statistics

* Training Images: 43,429
* Test Images: 10,876
* Classes: 38

Example Classes:

* Apple Scab
* Apple Black Rot
* Cedar Apple Rust
* Healthy Apple
* Healthy Blueberry
* Tomato Early Blight
* Tomato Late Blight
* Tomato Mosaic Virus
* Tomato Yellow Leaf Curl Virus
* Healthy Tomato

---

## Model Architecture

### Backbone

* EfficientNet-B3 (`tf_efficientnet_b3`)
* Pretrained on ImageNet

### Model Details

| Parameter    | Value           |
| ------------ | --------------- |
| Architecture | EfficientNet-B3 |
| Parameters   | 10.75 Million   |
| Input Size   | 256 × 256       |
| Classes      | 38              |

---

## Data Augmentation

The following augmentations are applied during training:

* Random Resized Crop
* Horizontal Flip
* Vertical Flip
* RandAugment
* Image Normalization

### Training Transform

```python
RandomResizedCrop(256)
RandomHorizontalFlip()
RandomVerticalFlip()
RandAugment(num_ops=2, magnitude=7)
Normalize()
```

---

## Training Strategy

### Stage 1: Classifier Head Training

The EfficientNet backbone is frozen, and only the classification head is trained.

**Configuration**

* Epochs: 3
* Learning Rate: 1e-3
* Optimizer: AdamW
* Label Smoothing: 0.05

#### Results

| Epoch | Accuracy |
| ----- | -------- |
| 1     | 86.58%   |
| 2     | 93.55%   |
| 3     | 95.03%   |

---

### Stage 2: Full Fine-Tuning

The entire model is unfrozen and fine-tuned.

**Configuration**

* Epochs: 8
* Learning Rate: 3e-5
* Optimizer: AdamW
* Scheduler: CosineAnnealingLR

#### Results

| Epoch | Accuracy |
| ----- | -------- |
| 1     | 97.53%   |
| 2     | 98.75%   |
| 3     | 99.23%   |
| 4     | 99.44%   |
| 5     | 99.57%   |
| 6     | 99.65%   |
| 7     | 99.61%   |
| 8     | 99.73%   |

---

## Technologies Used

### Deep Learning

* PyTorch
* Torchvision
* TIMM

### Data Processing

* Pandas
* PIL
* NumPy

### Utilities

* TQDM
* CUDA AMP
* Multi-GPU Training

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/plant-disease-classification.git

cd plant-disease-classification
```

### Install Dependencies

```bash
pip install torch torchvision
pip install timm==1.0.3
pip install pandas pillow tqdm
```

---

## Project Structure

```text
plant-disease-classification/
│
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── ...
│
├── test/
│   ├── image_00001.jpg
│   ├── image_00002.jpg
│   └── ...
│
├── train.py
├── inference.py
├── submission.csv
├── plant_disease_weights.pth
├── plant_disease_checkpoint.pth
└── README.md
```

---

## Model Saving

The project saves:

### Model Weights

```text
plant_disease_weights.pth
```

### Metadata Checkpoint

```text
plant_disease_checkpoint.pth
```

Checkpoint contains:

* Model Name
* Image Size
* Number of Classes

---

## Inference

Run inference on the test dataset:

```python
model.eval()

with torch.no_grad():
    outputs = model(images)
    predictions = outputs.argmax(dim=1)
```

The generated predictions are stored in:

```text
submission.csv
```

Format:

```csv
Image_ID,Label
image_00001,0
image_00002,0
image_00003,5
```

---

## Performance Optimizations

* Mixed Precision Training (AMP)
* Transfer Learning
* Cosine Learning Rate Scheduling
* Label Smoothing
* RandAugment
* Multi-GPU Training
* EfficientNet Backbone

---

## Results

### Final Training Accuracy

**99.73%**

### Dataset Coverage

* 38 Plant Disease Classes
* 43K+ Training Images
* 10K+ Test Images

### Advantages

* Fast convergence
* High accuracy
* Lightweight model
* Efficient deployment

---

## Future Improvements

* K-Fold Cross Validation
* Test Time Augmentation (TTA)
* Ensemble Models
* Vision Transformers (ViT)
* ConvNeXt Backbones
* Real-time Mobile Deployment
* Disease Severity Estimation

---

## Author

Developed using PyTorch and EfficientNet-B3 for large-scale plant disease classification and agricultural AI applications.
