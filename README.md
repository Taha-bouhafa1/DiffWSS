# DiffWSS

### Diffusion-Guided Weakly Supervised Semantic Segmentation of High-Resolution Aerial Images

DiffWSS is a research project investigating the application of diffusion-guided representations to Weakly Supervised Semantic Segmentation (WSSS) of high-resolution aerial imagery.

The project adapts the methodology introduced in **Diffusion-Guided Weakly Supervised Semantic Segmentation (DiG, ECCV 2024)** to the ISPRS Potsdam dataset.

The main objective is to investigate whether diffusion models can improve semantic localization and pseudo-label generation when only image-level annotations are available, reducing the need for expensive pixel-level annotations.

---

## Dataset

The project uses the **ISPRS Potsdam 2D Semantic Labeling Dataset**, which contains high-resolution aerial images with pixel-level annotations.

### Dataset Characteristics

- Number of scenes: 38
- Original image resolution: 6000 × 6000 pixels
- Spatial resolution: 5 cm
- Number of semantic classes: 6
- Image format: RGB TIFF
- Annotation format: RGB TIFF masks

### Semantic Classes

| ID | Class | RGB Color |
|---|---|---|
| 0 | Impervious surfaces | (255, 255, 255) |
| 1 | Building | (0, 0, 255) |
| 2 | Low vegetation | (0, 255, 255) |
| 3 | Tree | (0, 255, 0) |
| 4 | Car | (255, 255, 0) |
| 5 | Clutter / Background | (255, 0, 0) |

### Preprocessing

The original 6000 × 6000 images are divided into non-overlapping 512 × 512 patches.

- Patch size: 512 × 512
- Stride: 512
- Padding: Images are padded to 6144 × 6144.
- Patches per scene: 144
- Total patches: 5,472

### Dataset Split

The dataset is divided at the scene level to prevent spatial leakage between training, validation, and testing samples.

| Split | Scenes | Patches |
|---|---:|---:|
| Training | 26 | 3,744 |
| Validation | 6 | 864 |
| Testing | 6 | 864 |
| **Total** | **38** | **5,472** |

Image-level class labels are generated from the training annotations to simulate weak supervision.

Pixel-level annotations are reserved for validation and testing in the weakly supervised experiments.

---

## Methodology

The project adapts the Diffusion-Guided Weakly Supervised Semantic Segmentation (DiG) framework to high-resolution aerial imagery.

The methodology consists of two main stages.

### Stage 1: Diffusion Model Training

A Denoising Diffusion Probabilistic Model (DDPM) is trained on unlabeled Potsdam training images.

The diffusion model learns visual representations of the target aerial imagery and provides intermediate features that can be used during the subsequent weakly supervised learning stage.

**DDPM Configuration:**

- Architecture: U-Net
- Diffusion timesteps: 1,000
- Noise schedule: Sigmoid
- Prediction objective: Velocity prediction (`pred_v`)
- Optimizer: Adam
- Learning rate: 1e-4
- Mixed precision: FP16
- EMA: Exponential Moving Average

### Stage 2: Diffusion-Guided Weakly Supervised Semantic Segmentation

The second stage adapts the DiG methodology by combining Vision Transformer representations with diffusion-model features.

It incorporates two important components:

**1. Locality Fusion Cross Attention (LFCA)**

LFCA integrates semantic representations from the Vision Transformer with intermediate features extracted from the DDPM U-Net.

Through cross-attention, diffusion features provide complementary visual information that helps improve semantic localization and generate Diffusion-CAMs.

**2. Patch Affinity Consistency (PAC)**

PAC encourages consistency between representations extracted from the original input images and their diffusion-reconstructed counterparts.

By exploiting the relationship between patch-level representations and affinities, PAC helps improve the consistency of semantic localization.

### Pseudo-Label Generation

The localization maps are refined to generate pseudo-labels for semantic segmentation training.

These pseudo-labels provide pixel-level supervision for the downstream segmentation model.

### Semantic Segmentation

DeepLabV3+ is used as the downstream semantic segmentation architecture to learn pixel-level predictions from the generated pseudo-labels.

---

## Implementation

The project is implemented using Python and PyTorch.

### Main Technologies

- Python
- PyTorch
- Vision Transformers (ViT)
- Denoising Diffusion Probabilistic Models (DDPM)
- DeepLabV3+
- ISPRS Potsdam Dataset

### Hardware

Diffusion model training is performed using GPU acceleration.

---

## Repository Structure

```text
DiffWSS/
│
├── configs/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│
├── outputs/
│
├── scripts/
│   ├── tiling.py
│   └── weak_labels.py
│
├── README.md
└── .gitignore
```

---

## Research Reference

This project is based on the following research:

**Diffusion-Guided Weakly Supervised Semantic Segmentation**

ECCV 2024.

- Paper: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06482.pdf
- Official implementation: https://github.com/yoon307/DiG

The original framework is adapted to the ISPRS Potsdam dataset to investigate diffusion-guided weakly supervised semantic segmentation in high-resolution aerial imagery.

---

## Project Status

Research and implementation in progress.