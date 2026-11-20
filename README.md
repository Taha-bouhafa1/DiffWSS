# DiffWSS

### Diffusion-Guided Weakly Supervised Semantic Segmentation of High-Resolution Aerial Images

DiffWSS is a research project investigating the integration of diffusion models into weakly supervised semantic segmentation (WSSS) for high-resolution aerial imagery.

Inspired by **DiG (ECCV 2024)**, the project explores how diffusion-based visual representations and image reconstruction can improve semantic localization and segmentation using image-level supervision.

The framework is adapted to the **ISPRS Potsdam dataset**, focusing on aerial scene understanding and land-cover classification.

## Dataset

The project uses the ISPRS Potsdam aerial image dataset.

| Property | Description |
|---|---|
| Dataset | ISPRS Potsdam |
| Original image resolution | 6000 × 6000 |
| Spatial resolution | 5 cm |
| Number of scenes | 38 |
| Tile size | 512 × 512 |
| Tiling strategy | Non-overlapping |
| Total number of tiles | 5,472 |

### Semantic Classes

The dataset contains six semantic categories:

1. Impervious surfaces
2. Building
3. Low vegetation
4. Tree
5. Car
6. Clutter/background

### Dataset Splits

| Split | Scenes | Tiles |
|---|---:|---:|
| Training | 26 | 3,744 |
| Validation | 6 | 864 |
| Testing | 6 | 864 |
| **Total** | **38** | **5,472** |

The original scenes are divided into 512 × 512 tiles. Dataset splitting is performed at the scene level to prevent spatial leakage between training, validation, and testing sets.

Image-level multi-label annotations are derived from the semantic masks to provide weak supervision. Pixel-level annotations are reserved for evaluation.

## Methodology

The proposed framework follows a diffusion-guided weakly supervised segmentation approach inspired by DiG.

### 1. Denoising Diffusion Probabilistic Model (DDPM)

A domain-specific DDPM is trained on unlabeled Potsdam training images to learn the visual distribution of aerial imagery.

The diffusion model uses a U-Net architecture and a velocity prediction objective.

Main components:

- U-Net denoising architecture.
- Sigmoid noise schedule.
- 1,000 diffusion timesteps.
- Velocity prediction objective (`pred_v`).
- Adam optimizer.
- Mixed-precision training.
- Exponential Moving Average (EMA).

The trained diffusion model provides intermediate feature representations and reconstructed images for subsequent stages.

### 2. Diffusion-Guided Feature Learning

The framework incorporates two complementary components inspired by DiG:

**Learnable Feature Cross-Attention (LFCA)**

Integrates intermediate diffusion representations into Vision Transformer features through cross-attention, providing additional visual information for semantic localization and class activation map generation.

**Pairwise Affinity Consistency (PAC)**

Uses original and diffusion-reconstructed images to encourage consistency in learned visual representations and pairwise affinities, improving semantic localization.

The diffusion model is kept frozen during this stage.

### 3. Weakly Supervised Semantic Segmentation

Image-level class annotations are used to generate class activation maps and derive pseudo-labels for semantic segmentation.

The resulting pseudo-labels provide supervision for training the final segmentation network.

## Implementation

The project is implemented using Python and PyTorch.

**Technologies and libraries:**

- Python
- PyTorch
- Torchvision
- NumPy
- OpenCV
- Hugging Face

**Training environment:**

- NVIDIA Tesla T4 GPUs
- Kaggle GPU environment
- Mixed-precision training (FP16)

## Repository Structure

```text
DiffWSS/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── configs/
│
├── models/
│   ├── ddpm/
│   ├── vit/
│   └── segmentation/
│
├── scripts/
│
├── checkpoints/
│
├── outputs/
│
├── README.md
└── requirements.txt
```

## Research Reference

**Diffusion-Guided Weakly Supervised Semantic Segmentation (DiG)**

ECCV 2024.

- Official implementation: https://github.com/yoon307/DiG
- Research paper: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06482.pdf

DiffWSS investigates the adaptation of diffusion-guided weakly supervised semantic segmentation to high-resolution aerial imagery, using the ISPRS Potsdam dataset.

---

**Project status:** Research and implementation in progress.
