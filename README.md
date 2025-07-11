# Robust MRI reconstruction via self-supervised denoising | MRM 2025

Official implementation for "[Robust multi-coil MRI reconstruction via self-supervised denoising](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.30591)" (Magnetic Resonance in Medicine, 2025).

This repository contains the FastMRI implementation of **Elucidating the Design Space of Diffusion-Based Generative Models (EDM)** and **Model Based Deep Learning (MoDL)** with training and inference components for MRI reconstruction.

![samples](assets/pipeline.png)

Figure | Pipeline describing: (i) GSURE Denoising, (ii) GSURE-DPS, and (iii) GSURE-MoDL.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Prior Sampling](#prior-sampling)
- [Posterior Sampling](#posterior-sampling)
- [Project Structure](#project-structure)
- [Configuration Options](#configuration-options)
- [Citation](#citation)

## Overview

This project extends the original EDM/MoDL framework to work with FastMRI data, providing:

- **Multiple Data Loaders**: Support for different MRI data formats (Noisy, Numpy, Image, NIfTI)
- **GSURE Denoising**: Specialized training for denoising applications
- **Multi-Anatomy Support**: Brain and Knee MRI reconstruction
- **DPS Inference**: Diffusion Posterior Sampling for constrained reconstruction
- **MoDL Inference**: Model Based Deep Learning for constrained reconstruction

## Installation

### Environment Setup

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate edm
```

## Data Preparation

The project expects data in the following structure:
```
/path/to/data/
├── anatomy/                # 'brain' or 'knee'
│   ├── train/
│   │   ├── SNR/            # e.g., '32dB', '22dB', '12dB'
│   │   │   ├── data_type/  # 'noisy', 'denoised', etc.
│   │   │   │   └── data.pt
│   │   │   └── ...
│   │   └── ...
│   └── val/
│       └── ... (similar structure)
└── ...
```

## Training

### EDM Training

```bash
bash EDM-FastMRI/train_edm.sh
```

The `train_edm.sh` script contains the following key parameters:

```bash
# GPU and Process Configuration
CUDA_VISIBLE_DEVICES=0    # GPU to use
NPROC=1                   # Number of processes

# Data Configuration
LOADER=Noisy              # Data loader type: Image|Numpy|Noisy|Nifti
ANATOMY=brain             # Anatomy type: brain|knee
DATA=noisy                # Data type
SNR=32dB                  # Signal-to-noise ratio of the training dataset
ROOT=/path/to/root/       # Path to the root folder of the codebase
ROOT_DATA=/path/to/data/  # Path to the root folder of the dataset
BATCH_SIZE=1              # Batch size per GPU
NORMALIZE=0               # Input normalization (0=off, 1=on)

# Model Configuration
PRECOND=edm               # Preconditioning: vp|ve|edm|gsure
AUGMENT=0                 # Data augmentation probability

torchrun --standalone --nproc_per_node=$NPROC train.py \
 --outdir=$ROOT/models/$PRECOND/$ANATOMY/$SNR \
 --data=$ROOT_DATA/$ANATOMY/train/$SNR/$DATA \
 --cond=0 --arch=ddpmpp --duration=10 \
 --batch=$BATCH_SIZE --cbase=128 --cres=1,1,2,2,2,2,2 \
 --lr=1e-4 --ema=0.1 --dropout=0.0 \
 --desc=container_test --tick=1 --snap=10 \
 --dump=200 --seed=2023 --precond=$PRECOND --augment=$AUGMENT \
 --normalize=$NORMALIZE --loader=$LOADER --gpu=$CUDA_VISIBLE_DEVICES
```

### GSURE Denoiser Training

```bash
bash EDM-FastMRI/train_denoiser.sh
```

The `train_denoiser.sh` script contains the same parameters, except we set `PRECOND=gsure`:

```bash
# Model Configuration
PRECOND=gsure             # Preconditioning: vp|ve|edm|gsure
```

### MoDL Training

```bash
bash MoDL-FastMRI/train.sh
```

The `train.sh` script contains the following key parameters:

```bash
# GPU and Process Configuration
CUDA_VISIBLE_DEVICES=0    # GPU to use
NPROC=1                   # Number of processes

# Training Configuration
EPOCHS=10                 # Number of training epochs
ANATOMY=brain             # Anatomy type: brain|knee
DATA=noisy                # Data type
ROOT=/path/to/root/       # Path to the root folder of the codebase
METHOD=modl               # Method identifier

# Data Configuration
SNR=32dB                  # Signal-to-noise ratio of the training dataset
KSP_PATH=/path/to/data/$ANATOMY/train/$SNR/ksp/    # Path to k-space data
DATA_PATH=/path/to/data/$ANATOMY/train/$SNR/$DATA.pt    # Path to training data
R=4                       # Acceleration factor

torchrun --standalone --nproc_per_node=$NPROC train.py \
    --gpu=$CUDA_VISIBLE_DEVICES --data_R=$R --epochs=$EPOCHS \
    --snr=$SNR --anatomy=$ANATOMY --root=$ROOT \
    --ksp_path=$KSP_PATH --data_path=$DATA_PATH \
    --data_type=$DATA --method=$METHOD
```

## Prior Sampling

```bash
bash EDM-FastMRI/generate.sh
```

The `generate.sh` script generates samples across multiple conditions:

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0    # GPUs to use
NPROC=1                   # Number of processes

# Model Configuration
ROOT=/path/to/root/              # Path to the root folder of the codebase
MODEL_PATH=network-snapshot.pkl  # Trained model name

# Generation Parameters
SAMPLE_DIM=384,320        # Output dimensions
NUM_SAMPLES=100           # Number of samples to generate
BATCH_SIZE=40             # Batch size

# Data Configuration
ANATOMY=brain             # Anatomy type: brain|knee
DATA=noisy32dB            # Data type and snr of the training model

torchrun --standalone --nproc_per_node=$NPROC generate.py \
        --outdir=$ROOT/results/priors/$ANATOMY/$DATA --seeds=1-$NUM_SAMPLES \
        --batch=$BATCH_SIZE --network=$ROOT/models/edm/$ANATOMY/$DATA/$MODEL_PATH \
        --sample_dim=$SAMPLE_DIM --gpu=$CUDA_VISIBLE_DEVICES
```

## Posterior Sampling

### DPS Inference

```bash
bash EDM-FastMRI/dps.sh
```

The `dps.sh` script performs reconstruction across multiple conditions:

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=1    # GPU to use
NPROC=1                   # Number of processes

# Anatomy and Data Configuration
ANATOMY=brain                        # Anatomy type: brain|knee
NATIVE_SNR=32dB                      # Native SNR of the dataset
ROOT=/path/to/root/                  # Path to the root folder of the codebase
MODEL_PATH=network-snapshot.pkl      # Trained model name 
MEAS_PATH=/path/to/meas/$NATIVE_SNR  # Path to the native SNR kspace measurements

# Reconstruction Parameters
STEPS=500                 # Number of diffusion steps
SAMPLE_START=0            # Starting sample index
SAMPLE_END=100            # Ending sample index

# Inference Configuration
DATA=noisy32dB                          # Data type and SNR of the trained model
INFERENCE_SNR=32dB                      # SNR of the inference kspace input
R=4                                     # Acceleration factor of the inference kspace input
SEED=15                                 # Random generation seed
KSP_PATH=/path/to/ksp/$INFERENCE_SNR    # Path to the inference SNR kspace measurements

torchrun --standalone --nproc_per_node=$NPROC dps.py \
    --seed $SEED --latent_seeds $SEED --gpu=$CUDA_VISIBLE_DEVICES \
    --sample_start $SAMPLE_START --sample_end $SAMPLE_END \
    --inference_R $R --inference_snr $INFERENCE_SNR \
    --num_steps $STEPS --S_churn 0 \
    --measurements_path $MEAS_PATH \
    --ksp_path $KSP_PATH \
    --network=$ROOT/models/edm/$ANATOMY/$DATA/$MODEL_PATH \
    --outdir=$ROOT/results/posterior/$ANATOMY/$DATA
```

### MoDL Inference

```bash
bash MoDL-FastMRI/inference.sh
```

The `inference.sh` script performs MoDL reconstruction:

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0    # GPU to use
NPROC=1                   # Number of processes

# Anatomy and Data Configuration
ANATOMY=brain                        # Anatomy type: brain|knee
NATIVE_SNR=32dB                      # Native SNR of the dataset
ROOT=/path/to/root/                  # Path to the root folder of the codebase
MODEL_PATH=ckpt_9.pt                 # Trained model checkpoint
MEAS_PATH=/path/to/data/brain/val/$NATIVE_SNR    # Path to validation measurements

# Reconstruction Parameters
SAMPLE_START=0            # Starting sample index
SAMPLE_END=100            # Ending sample index
METHOD=modl               # Method identifier

# Inference Configuration
DATA=noisy32dB                          # Data type and SNR of the trained model
INFERENCE_SNR=32dB                      # SNR of the inference input
R=4                                     # Acceleration factor
KSP_PATH=/path/to/data/brain/val/$INFERENCE_SNR    # Path to inference k-space data

torchrun --standalone --nproc_per_node=$NPROC inference.py \
    --gpu=$CUDA_VISIBLE_DEVICES --anatomy=$ANATOMY\
    --sample_start $SAMPLE_START --sample_end $SAMPLE_END \
    --inference_R $R --inference_snr $INFERENCE_SNR \
    --measurements_path $MEAS_PATH \
    --ksp_path $KSP_PATH \
    --network=$ROOT/models_$METHOD/$ANATOMY/$DATA/R=$R/$MODEL_PATH \
    --outdir=$ROOT/results_$METHOD/$ANATOMY/$DATA/R=$R/snr$INFERENCE_SNR \
    --method=$METHOD
```

## Project Structure

```
EDM-FastMRI/
├── train.py                  # Main training script
├── generate.py               # Generation script
├── dps.py                    # DPS inference script
├── train_edm.sh              # EDM training shell script
├── train_denoiser.sh         # Denoiser training shell script
├── generate.sh               # Prior sampling shell script
├── dps.sh                    # DPS inference shell script
├── environment.yml           # Conda environment
├── training/                 # Training modules
│   ├── dataset.py            # Data loading classes
│   ├── networks.py           # Model architectures and preconditioning
│   ├── loss.py               # Loss functions (EDM, GSURE, VP, VE)
│   ├── training_loop.py      # Main training loop
│   └── augment.py            # Data augmentation
├── utils/                    # Utility functions
│   ├── brain_train_data.py   # Generate train data from original brain FastMRI
│   ├── brain_val_data.py     # Generate val data from original brain FastMRI
│   ├── knee_train_data.py    # Generate train data from original knee FastMRI
│   ├── knee_val_data.py      # Generate val data from original knee FastMRI
│   └── gsure_inference.ipynb # Script for generating denoised samples using trained denoiser
└── torch_utils/              # PyTorch utilities
    ├── distributed.py
    ├── misc.py
    └── ...
```
```
MoDL-FastMRI/
├── train.py                # Main MoDL training script
├── inference.py            # MoDL inference script
├── train.sh                # MoDL training shell script
├── inference.sh            # MoDL inference shell script
├── inference.ipynb         # Jupyter notebook for MoDL inference
├── models.py               # MoDL model architectures
├── unet.py                 # U-Net model implementation
├── losses.py               # Loss functions for MoDL
├── ops.py                  # Basic operations and utilities
├── opt.py                  # Optimization routines
├── core_ops.py             # Core MoDL operations
├── CG.py                   # Conjugate gradient solver
├── datagen.py              # Data generation utilities
└── utils.py                # General utility functions
```

## Configuration Options

### EDM Training Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--loader` | Data loader type | `Image` | `Image`, `Numpy`, `Noisy`, `Nifti` |
| `--precond` | Preconditioning method | `edm` | `vp`, `ve`, `edm`, `gsure` |
| `--arch` | Network architecture | `ddpmpp` | `ddpmpp`, `ncsnpp`, `adm` |
| `--duration` | Training duration (Mimg) | `200` | Any positive float |
| `--batch` | Total batch size | `512` | Any positive integer |
| `--lr` | Learning rate | `1e-4` | Any positive float |
| `--normalize` | Input normalization | `1` | `0` (off), `1` (on) |

### Prior Sampling Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--steps` | Sampling steps | `None` | Any positive integer |
| `--S_churn` | Stochasticity strength | `0` | Any non-negative float |
| `--solver` | ODE solver | `heun` | `euler`, `heun` |
| `--sample_dim` | Output dimensions | Auto | `H,W` format |

### DPS Parameters

| Parameter | Description |
|-----------|-------------|
| `--num_steps` | Diffusion steps |
| `--inference_R` | Acceleration factor |
| `--inference_snr` | Noise level |
| `--l_ss` | Likelihood step size |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{aali2025robust,
  title={Robust multi-coil MRI reconstruction via self-supervised denoising},
  author={Aali, Asad and Arvinte, Marius and Kumar, Sidharth and Arefeen, Yamin I and Tamir, Jonathan I},
  journal={Magnetic Resonance in Medicine},
  year={2025},
  publisher={Wiley Online Library}
}
```
