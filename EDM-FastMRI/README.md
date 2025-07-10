# Robust MRI reconstruction via self-supervised denoising | MRM 2025
by [Asad Aali](https://asadaali.com/) and [Jon Tamir](http://users.ece.utexas.edu/~jtamir/csilab.html), UT CSI Lab.

This repository contains the FastMRI implementation of **Elucidating the Design Space of Diffusion-Based Generative Models (EDM)** with specialized components for MRI reconstruction tasks. The implementation includes GSURE preconditioning for denoising applications.

![samples](docs/pipeline.png)

Figure | Pipeline describing the techniques utilized for: (i) GSURE Denoising, (ii) GSURE-DPS Training/Inference, and (iii) GSURE-MoDL Training/Inference

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Training](#training)
- [Prior Sampling](#prior-sampling)
- [Diffusion Posterior Sampling](#diffusion-posterior-sampling)
- [Project Structure](#project-structure)
- [Data Structure](#data-structure)
- [Configuration Options](#configuration-options)
- [Citations](#citations)

## Overview

This project extends the original EDM framework to work with FastMRI data, providing:

- **Multiple Data Loaders**: Support for different MRI data formats (Noisy, Numpy, Image, NIfTI)
- **GSURE Preconditioning**: Specialized preconditioning for denoising applications
- **Multi-Anatomy Support**: Brain and knee MRI reconstruction
- **DPS Inference**: Diffusion Posterior Sampling for constrained reconstruction

## Installation

### Environment Setup

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate edm
```

## Training

### EDM Training

Use the provided training script:

```bash
bash train_edm.sh
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

Use the provided training script:

```bash
bash train_denoiser.sh
```

The `train_denoiser.sh` script contains the same parameters, except we set PRECOND=gsure:

```bash
# Model Configuration
PRECOND=gsure             # Preconditioning: vp|ve|edm|gsure
```

## Prior Sampling

Use the provided generation script:

```bash
bash generate.sh
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

## Diffusion Posterior Sampling

DPS enables constrained MRI reconstruction by incorporating measurement consistency during the diffusion sampling process.

### Basic DPS Inference

```bash
bash dps.sh
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

## Project Structure

```
EDM-FastMRI/
├── train.py                # Main training script
├── generate.py             # Generation script
├── dps.py                  # DPS inference script
├── train_edm.sh            # EDM training shell script
├── train_denoiser.sh       # Denoiser training shell script
├── generate.sh             # Prior sampling shell script
├── dps.sh                  # DPS inference shell script
├── environment.yml         # Conda environment
├── training/               # Training modules
│   ├── dataset.py          # Data loading classes
│   ├── networks.py         # Model architectures and preconditioning
│   ├── loss.py             # Loss functions (EDM, GSURE, VP, VE)
│   ├── training_loop.py    # Main training loop
│   └── augment.py          # Data augmentation
├── utils/                  # Utility functions
│   ├── brain_train_data.py # Generate train data from original brain FastMRI
│   ├── brain_val_data.py   # Generate val data from original brain FastMRI
│   ├── knee_train_data.py  # Generate train data from original knee FastMRI
│   ├── knee_val_data.py    # Generate val data from original knee FastMRI
│   └── gsure_inference.ipynb $ Script for generating denoised samples using trained denoiser
└── torch_utils/          # PyTorch utilities
    ├── distributed.py
    ├── misc.py
    └── ...
```

## Data Structure

The project expects data in the following structure, but can:
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

## Configuration Options

### Training Parameters

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

## Citations

If you use this code in your research, please cite:

```bibtex
@article{aali2025robust,
  title={Robust multi-coil MRI reconstruction via self-supervised denoising},
  author={Aali, Asad and Arvinte, Marius and Kumar, Sidharth and Arefeen, Yamin I and Tamir, Jonathan I},
  journal={Magnetic Resonance in Medicine},
  year={2025},
  publisher={Wiley Online Library}
}

@inproceedings{Karras2022edm,
  author    = {Tero Karras and Miika Aittala and Timo Aila and Samuli Laine},
  title     = {Elucidating the Design Space of Diffusion-Based Generative Models},
  booktitle = {Proc. NeurIPS},
  year      = {2022}
}
```
