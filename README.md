# Voformer-EC
<img width="4267" height="1646" alt="graphic abstract" src="https://github.com/user-attachments/assets/4bdd19a9-7fb4-45e8-8d7a-a76a387df388" />
Official implementation of **Voformer-EC: A Deep Clustering Framework with Statistical Physical Priors for Multivariate Time Series**.

Voformer-EC is a deep clustering framework for **multivariate climate time series (MTS)**, designed to identify drought-relevant regimes from **unlabeled meteorological sequences**. The method combines:

- a **dual-branch encoder** that fuses deep temporal features with **statistical physical priors**,
- the **Neural Network Volatility Activation Function (NNVAF)** for volatility-aware feature modulation,
- and a **three-phase optimization strategy** with **Extreme Clustering (EC)** center initialization.

---

## Highlights

- **Dual-branch representation learning**: combines a statistical-prior branch and a deep temporal branch.
- **Volatility-aware activation**: NNVAF dynamically amplifies informative bursts associated with extreme events.
- **Three-phase optimization**:
  1. reconstruction pretraining,
  2. EC-based cluster center initialization,
  3. joint reconstruction + clustering + contrastive learning.
- **Auxiliary agreement analysis**: reports NMI and ARI against precipitation-based proxy classes for evaluation only.
- **Visualization support**: includes both latent-space visualization and geographic clustering map rendering.

### Script overview

#### `MAIN_SCRIPT.py`
Main end-to-end pipeline for:
- loading climate MTS data,
- preprocessing with Z-score standardization,
- training or loading the Voformer-EC model,
- proxy-label generation for auxiliary evaluation,
- latent-space clustering quality evaluation,
- optional latent-space visualization.

#### `MAP_VIS_SCRIPT.py`
Visualization pipeline for:
- bar / pie / scatter summaries of clustering results,
- interactive geographic rendering using **Pyecharts BMap**,
- exporting an HTML map of station-wise cluster assignments.

---

## Method Overview

Voformer-EC follows the workflow below:

1. **Input & preprocessing**
   - select climate variables,
   - handle missing values / outliers,
   - apply Z-score standardization,
   - perform integrity checks.

2. **Dual-branch encoder**
   - **Statistical Prior Injector** extracts sequence-level statistics: mean, standard deviation, and maximum.
   - **Deep Temporal Encoder** uses convolutional embedding, positional encoding, Transformer encoder layers, and NNVAF.

3. **Three-phase optimization**
   - **Phase 1:** reconstruction pretraining,
   - **Phase 2:** EC-based center initialization,
   - **Phase 3:** joint optimization with reconstruction loss, clustering KL loss, and contrastive NT-Xent loss.

4. **Evaluation**
   - internal metrics: **Silhouette Score (SS)**, **Calinski-Harabasz (CH)**, **Davies-Bouldin (DB)**,
   - auxiliary agreement metrics: **NMI** and **ARI** against precipitation-based proxy labels.

> The precipitation-based proxy labels are used **for evaluation only** and are **not** used for training, center initialization, or hyperparameter tuning.

---

## Installation

### Recommended environment

- Python 3.10+
- PyTorch with CUDA support recommended

### Dependencies

Install the main dependencies with:

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch pandas-bokeh pyecharts openpyxl
```

Depending on your PyTorch/CUDA setup, install `torch` from the official PyTorch channel that matches your system.

---

## Data Preparation

### 1. Main training data

The main script expects a NumPy file:

```text
combined_precipitation_temperature_data.npy
```

Expected format:

- shape: **(N, T, F)**
- `N`: number of samples / stations
- `T`: sequence length
- `F`: number of input variables
- default `F = 2`

Recommended channel order based on the current code:

```text
[..., 0] = precipitation
[..., 1] = temperature
```

This order matters because the proxy-label generation logic uses the **first channel** to compute precipitation-based evaluation classes.

### 2. Map visualization data

The map visualization script expects:

- `result analysis.xlsx`
- `map.json`

The Excel file should contain at least:

#### `Sheet`
Used for bar / pie / scatter plots.
Suggested columns:

- `clusters`
- `count`
- `size`

#### `Sheet1`
Used for geographic visualization.
Required columns:

- `Name`
- `longitude`
- `latitude`
- `clusters`

---

## Quick Start

### Train Voformer-EC from scratch

1. Put `combined_precipitation_temperature_data.npy` in the project root, or update the path in `Config.data_path`.
2. In `Config`, set:

```python
train_voformer = True
load_voformer =
