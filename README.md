# SDO FM v2: A Multi-Instrument Foundation Model for the Solar Dynamics Observatory with Transferable Downstream Applications

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch_Lightning-%23792EE5.svg?style=flat&logo=pytorchlightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction
**SDOFMv2** is an advanced multi-instrument foundation model designed to analyze Solar Dynamics Observatory (SDO) data and drive large-scale, data-driven heliophysics research. Building upon the original SDOFM framework, this version addresses previous limitations like restricted temporal coverage and reconstruction artifacts to significantly improve spatial coherence and global consistency.

![Model architecture](https://raw.githubusercontent.com/Joaggi/sdofmv2/main/sdofmv2.svg)
*A Masked Autoencoder (MAE) based on a Vision Transformer (ViT) architecture is utilized for pretraining. During this phase, a% of the image patches are masked, while the remaining (100 - a)% are processed by the encoder. The decoder block then reconstructs all patches, optimized via a customized loss function.*

---

## Getting Started

### Prerequisites
* Linux or macOS
* Python 3.11+
* NVIDIA GPU + CUDA toolkit (Recommended for training)

### Environment Setup
We recommend using `mamba` to manage dependencies. 

> **Important Hardware Note:** > The `sdofmv2_environment.yml` file is configured for **CUDA 12.8** by default. If your hardware or drivers require a different CUDA version (e.g., CUDA 11.8), please open `sdofmv2_environment.yml` and modify the `pip` section at the bottom to match your system (e.g., change `cu128` to `cu118`) before running the setup commands.

**Using Mamba:**
```bash
# Clone the repository
git clone [https://github.com/Joaggi/sdofmv2.git](https://github.com/Joaggi/sdofmv2.git)
cd sdofmv2

# Create and activate the environment (This automatically installs PyTorch and the local package)
mamba env create -f sdofmv2_environment.yml
mamba activate sdofmv2
```
---

## Repository Structure

```text
.
├── configs/                # YAML configurations for experiments
│   ├── downstream/         # Configs for downstream tasks (F10.7, solar wind)
│   └── pretrain/           # Configs for MAE pretraining (AIA, HMI)
├── notebooks/              # Jupyter notebooks for analysis and visualization
│   ├── analysis/           # Attention maps, PCA, and masking analysis
│   └── downstream_apps/    # How to use downstream scripts (Notebooks) for F10.7 and missing data applications
├── scripts/                # Executable scripts for training and testing
│   ├── pretrain.py         # Main pretraining script
│   ├── finetuning_*.py     # Scripts for downstream finetuning
│   └── test.py             # Script for evaluating checkpoints
├── src/                    # Core source code package
│   └── sdofmv2/
│       ├── core/           # Base model architectures and modules
│       ├── tasks/          # PyTorch Lightning modules (model & data module) for downstream tasks
│       └── utils/          # Helper functions, physical constants and metrics
├── pyproject.toml          # Project metadata and build dependencies
└── sdofmv2_environment.yml # Mamba environment definition file
```

---

## How to Use

*(Note: It is recommended to run all scripts from the root directory of the repository so that file paths to `configs/` and `src/` resolve correctly.)*

### 1. Data Preparation
Before training or running inference, you need to prepare the dataset. 
[Explain where to download the data, or provide a command if you have a script for it.]
```bash
python scripts/download_data_cache.py --target_dir ./assets/
```

### 2. Training the Model
To train the model from scratch, execute the pretraining script and pass the relevant configuration file.
```bash
python scripts/pretrain.py --config-name pretrain_mae_AIA.yaml
```

### 3. Inference and Evaluation
To evaluate a pre-trained checkpoint on the test set:
```bash
python scripts/test.py --config-name pretrain_mae_AIA.yaml
```

### 4. Downstream Finetuning
To finetune the model on a specific downstream task (e.g., solar wind forecasting):
```bash
python scripts/finetuning_solarwind.py --config-name finetune_solarwind_config.yaml
```
---

## Results & Visualizations
[Include a brief summary of the model's performance. You can add a table of metrics or a sample plot showing predictions vs. ground truth.]

![Sample Visualization](https://raw.githubusercontent.com/Joaggi/sdofmv2/main/notebooks/analysis/SDOFMv2_AIA_results_exp.png) 
*The first row displays the original ground-truth images. The second and third rows show the model's reconstructed images using masking ratios of 0% and 50%, respectively.*

---

## Citation
If you find this repository or model useful in your academic research, please consider citing our work:

```bibtex
@misc{sdofmv2,
  author = {Hong, Jinsu and Martin, Daniela and Gallego, Joseph},
  title = {SDOFMv2: A Multi-Instrument Foundation Model for the Solar Dynamics Observatory with Transferable Downstream Applications},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/Joaggi/sdofmv2](https://github.com/Joaggi/sdofmv2)}},
  note = {Jinsu Hong, Daniela Martin, and Joseph Gallego contributed equally to this work}
}
```

## Contributing
Contributions, bug reports, and feature requests are welcome! Please feel free to check the [issues page](https://github.com/Joaggi/sdofmv2/issues) or submit a pull request.