import os
import sys
from pathlib import Path

# Point Sphinx to the src directory so it can read your package
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "sdofmv2"
copyright = "2026, Hong, Jinsu and Martin, Daniela and Gallego, Joseph"
author = "Hong, Jinsu and Martin, Daniela and Gallego, Joseph"
release = "v0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add the extensions here!
extensions = [
    "sphinx.ext.autodoc",  # Pulls documentation from docstrings
    "sphinx.ext.napoleon",  # Supports Google/NumPy-style docstrings
    "sphinx.ext.viewcode",  # Adds links to highlighted source code
    "myst_parser",  # Allows you to use Markdown (like your README)
]

templates_path = ["_templates"]
exclude_patterns = [
    "**/*_old.py",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Change the theme to the Read the Docs style
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_mock_imports = [
    # Core & Math
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "skimage",
    "matplotlib",
    "xarray",
    # Domain Specific
    "sunpy",
    "zarr",
    "blosc",
    "h5py",
    "astropy",
    # Deep Learning & Models
    "torch",
    "torchvision",
    "torchaudio",
    "torchcodec",
    "lightning",
    "torchmetrics",
    "transformers",
    "tokenizers",
    "timm",
    "einops",
    "safetensors",
    "segmentation_models_pytorch",
    "efficientnet_pytorch",
    "pretrainedmodels",
    # Config & Tracking
    "wandb",
    "hydra",
    "omegaconf",
    "optuna",
    "loguru",
    "rich",
    "yaml",
    # Utilities
    "joblib",
    "overrides",
    "tqdm",
    "dask",
    "datasets",
    "s3fs",
]
