.. _getting_started:

***************
Getting Started
***************

This guide provides instructions on how to set up your environment, download the required data, and run the SDOFMv2 scripts.

Environment Setup
=================

Prerequisites
-------------

*   Linux or macOS
*   Python 3.11+
*   NVIDIA GPU + CUDA toolkit (recommended for training)

Installation
------------

We use ``mamba`` (or ``conda``) for fast dependency resolution.

.. note::
    **Hardware Note:** ``sdofmv2_environment.yml`` is configured for **CUDA 12.8** by default. If your system requires a different CUDA version (e.g., 11.8), edit the ``pip`` section in ``sdofmv2_environment.yml`` before running setup — change ``cu128`` to the appropriate tag (e.g., ``cu118``).

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/Joaggi/sdofmv2.git
    cd sdofmv2

    # Create and activate the environment
    # (installs PyTorch and the local package automatically)
    mamba env create -f sdofmv2_environment.yml
    mamba activate sdofmv2

Data Preparation
================

SDOFMv2 uses the **SDOMLv2** dataset — a curated, multi-instrument dataset for the Solar Dynamics Observatory, hosted on NASA's HDRL S3 bucket. Data is streamed via ``s3fs`` and stored in the Zarr format.

Dataset Components
------------------

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Component
     - Instrument
     - Data Type
     - Description
   * - ``aia``
     - AIA
     - EUV Images
     - 9 extreme ultraviolet channels (``94 Å``, ``131 Å``, ``171 Å``, ``193 Å``, ``211 Å``, ``304 Å``, ``335 Å``, ``1600 Å``, ``1700 Å``), capturing the solar atmosphere
   * - ``hmi``
     - HMI
     - Magnetograms
     - 3-component vector magnetic field (Bx, By, Bz) for the solar photosphere

.. warning::
    Zarr datasets require significant local disk space. Verify your target drive has sufficient capacity before downloading.

Downloading the Data
--------------------

The download script is **resumable** — it checks for existing local files and only fetches what's missing.

.. code-block:: bash

    # Download AIA only
    python scripts/download_data.py --target /path/to/your/storage --component aia

    # Download HMI only
    python scripts/download_data.py --target /path/to/your/storage --component hmi

    # Download the full dataset
    python scripts/download_data.py --target /path/to/your/storage --component both

Training & Evaluation
=====================

Pretraining
-----------

.. code-block:: bash

    python scripts/pretrain.py --config-name pretrain_mae_AIA.yaml

Evaluation
----------

.. code-block:: bash

    python scripts/test.py --config-name pretrain_mae_AIA.yaml

Downstream Finetuning
---------------------

.. code-block:: bash

    # Example: solar wind forecasting
    python scripts/finetuning_solarwind.py --config-name finetune_solarwind_config.yaml

Configuration files for all tasks are in ``configs/downstream/``. Notebook-based walkthroughs are available in ``notebooks/downstream_apps/``.
