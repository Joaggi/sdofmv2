============================================
SDO Foundation Models for Heliophysics (v2)
============================================

.. .. image:: https://raw.githubusercontent.com/Joaggi/sdofmv2/main/sdofmv2.png
..    :align: center
..    :alt: SDOFMv2 Architecture
..    :width: 600px

Welcome to **SDOFMv2**! This library provides a scalable framework for extracting general-purpose representations from the vast Solar Dynamics Observatory (SDO) data archive.

The Challenge: From Data Abundance to Analysis Bottlenecks
----------------------------------------------------------
Since its launch, the **Solar Dynamics Observatory (SDO)** has revolutionized heliophysics, providing a continuous, multi-wavelength record of the solar atmosphere for over a solar cycle. This data is vital for protecting Earth’s infrastructure—from GPS to power grids—against solar flares and coronal mass ejections.

However, as our data archives grow, our analysis methods have struggled to keep pace:

* **Task-Specific Limitations:** Most current models are trained for a single scientific objective (e.g., flare detection or segmentation).
* **Handcrafted Features:** Many pipelines rely on manual event catalogs that are difficult to scale.
* **Fragmented Knowledge:** Valuable spatial and temporal relationships are often relearned from scratch for every new project.

Our Solution: A Representational Scaffolding
--------------------------------------------
In other domains, **Foundation Models** have solved similar bottlenecks by learning general representations from massive datasets. SDOFMv2 brings that "representational scaffolding" to solar physics.

Rather than optimizing for one task, our models encode the underlying structural, temporal, and multi-wavelength relationships inherent in SDO data. These pre-trained models can then be adapted (fine-tuned) for a wide range of downstream applications—from data reconstruction to event forecasting—without requiring extensive manual labeling.

Why SDO is Perfect for Foundation Modeling
------------------------------------------
The SDO archive offers a uniquely favorable environment for self-supervised learning:

* **Long-term Continuity:** Over a decade of consistently calibrated observations.
* **High Cadence:** Enables models to learn both instantaneous features and long-term evolution.
* **Multi-Channel Synergy:** Simultaneous coverage across wavelengths allows for rich, multi-modal representation learning.

Key Features
------------
* **Pre-trained Backbones:** Access state-of-the-art weights trained on years of SDO data.
* **Self-Supervised Learning:** Training paradigms that leverage the scale of the SDO archive without needing manual labels.
* **Modular Architecture:** Easily swap "heads" for different scientific tasks (classification, regression, or reconstruction).
* **Community Focused:** Robust and reusable models designed for the broader heliophysics community.

.. tip::
   **Ready to get started?**
   
   * **Source Code:** Visit our `GitHub Repository <https://github.com/Joaggi/sdofmv2>`_ to view the code and latest releases.
   * **Setup:** Follow our `installation <https://github.com/Joaggi/sdofmv2?tab=readme-ov-file#getting-started>`_ guide to set up your environment.
   * **Tutorial:** Check out the `usage <https://github.com/Joaggi/sdofmv2?tab=readme-ov-file#data-preparation>`_ tutorial to load your first pre-trained model.
   * **Contribute:** Found a bug? Open an issue on `GitHub Issues <https://github.com/Joaggi/sdofmv2/issues>`_.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   Core Models & Data <sdofmv2.core>
   Downstream Tasks <sdofmv2.tasks>
   Utilities <sdofmv2.utils>

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`