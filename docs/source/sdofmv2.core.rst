Core Models & Data
====================

Attention Map
-------------

.. automodule:: sdofmv2.core.attention_map
   :members:
   :show-inheritance:
   :undoc-members:

Base Module
-----------

.. autoclass:: sdofmv2.core.basemodule.BaseModule
   :members:
   :show-inheritance:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      configure_optimizers
      training_step
      validation_step

Data Module
-----------

.. autoclass:: sdofmv2.core.datamodule.SDOMLDataset
   :members:
   :show-inheritance:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      get_aia_image
      get_eve
      get_hmi_image

.. autoclass:: sdofmv2.core.datamodule.SDOMLDataModule
   :members:
   :show-inheritance:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      setup
      test_dataloader
      train_dataloader
      val_dataloader

Losses
------

.. automodule:: sdofmv2.core.losses
   :members:
   :show-inheritance:
   :undoc-members:

Masked Autoencoder (MAE)
------------------------

.. autoclass:: sdofmv2.core.mae3d.MaskedAutoencoderViT3D
   :members:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      forward
      get_intermediate_layers
      random_masking

MAE Module
----------

.. autoclass:: sdofmv2.core.mae_module.MAE
   :members:
   :show-inheritance:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      forward
      forward_encoder
      on_test_epoch_end
      on_validation_epoch_end
      test_step
      training_step
      validation_step

Principal Component Analysis
----------------------------

.. automodule:: sdofmv2.core.pca_analysis
   :members:
   :show-inheritance:
   :undoc-members:

.. reconstruction module
.. ----------------------------------

.. .. automodule:: sdofmv2.core.reconstruction
..    :members:
..    :show-inheritance:
..    :undoc-members:

.. Module contents
.. ---------------

.. .. automodule:: sdofmv2.core
..    :members:
..    :show-inheritance:
..    :undoc-members:
