Downstream App: Solar Wind
==========================

Data Module
-----------

.. autoclass:: sdofmv2.tasks.solar_wind.datamodule.SWDataset
   :members:
   :show-inheritance:
   :no-undoc-members:

.. autoclass:: sdofmv2.tasks.solar_wind.datamodule.SWDataModule
   :members:
   :show-inheritance:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      predict_dataloader
      setup
      test_dataloader
      train_dataloader
      val_dataloader

Head Module
-----------

.. autoclass:: sdofmv2.tasks.solar_wind.head_networks.TransformerHead
   :members:
   :show-inheritance:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      forward

.. autoclass:: sdofmv2.tasks.solar_wind.head_networks.SimpleLinear
   :members:
   :show-inheritance:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      forward

.. autoclass:: sdofmv2.tasks.solar_wind.head_networks.SkipLinearHead
   :members:
   :show-inheritance:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      forward

.. autoclass:: sdofmv2.tasks.solar_wind.head_networks.ClsLinear
   :members:
   :show-inheritance:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      forward

Model Module
------------

.. autoclass:: sdofmv2.tasks.solar_wind.model.SWClassifier
   :members:
   :show-inheritance:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      forward
      forward_analysis
      on_before_optimizer_step
      on_test_epoch_end
      on_train_epoch_end
      on_validation_epoch_end
      predict_step
      test_step
      training_step
      validation_step

Focal Loss
----------

.. automodule:: sdofmv2.tasks.solar_wind.focal_loss
   :members:
   :show-inheritance:
   :undoc-members:

Visualization
-------------

.. automodule:: sdofmv2.tasks.solar_wind.visualization
   :members:
   :show-inheritance:
   :undoc-members:

.. Module contents
.. ---------------

.. .. automodule:: sdofmv2.tasks.solar_wind
..    :members:
..    :show-inheritance:
..    :undoc-members:
