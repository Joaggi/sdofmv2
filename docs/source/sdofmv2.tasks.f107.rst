Downstream App: F10.7
=====================

Data Module
-----------

.. autoclass:: sdofmv2.tasks.f107.f107_datamodule.EmbSolarProxyDataset
   :members:
   :show-inheritance:
   :no-undoc-members:

.. autoclass:: sdofmv2.tasks.f107.f107_datamodule.EmbSolarProxyDataModule
   :members:
   :show-inheritance:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      setup
      test_dataloader
      train_dataloader
      val_dataloader

Model Module
------------

.. autoclass:: sdofmv2.tasks.f107.f107_module.MultiLayerPerceptron
   :members:
   :show-inheritance:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      forward
      on_before_optimizer_step
      on_train_start
      test_step
      training_step
      validation_step

.. Module contents
.. ---------------

.. .. automodule:: sdofmv2.tasks.f107
..    :members:
..    :show-inheritance:
..    :undoc-members:
