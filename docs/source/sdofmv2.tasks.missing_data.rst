Downstream App: Missing Data
============================

Data Module
------------

.. autoclass:: sdofmv2.tasks.missing_data.missing_data_module.MissingDataModel
   :members:
   :show-inheritance:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      forward
      forward_random_channel_drop
      training_step
      validation_step

Neck Module
-----------

.. autoclass:: sdofmv2.tasks.missing_data.necks.Norm2d
   :members:
   :show-inheritance:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      forward

.. autoclass:: sdofmv2.tasks.missing_data.necks.ConvTransformerTokensToEmbeddingNeck
   :members:
   :show-inheritance:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      forward

Wrap Encoder
------------

.. autoclass:: sdofmv2.tasks.missing_data.wrap_encoder.WrapEncoder
   :members:
   :show-inheritance:
   :no-undoc-members:

   .. rubric:: Methods

   .. autosummary::
      
      forward
      forward_features

.. Module contents
.. ---------------

.. .. automodule:: sdofmv2.tasks.missing_data
..    :members:
..    :show-inheritance:
..    :undoc-members:
