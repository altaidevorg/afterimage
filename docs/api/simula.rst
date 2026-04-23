Simula / OpenSimula
===================

Experimental Simula-style synthetic data pipeline (`afterimage.simula`).

.. seealso::

   * :doc:`../opensimula` — narrative guide and monitoring notes
   * :doc:`../monitoring` — :class:`~afterimage.monitoring.GenerationMonitor` usage and export

.. autoclass:: afterimage.simula.OpenSimula
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: afterimage.simula.SimulaInstructionGeneratorCallback
   :members:
   :undoc-members:
   :show-inheritance:

Checkpointing and export
------------------------

.. autoclass:: afterimage.simula.Checkpointer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: afterimage.simula.SimulaCheckpoint
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: afterimage.simula.OpenSimulaRunConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: afterimage.simula.OpenSimulaManifest
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: afterimage.simula.save_checkpoint

.. autofunction:: afterimage.simula.load_checkpoint

.. autofunction:: afterimage.simula.push_checkpoint_to_hub

.. autofunction:: afterimage.simula.pull_checkpoint_from_hub

.. autofunction:: afterimage.simula.append_datapoints_jsonl

.. autofunction:: afterimage.simula.configure_example_console

.. autofunction:: afterimage.simula.silence_noisy_third_party_loggers
