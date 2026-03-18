.. _observation:

Observations
============

Observation Base
++++++++++++++++

Abstract base class shared by all observation types.

.. automodule:: ForMoSA.observation.observation_base
	:members:
	:undoc-members:
	:show-inheritance:

Observation Loader
++++++++++++++++++

Factory that creates :class:`~ForMoSA.observation.observation_base.Observation`
instances from FITS files, dictionaries, or raw attributes.

.. automodule:: ForMoSA.observation.observation_loader
	:members:
	:undoc-members:
	:show-inheritance:

Spectral Observation
++++++++++++++++++++

.. automodule:: ForMoSA.observation.observation_spectroscopy
	:members:
	:undoc-members:
	:show-inheritance:

Photometry Observation
++++++++++++++++++++++

.. automodule:: ForMoSA.observation.observation_photometry
	:members:
	:undoc-members:
	:show-inheritance:

Observation Set
+++++++++++++++

Container that groups multiple observations together.

.. automodule:: ForMoSA.observation.observation_set
	:members:
	:undoc-members:
	:show-inheritance:
