.. _transform:

Model Transformations
=====================

Observed Model & Parameters
++++++++++++++++++++++++++++

Data containers for parameter draws and their corresponding model spectra
produced during nested sampling.

.. automodule:: ForMoSA.transform.observed
	:members:
	:undoc-members:
	:show-inheritance:

Apply Effects
+++++++++++++

Static helpers that apply individual physical and observational effects
(radial velocity, v sin i, reddening, scaling, etc.).

.. automodule:: ForMoSA.transform.apply_effects
	:members:
	:undoc-members:
	:show-inheritance:

Spectroscopic Effects
+++++++++++++++++++++

Orchestrates the full chain of physics and observational effects for
spectroscopic data.

.. automodule:: ForMoSA.transform.spectroscopic_effects
	:members:
	:undoc-members:
	:show-inheritance:

Photometric Effects
+++++++++++++++++++

Orchestrates the full chain of physics and observational effects for
photometric data.

.. automodule:: ForMoSA.transform.photometric_effects
	:members:
	:undoc-members:
	:show-inheritance:
