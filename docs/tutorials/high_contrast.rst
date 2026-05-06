.. _tutorials-high-contrast:

High-Contrast Spectroscopy
==========================

ForMoSA v2.0.0 includes verified high-contrast support in the observation and
analysis layers. In the current codebase, high-contrast mode is activated when a
spectroscopic observation provides one or more stellar-flux series through keys
prefixed by ``STAR_FLUX``.

Verified Input Features
-----------------------

The loader recognises the following high-contrast-related spectroscopic content:

* ``STAR_FLUX*`` to provide one or more stellar reference vectors
* ``SYSTEMATICS*`` for one or more systematics vectors
* ``TRANSMISSION`` for atmospheric or instrumental transmission
* ``FLUX_CONT`` and ``STAR_FLUX_CONT`` for saved continua
* ``WAVE_CONT`` and ``RES_CONT`` for continuum metadata

The code also warns that covariance handling is not implemented for
high-contrast observations. If a covariance matrix is present together with
stellar-flux inputs, the covariance is not used.

Status Of The Tutorial Material
-------------------------------

* [AF Lep b HiRISE notebook pending.]
* [Notebook from Pablo to be integrated.]
* [Recommended units, systematics conventions, and file-format examples to be
  added here once they are validated by the maintainers.]

Related Functionality
---------------------

Once a fit is complete, high-resolution spectroscopic analyses can also use:

* :meth:`ForMoSA.Analysis.plot_ccf`
* :meth:`ForMoSA.Analysis.plot_rv_vsini_map`
