.. _tutorials-general:

General Workflow
================

ForMoSA 2.0.0 supports two closely related user-facing workflows:

* the recommended class-based API using :class:`ForMoSA.Analysis` and Python
  dataclasses;
* an INI-driven workflow using
  :class:`ForMoSA.config.global_config.ConfigGenerator` and
  :class:`ForMoSA.config.global_config.ConfigLoader`.

Both workflows ultimately feed the same core classes.

Recommended File Layout
-----------------------

ForMoSA does not require a single fixed directory structure, but the codebase
and bundled examples work cleanly with a separation between native grids,
per-project observations, adapted grids, and results:

.. code-block:: text

   /your/project/root/
   ├── atm_grids/
   │   └── model_grid.nc
   ├── targets/
   │   └── target_name/
   │       ├── data/
   │       │   ├── obs_1.fits
   │       │   └── obs_2.fits
   │       ├── adapted_grid/
   │       └── results/
   └── notebooks/  [optional]

At runtime, the only paths that the public API requires are:

* one or more observation FITS files;
* one model-grid NetCDF file;
* one directory where adapted grids can be written;
* one directory where results can be written.

That mapping is expressed directly through
:class:`ForMoSA.config.global_config.ConfigPath`.

Two Configuration Routes
------------------------

Dataclass route
+++++++++++++++

The class-based route is the clearest entry point for new analyses:

.. code-block:: python

   from ForMoSA import Analysis
   from ForMoSA.config.global_config import (
       ConfigPath,
       ConfigAdapt,
       ConfigInversion,
       ConfigParameters,
       Config_NS,
   )

   config_path = ConfigPath(
       observation_path=["target/data/obs_1.fits"],
       adapt_store_path="target/adapted_grid",
       result_path="target/results",
       model_path="atm_grids/model_grid.nc",
   )

   analysis = Analysis(config_path)
   analysis.adapt(ConfigAdapt(), ConfigInversion())
   analysis.nested_sampling(
       ConfigParameters(
           par1=["uniform", "500", "3000"],
           par2=["uniform", "2.5", "5.5"],
           r=["uniform", "0.5", "3.0"],
           d=["constant", "50"],
       ),
       config_NS=Config_NS(),
   )
   analysis.plot(analysis.ns.results)

INI route
+++++++++

The repository still ships a supported INI workflow. The maintained example
notebooks use:

* :class:`ForMoSA.config.global_config.ConfigGenerator` to write a template
  ``config.ini`` file;
* :class:`ForMoSA.config.global_config.ConfigLoader` to read that file back into
  the same dataclass objects used by the class-based API.

That makes the INI workflow a convenience layer rather than a separate engine.

Observation FITS Format
-----------------------

ForMoSA accepts aliases for many column names, but the canonical keys below are
the safest names to document and generate.

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Data type
     - Required FITS content
     - Notes
   * - Photometry
     - ``WAVELENGTH``, ``FLUX``, ``ERROR``, ``FACILITY``, ``INSTRUMENT``,
       ``FILTER_ID``
     - ``WAVELENGTH_UNIT`` is optional. If it is absent, ForMoSA assumes
       micrometres.
   * - Spectroscopy
     - ``WAVELENGTH``, ``FLUX``, ``RESOLUTION``, and either ``ERROR`` or
       ``COVARIANCE``
     - ``FACILITY`` and ``INSTRUMENT`` are optional for spectroscopy; the loader
       falls back to ``unknown`` if they are missing.
   * - High-contrast spectroscopy
     - spectroscopic requirements above, plus one or more ``STAR_FLUX*`` columns
     - The presence of stellar-flux series activates high-contrast mode. Optional
       companion columns include ``SYSTEMATICS*``, ``TRANSMISSION``,
       ``FLUX_CONT``, ``STAR_FLUX_CONT``, ``WAVE_CONT``, and ``RES_CONT``.

Notes on FITS inputs:

* Photometric observations are detected from the presence of the required
  photometric keys.
* Spectroscopic observations are detected from the presence of
  ``RESOLUTION`` plus either ``ERROR`` or ``COVARIANCE``.
* A ``FILTER_ID`` column filled only with ``NA`` or empty strings is treated as
  a placeholder, and the loader falls back to spectroscopic interpretation.
* High-contrast mode is triggered when stellar flux vectors are provided.

For the existing notebook walkthroughs of the input formats, see:

* :doc:`../guidelines/input_format/obs_format`
* :doc:`../guidelines/input_format/model_format`

The legacy ``config_format.ipynb`` notebook is still present in the repository,
but it is under refresh and should not be treated as the primary v2.0.0
reference.

Knowing Your Grid File
----------------------

The v2.0.0 grid loader expects a NetCDF file readable as an ``xarray.Dataset``.
The validation rules enforced by :class:`ForMoSA.grid.grid_loader.GridLoader`
are:

* the dataset must contain a data variable named ``grid``;
* the first dimension of ``grid`` must be ``wavelength``;
* the dataset attributes must include ``key``, ``par``, ``title``, ``unit``,
  and ``res``;
* the dimensions after ``wavelength`` must match ``attrs["key"]`` exactly;
* ``attrs["res"]`` must have the same length as the wavelength axis;
* if ``wave_unit`` is provided, it must be a string.

In practice, that means a valid model-grid file must provide both the spectral
axis and the metadata that let ForMoSA map generic grid parameters such as
``par1`` or ``par2`` onto meaningful titles during the analysis.

Knowing Your Configuration
--------------------------

The main configuration surface is split across four dataclasses and one optional
nested-sampling settings container:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Object / INI section
     - Purpose
     - Key fields verified in the code
   * - ``ConfigPath`` / ``[config_path]``
     - File and output locations
     - ``observation_path``, ``adapt_store_path``, ``result_path``,
       ``model_path``
   * - ``ConfigAdapt`` / ``[config_adapt]``
     - Grid and observation adaptation
     - ``method``, ``target_res_obs``, ``target_res_mod``, ``wav_cont``,
       ``res_cont``, ``backend``, ``n_jobs``
   * - ``ConfigInversion`` / ``[config_inversion]``
     - Inference setup
     - ``logL_type``, ``wav_fit``, ``ns_algo``, ``npoints``,
       ``hc_lower_bounds_lsq``, ``hc_higher_bounds_lsq``
   * - ``ConfigParameters`` / ``[config_parameters]``
     - Prior definitions
     - ``par1`` to ``par4``, ``r``, ``d``, ``alpha``, ``bb_T``, ``rv``,
       ``vsini``, ``ld``
   * - ``Config_NS`` / backend-specific sections
     - Optional backend tuning
     - ``config_nestle``, ``config_pymultinest``, ``config_ultranest``

Supported inference values verified in the current source:

* nested-sampling backends: ``nestle``, ``pymultinest``, ``ultranest``
* likelihood labels: ``chi2``, ``chi2_covariance``, ``chi2_noisescaling``,
  ``chi2_noisescaling_covariance``, ``CCF_Brogi``, ``CCF_Zucker``,
  ``CCF_custom``
* adaptation backends: ``loky``, ``multiprocessing``, ``threading``,
  ``sequential``, ``dask``, ``ray``

MOSAIC Behaviour
----------------

Several configuration fields in ``ConfigAdapt`` and ``ConfigInversion`` accept
either:

* one value, which ForMoSA will broadcast to all observations; or
* one value per observation, when fitting multiple datasets together.

That broadcasting behaviour is part of the v2.0.0 code and is the basis of the
MOSAIC workflow.

Parallelisation Note
--------------------

``ConfigAdapt.n_jobs`` controls parallel work during the grid-adaptation stage.
It does not control the nested-sampling backend itself.

That distinction matters in practice:

* increase ``n_jobs`` to speed up grid adaptation;
* configure MPI or backend-specific settings separately when using
  ``pymultinest`` or another sampler.
