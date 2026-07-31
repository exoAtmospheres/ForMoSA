.. _whats_new:

What's New in v2.0.0
====================

Version 2.0.0 is a **complete redesign** of ForMoSA. The package has been rebuilt
from the ground up around a modern, object-oriented API. While the core physics
and nested-sampling approach remain the same, almost every interface has changed.
Users coming from v1.x should read this page carefully before upgrading.

.. contents:: On this page
   :local:
   :depth: 2

----

.. _whats_new.why:

Why a new major version?
------------------------

The v1.x series accumulated several functional improvements (MOSAIC multi-instrument
support, UltraNest, high-contrast models, …), but the underlying code remained
largely procedural and tightly coupled to a single INI configuration file.
This made it hard to:

* use ForMoSA interactively in a notebook;
* compose multiple steps programmatically;
* extend or test individual components in isolation.

v2.0.0 addresses these limitations with a class-based rewrite that keeps the
scientific capabilities of v1.x and adds significant new ones.

----

.. _whats_new.api:

New class-based API
-------------------

The most visible change is the introduction of the :class:`~ForMoSA.Analysis`
class as the single entry point for every analysis. In v1.x, a typical workflow
was driven by a ``main.py`` script that called two standalone functions:

.. code-block:: python

   # v1.x (old)
   from ForMoSA.global_file import GlobFile
   from ForMoSA.adapt.adapt_obs_mod import launch_adapt
   from ForMoSA.nested_sampling.nested_sampling import launch_nested_sampling

   global_params = GlobFile("config.ini")
   launch_adapt(global_params, adapt_model=True)
   launch_nested_sampling(global_params)

In v2.0.0 the same workflow becomes:

.. code-block:: python

   # v2.0.0 (new)
   from ForMoSA import Analysis
   from ForMoSA.config.global_config import (
       ConfigPath, ConfigAdapt, ConfigInversion, ConfigParameters
   )

   config_path = ConfigPath(
       observation_path=["path/to/observation.fits"],
       adapt_store_path="path/to/adapted_grid/",
       result_path="path/to/results/",
       model_path="path/to/model_grid.nc",
   )

   analysis = Analysis(config_path)
   analysis.adapt(ConfigAdapt(), ConfigInversion())
   analysis.nested_sampling(
       ConfigParameters(
           par1=["uniform", "500", "3000"],
           par2=["uniform", "2.5", "5.5"],
           r=["uniform", "0.5", "3.0"],
           d=["constant", "50"],
       )
   )
   analysis.plot(analysis.ns.results)

The ``Analysis`` object manages the full lifecycle — loading observations and
model grids, adapting the grid, running nested sampling, and producing plots —
so you never have to worry about passing state between functions.

----

.. _whats_new.config:

Python dataclasses for configuration
-------------------------------------

v1.x relied on a single ``config.ini`` file, read through the ``GlobFile``
class, to control every aspect of an analysis.  v2.0.0 replaces this with a
set of lightweight Python **dataclasses** that can be instantiated directly in
code *or* loaded from an INI file:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Dataclass
     - Controls
   * - :class:`~ForMoSA.config.global_config.ConfigPath`
     - File paths (observations, grid, results)
   * - :class:`~ForMoSA.config.global_config.ConfigAdapt`
     - Grid adaptation settings (method, resolution)
   * - :class:`~ForMoSA.config.global_config.ConfigInversion`
     - Nested-sampling algorithm, likelihood, priors
   * - :class:`~ForMoSA.config.global_config.ConfigParameters`
     - Prior distributions for each fitted parameter

This makes it straightforward to build configuration programmatically, inspect
and modify individual values, and loop over parameter grids in notebooks.

----

.. _whats_new.structure:

Restructured package layout
----------------------------

The internal package layout has been reorganised into well-separated modules:

.. code-block:: text

   ForMoSA/
   ├── analysis.py          ← new: main Analysis class
   ├── config/              ← new: dataclass-based configuration
   ├── core/                ← new: enums, error types, logging
   ├── filter/              ← new: SVO filter retrieval & caching
   ├── grid/                ← replaces adapt/: model grids & subgrids
   ├── nested_sampling/     ← retained & extended
   ├── observation/         ← new: typed observation classes
   ├── parameter/           ← new: parameter & prior objects
   ├── transform/           ← new: physics/observational effect pipeline
   └── utils/               ← replaces utils_spec.py

Old top-level scripts (``main.py``, ``global_file.py``, ``utils_spec.py``) and
the flat ``adapt/`` directory are no longer present. Their functionality lives
in the modules listed above.

----

.. _whats_new.observations:

Typed observation classes
--------------------------

Observations are now represented by proper Python classes instead of raw NumPy
arrays read from an INI path list:

* :class:`~ForMoSA.observation.observation_spectroscopy.SpectralObservation`
  — single spectroscopic dataset
* :class:`~ForMoSA.observation.observation_photometry.PhotometryObservation`
  — photometric dataset (filter list)
* :class:`~ForMoSA.observation.observation_set.ObservationSet`
  — container that groups any mix of the above

The ``ObservationSet`` is built automatically from your ``.fits`` files when
you construct an ``Analysis`` object, and it handles the MOSAIC multi-instrument
case transparently.

----

.. _whats_new.filters:

Automatic photometry filter retrieval
---------------------------------------

v2.0.0 introduces a :class:`~ForMoSA.filter.filter.PhotometryFilter` service that
automatically downloads and caches filter transmission curves from the
`SVO Filter Profile Service <https://svo2.cab.inta-csic.es/theory/fps/>`_. You
no longer have to manage filter files manually — just provide the facility,
instrument, and filter identifier in your observation file and ForMoSA handles
the rest.

----

.. _whats_new.plotting:

Extended plotting
-----------------

The :meth:`~ForMoSA.Analysis.plot` method and the underlying
:class:`~ForMoSA.nested_sampling.plotting.Plotting` class now produce:

* **Corner plot** — joint and marginal posterior distributions
* **Chain diagnostics** — nested-sampling chain traces
* **Radar diagram** — visual comparison of multiple targets
* **Best-fit spectrum** — observed data versus best-fit model with 1-σ and 2-σ
  confidence intervals
* **CCF** — Cross-Correlation Function via :meth:`~ForMoSA.Analysis.plot_ccf`
* **RV–v sin i map** — 2-D log-likelihood map via
  :meth:`~ForMoSA.Analysis.plot_rv_vsini_map`

The CCF and RV–v sin i plots are entirely new in v2.0.0 and require a completed
nested-sampling run.

----

.. _whats_new.logging:

Structured logging
------------------

v2.0.0 adds a proper logging infrastructure built on the Python standard
``logging`` module. The verbosity level can be controlled when instantiating
``Analysis``:

.. code-block:: python

   analysis = Analysis(config_path, log_level="debug")   # verbose
   analysis = Analysis(config_path, log_level="warning")  # quiet

----

.. _whats_new.errors:

Typed error handling
---------------------

All internal errors now raise :class:`~ForMoSA.core.errors.ForMoSAError` (a
subclass of ``Exception``) instead of generic exceptions or silent failures.
This makes it much easier to catch ForMoSA-specific problems in your own code:

.. code-block:: python

   from ForMoSA.core.errors import ForMoSAError

   try:
       analysis = Analysis(config_path)
   except ForMoSAError as e:
       print(f"ForMoSA configuration error: {e}")

----

.. _whats_new.migration:

Migrating from v1.x
--------------------

The table below summarises the mapping from the v1.x public interface to v2.0.0.

+----------------------------------------------+----------------------------------------------+
| v1.x                                         | v2.0.0                                       |
+==============================================+==============================================+
| ``GlobFile("config.ini")``                   | ``ConfigPath(…)`` + ``Analysis(config_path)``|
+----------------------------------------------+----------------------------------------------+
| ``launch_adapt(global_params)``              | ``analysis.adapt(config_adapt, config_inv)`` |
+----------------------------------------------+----------------------------------------------+
| ``launch_nested_sampling(global_params)``    | ``analysis.nested_sampling(config_params)``  |
+----------------------------------------------+----------------------------------------------+
| ``plotting_class.py`` standalone calls       | ``analysis.plot(analysis.ns.results)``       |
+----------------------------------------------+----------------------------------------------+
| ``ForMoSA/adapt/``                           | ``ForMoSA/grid/``                            |
+----------------------------------------------+----------------------------------------------+
| ``ForMoSA/utils_spec.py``                    | ``ForMoSA/utils/``                           |
+----------------------------------------------+----------------------------------------------+
| ``ForMoSA/global_file.py``                   | ``ForMoSA/config/global_config.py``          |
+----------------------------------------------+----------------------------------------------+

.. important::

   v2.0.0 is **not backwards-compatible** with v1.x. Scripts and notebooks
   written for v1.x will need to be updated. If you need the old behaviour
   in the short term, pin your installation to ``formosa==1.1.6``.

----

.. _whats_new.summary:

Summary of changes at a glance
-------------------------------

+---------------------------------------+------------------------------------+---------------------------------+
| Feature                               | v1.x                               | v2.0.0                          |
+=======================================+====================================+=================================+
| Entry point                           | ``main.py`` script                 | ``Analysis`` class              |
+---------------------------------------+------------------------------------+---------------------------------+
| Configuration                         | ``config.ini`` + ``GlobFile``      | Python dataclasses              |
+---------------------------------------+------------------------------------+---------------------------------+
| Observations                          | Raw arrays from INI paths          | Typed ``ObservationSet``        |
+---------------------------------------+------------------------------------+---------------------------------+
| Model grid handling                   | ``adapt/`` functions               | ``grid/`` subgrid classes       |
+---------------------------------------+------------------------------------+---------------------------------+
| Photometry filters                    | Bundled ``.npz`` files             | SVO auto-download + cache       |
+---------------------------------------+------------------------------------+---------------------------------+
| Nested-sampling back-ends             | nestle, PyMultiNest, UltraNest     | same (+ improved interface)     |
+---------------------------------------+------------------------------------+---------------------------------+
| CCF / RV-vsini analysis               | ✗ not available                    | ✓ new                           |
+---------------------------------------+------------------------------------+---------------------------------+
| Logging                               | ``print`` statements               | Python ``logging`` module       |
+---------------------------------------+------------------------------------+---------------------------------+
| Error handling                        | Generic exceptions                 | ``ForMoSAError``                |
+---------------------------------------+------------------------------------+---------------------------------+
| Package version                       | 1.1.6                              | **2.0.0**                       |
+---------------------------------------+------------------------------------+---------------------------------+
