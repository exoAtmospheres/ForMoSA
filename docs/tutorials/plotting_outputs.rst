.. _tutorials-plotting:

Advanced Plotting And Outputs
=============================

The public plotting surface in v2.0.0 is split between ``NSResults``,
``NSAnalysis``, and the convenience methods exposed by
:class:`ForMoSA.Analysis`.

Built-In Figures
----------------

``analysis.plot(analysis.ns.results)`` produces:

* a corner plot;
* posterior chains;
* a radar plot;
* a best-fit comparison figure.

Additional high-resolution diagnostics are available through:

* ``analysis.plot_ccf(rv_grid)``
* ``analysis.plot_rv_vsini_map(rv_grid, vsini_grid)``

Working With Results In Python
------------------------------

The current codebase exposes several reusable objects for post-processing:

* ``analysis.ns.results.median_parameters`` for weighted posterior medians
* ``analysis.ns.results.param_samples_dict`` for posterior samples by parameter
* ``analysis.ns_analysis.best_fit`` for adapted best-fit models
* ``analysis.ns_analysis.native_best_fit`` for best-fit parameters applied to a
  restricted native grid
* ``analysis.ns_analysis.best_fit_interval(perc=0.68)`` for confidence bands

Plot Styling
------------

Observation colours are assigned automatically from each observation's central
wavelength when an :class:`ForMoSA.Analysis` instance is created. Those colours
are then propagated to the restricted observations used during plotting.

This is the verified colour-handling path in the current plotting stack.

Planned Additions
-----------------

* [Notebook on modifying plots after the main run.]
* [Examples for BIC, reduced chi-squared, and molecular-absorption post-analysis
  once those public workflows are documented and verified.]
* [Notes on interactive ``matplotlib_qt`` usage, if that workflow is adopted and
  documented by the maintainers.]
