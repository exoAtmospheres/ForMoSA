.. _scaling:

Scaling
=======

This page gathers a few implementation-level notes that are easy to lose inside
the tutorials.

Analytical Vs Physical Scaling
------------------------------

The v2.0.0 analysis stack distinguishes between two broad cases when rebuilding
best-fit models:

* if radius is part of the fitted parameter set, the model carries its physical
  scaling directly;
* if radius is not fitted, ForMoSA can analytically rescale the model when
  reconstructing native best-fit outputs.

That behaviour is implemented in :class:`ForMoSA.nested_sampling.ns_analysis.NSAnalysis`.

Good Practice With MOSAIC
-------------------------

When fitting multiple observations together:

* decide which settings are truly global before relying on one-value
  broadcasting;
* keep the observation files clearly separated and labelled;
* remember that adaptation parallelism is controlled by ``ConfigAdapt.n_jobs``,
  not by the nested sampler.

Status
------

* [A fuller analytical-vs-physical scaling discussion can be added here once the
  maintainers decide the preferred scientific framing.]
