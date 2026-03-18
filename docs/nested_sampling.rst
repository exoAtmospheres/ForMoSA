.. _nested_sampling:

Nested Sampling
===============

Nested Sampling Engine
++++++++++++++++++++++

Orchestrates the nested-sampling run with support for
*nestle*, *PyMultiNest*, and *UltraNest* back-ends.

.. automodule:: ForMoSA.nested_sampling.nested_sampling
	:members:
	:undoc-members:
	:show-inheritance:

NS Analysis
+++++++++++

Post-processing of nested-sampling products: best-fit reconstruction,
CCF computation, RV–v sin i maps, and confidence intervals.

.. automodule:: ForMoSA.nested_sampling.ns_analysis
	:members:
	:undoc-members:
	:show-inheritance:

Results
+++++++

Data container that stores the raw output of a nested-sampling run.

.. automodule:: ForMoSA.nested_sampling.results
	:members:
	:undoc-members:
	:show-inheritance:

Prior Functions
+++++++++++++++

Low-level prior sampling functions used internally by the
:mod:`~ForMoSA.parameter.prior` classes.

.. automodule:: ForMoSA.utils.prior_functions
	:members:
	:undoc-members:
	:show-inheritance:

Likelihood Functions
++++++++++++++++++++

Log-likelihood functions available for nested sampling.

.. automodule:: ForMoSA.utils.logL_functions
	:members:
	:undoc-members:
	:show-inheritance:
