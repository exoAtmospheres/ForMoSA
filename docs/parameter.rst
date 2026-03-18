.. _parameter:

Parameters
==========

Parameter
+++++++++

Represents a single nested-sampling parameter with its prior, kind, and scope.

.. automodule:: ForMoSA.parameter.parameter
	:members:
	:undoc-members:
	:show-inheritance:

Parameter Set
+++++++++++++

Container for the full set of parameters explored during nested sampling.

.. automodule:: ForMoSA.parameter.parameter_set
	:members:
	:undoc-members:
	:show-inheritance:

Priors
++++++

Prior distribution classes used to define parameter search ranges.

.. inheritance-diagram:: ForMoSA.parameter.prior.UniformPrior ForMoSA.parameter.prior.LogUniformPrior ForMoSA.parameter.prior.ConstantPrior ForMoSA.parameter.prior.GaussianPrior
   :parts: 1

.. automodule:: ForMoSA.parameter.prior
	:members:
	:undoc-members:
	:show-inheritance:
