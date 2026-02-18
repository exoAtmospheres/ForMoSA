import logging

from ForMoSA.nested_sampling.Prior import Prior
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.enums import VsiniFunction
from ForMoSA.core.loggings import setup_logging


class Parameter(object):
    '''
    ForMoSA Parameter class. Handles a single parameter for the nested sampling algorithm.

    Parameters
    ----------
    name                       (str): Name of the parameter ('par1', 'par2', 'rv', 'd', ...)
    prior                    (Prior): Prior object associated with the parameter (UniformPrior, GaussianPrior, ConstantPrior, LogUniformPrior)
    vsini_function   (VsiniFunction): Vsini function used for the prior (required if name starts with 'vsini').

    Authors: Allan Denis
    '''

    def __init__(self, name: str, prior: Prior, vsini_function: VsiniFunction | None = None, logger: logging.Logger | None = None, log_level: str='INFO'):
        if not isinstance(prior, Prior):
            raise ForMoSAError(f"<Parameter '{name}' must be initialized with a Prior object>")

        if name.startswith('vsini') and vsini_function is None:
            raise ForMoSAError("<Vsini parameter needs a vsini_function>")

        self._name = name
        self._prior = prior
        self._vsini_function = vsini_function
        self._theta = None
        self._logger = logger or setup_logging(log_level, name='Parameter')

    # ==================================================
    # Representation
    # ==================================================

    def __repr__(self):
        return f"Parameter(name={self.name}, prior={self.prior}, vsini_function={self.vsini_function})"

    # ==================================================
    # Properties
    # ==================================================

    @property
    def logger(self) -> logging.Logger:                # Logger
        return self._logger

    @property
    def name(self) -> str:                             # Name of the parameter
        return self._name

    @property
    def prior(self) -> Prior:                          # Prior object associated with the parameter
        return self._prior

    @property
    def vsini_function(self) -> str:                   # Vsini function used for the prior
        return self._vsini_function

    @vsini_function.setter
    def vsini_function(self, value: str):              # Setter for vsini_function
        self._vsini_function = value

    @property
    def is_fixed(self) -> bool:                        # Whether the parameter is fixed (constant prior) or free
        return self._prior._is_fixed

    @property
    def theta(self) -> float:                          # Current value of the parameter
        return self._theta

    # ==================================================
    # Methods
    # ==================================================

    def apply_prior(self, theta: float) -> float:
        '''
        Apply the prior to a parameter given a random value theta uniformly drawn from [0, 1].

        Parameters
        ----------
        theta (float): Value randomly drawn between 0 and 1.

        Returns
        -------
        float: Transformed parameter value according to the prior.

        Authors: Allan Denis
        '''

        self._theta = self._prior.sample(theta)
        return self._theta