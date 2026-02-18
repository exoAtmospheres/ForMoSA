import logging

from ForMoSA.parameter.prior import Prior
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.loggings import setup_logging
from ForMoSA.core.enums import VsiniFunction, ParameterKind


class Parameter(object):
    '''
    ForMoSA Parameter class. Handles a single parameter for the nested sampling algorithm.

    Parameters
    ----------
    name                       (str): Name of the parameter ('par1', 'par2', 'rv', 'd', ...)
    prior                    (Prior): Prior object associated with the parameter (UniformPrior, GaussianPrior, ConstantPrior, LogUniformPrior)
    kind             (ParameterKind): Type of parameter used to identify the parameter
    scope                      (str): 'global' if this is a global parameter or 'local' if it is applied to specific observations
    obs_index           (int | None): Index of the obervation the parameter is applied to (None if scope is 'global')
    title               (str | None): Title of the parameter (used to connect grid parameters to their associated title)
    vsini_function   (VsiniFunction): Vsini function used for the prior (required if name starts with 'vsini')
    logger          (logging.Logger): Logger
    log_level                  (str): Level of the Logger


    Authors: Allan Denis
    '''

    def __init__(self, name: str, prior: Prior, kind: ParameterKind, scope: str = 'global', obs_index: list[int] | None = None, title: str | None = None, vsini_function: VsiniFunction | None = None, logger: logging.Logger | None = None, log_level: str='INFO'):

        self._name = name
        self._prior = prior
        self._kind = kind
        self._scope = scope
        self._obs_index = obs_index
        self._vsini_function = vsini_function
        self._title = title
        self._logger = logger or setup_logging(log_level, name = 'Parameter')

        self._validate()

    # ======================
    # Representation
    # ======================

    def __repr__(self):
        return f"Parameter(name={self.name}, kind={self.kind}, prior={self.prior}, vsini_function={self.vsini_function}, scope={self.scope}, obs_index={self.obs_index})"

    # ======================
    # Properties
    # ======================

    @property
    def logger(self) -> logging.Logger:                # Logger
        return self._logger

    @property
    def name(self) -> str:                             # Name of the parameter
        return self._name

    @property
    def kind(self) -> ParameterKind:                   # Parameter type
        return self._kind

    @property
    def scope(self) -> str:                            # Scope ('global' or 'local')
        return self._scope

    @property
    def is_local(self) -> bool:                        # Whether the parameter is local
        return self.scope == 'local'

    @property
    def title(self) -> str:                            # Title
        return self._title

    @property
    def obs_index(self) -> list[int] | None:           # Index of the observation the parameter is applied to
        return self._obs_index

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
        return self._prior.is_fixed

    @property
    def to_dict(self) -> dict:                         # Dictionary representation of the parameter
        return {
            'name': self.name,
            'prior': self.prior.to_dict,
            'kind': self.kind.kind,
            'scope': self.scope,
            'obs_index': self.obs_index,
            'title': self.title,
            'vsini_function': self.vsini_function
            }

    # ======================
    # Clas methods
    # ======================

    @classmethod
    def from_dict(cls, data: dict, logger: logging.Logger | None = None, log_level: str = 'INFO') -> "Parameter":
        '''
        Reconstruct a Parameter from a dictionary of Parameter.

        Parameters
        ----------
        data              (dict): Dictionary containing parameter parameters
        logger  (logging.Logger): Logger
        log_level          (str): Level of the Logger


        Returns
        -------
        Parameter: An instance of class Parameter

        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name='Parameter')
        logger.debug('Extract Parameter from dictionary')

        name, prior, kind, scope, title, obs_index, vsini_function = data['name'], data['prior'], data['kind'], data['scope'], data['title'], data['obs_index'], data['vsini_function']
        logger.info(f'    {kind} parameter detected with name {name}')

        return cls(name=name, prior=Prior.from_dict(prior), kind=ParameterKind[kind], scope=scope, title=title, obs_index=obs_index, vsini_function=vsini_function, logger=logger)

    # ======================
    # Methods
    # ======================

    def _validate(self) -> None:
        '''
        Validation of the parameter.

        Authors: Allan Denis
        '''

        if not isinstance(self.prior, Prior):
            raise ForMoSAError(f"Parameter {self.name} must be initialized with a Prior object", self._logger)

        if not isinstance(self.kind, ParameterKind):
            raise ForMoSAError(f"Parameter '{self.name}' must have a valid ParameterKind", self._logger)

        if self.kind is ParameterKind.VSINI:
            valid_vsini_functions = list(VsiniFunction)
            if self.vsini_function is None or self.vsini_function not in valid_vsini_functions:
                raise ForMoSAError(f"VSINI parameter '{self.name}' requires a vsini_function amongst {valid_vsini_functions}. Got {self.vsini_function} instead", self._logger)

        if self.scope not in ('global', 'local'):
            raise ForMoSAError(f"Parameter '{self.name}' has invalid scope '{self.scope}'. Must be 'global' or 'local'", self._logger)

        if self.scope == 'global' and self.obs_index is not None:
            raise ForMoSAError(f"Global parameter '{self.name}' must have obs_index = None", self._logger)

        if self.scope == 'local' and self.obs_index is None:
            raise ForMoSAError(f"Local parameter '{self.name}' must have a valid obs_index", self._logger)

        if self.obs_index is not None:
            if not isinstance(self.obs_index, list):
                raise ForMoSAError(f"<Parameter {self.name} must have a list for obs_index. Got {type(self.obs_index)} instead>", self.logger)

        if self._title is None:
            self._title = self.name