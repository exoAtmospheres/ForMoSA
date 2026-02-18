import logging

from abc import ABC, abstractmethod
from ForMoSA.core.enums import PriorType
from ForMoSA.utils import prior_functions
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.loggings import setup_logging


class Prior(ABC):
    """
    Abstract base class for prior distributions.

    Authors: Allan Denis
    """

    def __init__(self, logger: logging.Logger | None = None, log_level: str='INFO') -> None:
        self._logger = logger or setup_logging(log_level, name='Prior')

    # ==================================================
    # Properties
    # ==================================================

    @property
    def logger(self) -> logging.Logger:                # Logger
        return self._logger

    @property
    def to_dict(self) -> dict:                         # Dictionary representation of the prior
        return {
            "prior_type": self.prior_type.priortype,
            "params": self.get_params_dict()
        }

    # ==================================================
    # Class Methods
    # ==================================================

    @classmethod
    def from_dict(cls, data: dict, logger: logging.Logger | None = None, log_level: str = 'INFO') -> "Prior":
        '''
        Reconstruct a Prior from a dictionary of Prior.

        Parameters
        ----------
        data              (dict): Dictionary containing prior parameters
        logger  (logging.Logger): Logger
        log_level          (str): Level of the Logger

        Returns
        -------
        Prior: An instance of Prior

        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name='Prior')
        logger.debug('Extract Prior from dictionary')

        prior_type = PriorType(data["prior_type"])
        params = data["params"]

        logger.info(f'      {prior_type.priortype} prior extracted ({params})')
        return cls.parse_prior(cls, prior_type, params, logger=logger, log_level=log_level)

    # ==================================================
    # Abstract Methods
    # ==================================================

    @abstractmethod
    def sample(self, theta: float) -> float:
        '''
        Sample from the prior distribution.

        Parameters
        ----------
        theta (float): A value between 0 and 1 to sample from the prior

        Returns
        -------
        float: Sampled value from the prior distribution

        Authors: Allan Denis
        '''

        pass

    @property
    @abstractmethod
    def is_fixed(self) -> bool:                                               # Whether the prior is fixed
        pass

    @property
    @abstractmethod
    def prior_type(self) -> PriorType:                                         # Type of prior
        pass

    # ==================================================
    # Methods
    # ==================================================

    def parse_prior(self, prior_type: PriorType, params: dict, logger: logging.Logger | None = None, log_level: str = 'INFO') -> 'Prior':
        '''
        Parse prior parameters into corresponding prior object.

        Parameters
        ----------
        prior_type   (PriorType): Type of the prior
        params            (dict): Dictionary of prior parameters
        logger  (logging.Logger): Logger
        log_level          (str): Level of the Logger

        Returns
        -------
        Prior: Prior object corresponding to the specified type and parameters

        Usage: Prior = Prior.parse_prior(prior_type, params)

        Authors: Allan Denis
        '''

        if prior_type.is_gaussian:
            return GaussianPrior(mean=params.get('mean'), stddev=params.get('stddev'), logger=logger, log_level=log_level)
        elif prior_type.is_uniform:
            return UniformPrior(lower=params.get('lower'), upper=params.get('upper'), logger=logger, log_level=log_level)
        elif prior_type.is_log_uniform:
            return LogUniformPrior(lower=params.get('lower'), upper=params.get('upper'), logger=logger, log_level=log_level)
        elif prior_type.is_constant:
            return ConstantPrior(value=params.get('value'), logger=logger, log_level=log_level)
        else:
            raise ForMoSAError(f'Unknown prior type: {prior_type}', self.logger)

# =========================
# Uniform Prior
# =========================

class UniformPrior(Prior):
    '''
    Class defining a Uniform prior.

    Parameters
    ----------
    lower            (float): Lower bound of the uniform prior
    upper            (float): Upper bound of the uniform prior
    logger  (logging.Logger): Logger
    log_level          (str): Level of the Logger

    Authors: Allan Denis
    '''

    def __init__(self, lower: float, upper: float, logger: logging.Logger | None = None, log_level: str='INFO') -> None:
        super().__init__(logger=logger, log_level=log_level)

        if lower is None or upper is None:
            raise ForMoSAError("Lower and upper bounds must be provided for Uniform prior", self.logger)

        self._lower = float(lower)
        self._upper = float(upper)

        self._validate()

    # ========================
    # Representation
    # ========================

    def __repr__(self):
        return f"UniformPrior(lower={self.lower}, upper={self.upper})"

    # =========================
    # Properties
    # =========================

    @property
    def lower(self) -> float:
        return self._lower

    @lower.setter
    def lower(self, value: float) -> None:
        if value >= self._upper:
            raise ForMoSAError("Lower bound must be less than upper bound for Uniform prior", self.logger)
        self._lower = value

    @property
    def upper(self) -> float:
        return self._upper

    @upper.setter
    def upper(self, value: float) -> None:
        if value <= self._lower:
            raise ForMoSAError("Upper bound must be greater than lower bound for Uniform prior", self.logger)
        self._upper = value

    @property
    def bounds(self) -> list:
        return [self.lower, self.upper]

    @property
    def prior_type(self) -> PriorType:
        return PriorType.UNIFORM

    @property
    def is_fixed(self) -> bool:
        return False

    # =========================
    # Methods
    # =========================

    def _validate(self) -> None:
        '''
        Validation

        Authors: Allan Denis
        '''

        if self.lower >= self.upper:
            raise ForMoSAError("Lower bound must be less than upper bound for Uniform prior", self.logger)

    def sample(self, theta: float) -> float:
        '''
        Sample from uniform prior and theta value.

        Parameters
        ----------
        theta (float): theta value between 0 and 1

        Returns
        -------
        float: Sampled value

        '''

        try:
            return prior_functions.uniform_prior([self.lower, self.upper], theta)
        except ForMoSAError as e:
            raise ForMoSAError(e, self.logger)

    def get_params_dict(self) -> dict:
        '''
        Return a dictionary representation of the uniform prior.

        Returns
        -------
        dict: Dictionary of the parameter

        Authors: Allan Denis
        '''

        return {"lower": self.lower, "upper": self.upper}

# =========================
# Log-Uniform Prior
# =========================

class LogUniformPrior(Prior):
    '''
    Class defining a Log-Uniform prior.

    Parameters
    ----------
    lower            (float): Lower bound of the log-uniform prior
    upper            (float): Upper bound of the log-uniform prior
    logger  (logging.Logger): Logger
    log_level          (str): Level of the Logger

    Authors: Allan Denis
    '''

    def __init__(self, lower: float, upper: float, logger: logging.Logger | None = None, log_level: str = 'INFO') -> None:
        super().__init__(logger=logger, log_level=log_level)

        if lower is None or upper is None:
            raise ForMoSAError("Lower and upper bounds must be provided for Log-Uniform prior.")

        self._lower = float(lower)
        self._upper = float(upper)

        self._validate()

    # ========================
    # Representation
    # ========================

    def __repr__(self):
        return f"LogUniformPrior(lower={self.lower}, upper={self.upper})"

    # =========================
    # Properties
    # =========================

    @property
    def lower(self) -> float:
        return self._lower

    @lower.setter
    def lower(self, value: float) -> None:
        if value <= 0 or value >= self._upper:
            raise ForMoSAError("Lower bound must be positive and less than upper bound for Log-Uniform prior", self.logger)
        self._lower = value

    @property
    def upper(self) -> float:
        return self._upper

    @upper.setter
    def upper(self, value: float) -> None:
        if value <= self._lower:
            raise ForMoSAError("Upper bound must be greater than lower bound for Log-Uniform prior", self.logger)
        self._upper = value

    @property
    def bounds(self) -> list:
        return [self.lower, self.upper]

    @property
    def prior_type(self) -> PriorType:
        return PriorType.LOG_UNIFORM

    @property
    def is_fixed(self) -> bool:
        return False

    # =========================
    # Methods
    # =========================

    def _validate(self) -> None:
        '''
        Validation.

        Authors: Allan Denis
        '''

        if self.lower <= 0 or self.upper <= 0:
            raise ForMoSAError("Lower and upper bounds must be positive for Log-Uniform prior", self.logger)
        if self.lower >= self.upper:
            raise ForMoSAError("Lower bound must be less than upper bound for Log-Uniform prior", self.logger)

    def sample(self, theta: float) -> float:
        '''
        Sample from loguniform prior and theta value.

        Parameters
        ----------
        theta (float): theta value between 0 and 1

        Returns
        -------
        float: Sampled value

        '''

        try:
            return prior_functions.loguniform_prior([self.lower, self.upper], theta)
        except ForMoSAError as e:
            raise ForMoSAError(e, self.logger)

    def get_params_dict(self) -> dict:
        '''
        Return a dictionary representation of the loguniform prior.

        Returns
        -------
        dict: Dictionary of the parameter

        Authors: Allan Denis
        '''

        return {"lower": self.lower, "upper": self.upper}

# =========================
# Constant Prior
# =========================
class ConstantPrior(Prior):
    '''
    Class defining a Constant prior.

    Parameters
    ----------
    value            (float): Constant value of the prior
    logger  (logging.Logger): Logger
    log_level          (str): Level of the Logger

    Authors: Allan Denis
    '''

    def __init__(self, value: float, logger: logging.Logger | None = None, log_level: str = 'INFO') -> None:
        super().__init__(logger=logger, log_level=log_level)

        if value is None:
            raise ForMoSAError("Value must be provided for Constant prior", self.logger)

        self._value = float(value)

    # ========================
    # Representation
    # ========================

    def __repr__(self):
        return f"ConstantPrior(value={self.value})"

    # =========================
    # Properties
    # =========================

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, val: float) -> None:
        self._value = val

    @property
    def prior_type(self) -> PriorType:
        return PriorType.CONSTANT

    @property
    def is_fixed(self) -> bool:
        return True

    @property
    def bounds(self) -> float:
        return self.value

    # =========================
    # Methods
    # =========================

    def sample(self, theta: float) -> float:
        '''
        Sample from constant prior and theta value.

        Parameters
        ----------
        theta (float): theta value between 0 and 1

        Returns
        -------
        float: Sampled value

        '''

        return self.value

    def get_params_dict(self) -> dict:
        '''
        Return a dictionary representation of the uniform prior.

        Returns
        -------
        dict: Dictionary of the parameter

        Authors: Allan Denis
        '''

        return {"value": self.value}

# =========================
# Gaussian Prior
# =========================
class GaussianPrior(Prior):
    '''
    Class defining a Gaussian prior.

    Parameters
    ----------
    mean             (float): Mean of the Gaussian prior
    stddev           (float): Standard deviation of the Gaussian prior
    logger  (logging.Logger): Logger
    log_level          (str): Level of the Logger

    Authors: Allan Denis
    '''

    def __init__(self, mean: float, stddev: float, logger: logging.Logger | None = None, log_level: str = 'INFO') -> None:
        super().__init__(logger=logger, log_level=log_level)

        if mean is None or stddev is None:
            raise ForMoSAError("Mean and standard deviation must be provided for Gaussian prior", self.logger)

        self._mean = float(mean)
        self._stddev = float(stddev)

        self._validate()

    # ========================
    # Representation
    # ========================

    def __repr__(self):
        return f"GaussianPrior(mean={self.mean}, stddev={self.stddev})"

    # =========================
    # Properties
    # =========================

    @property
    def mean(self) -> float:
        return self._mean

    @mean.setter
    def mean(self, value: float) -> None:
        self._mean = value

    @property
    def stddev(self) -> float:
        return self._stddev

    @stddev.setter
    def stddev(self, value: float) -> None:
        if value <= 0:
            raise ForMoSAError("Standard deviation must be positive for Gaussian prior", self.logger)
        self._stddev = value

    @property
    def prior_type(self) -> PriorType:
        return PriorType.GAUSSIAN

    @property
    def is_fixed(self) -> bool:
        return False

    @property
    def bounds(self) -> tuple[float, float]:
        return self.mean, self.stddev

    # =========================
    # Methods
    # =========================

    def _validate(self) -> None:
        '''
        Validation.

        Authors: Allan Denis
        '''

        if self.stddev <= 0:
            raise ForMoSAError("Standard deviation must be positive for Gaussian prior", self.logger)

    def sample(self, theta: float) -> float:
        '''
        Sample from gaussian prior and theta value.

        Parameters
        ----------
        theta (float): theta value between 0 and 1

        Returns
        -------
        float: Samples value

        '''

        try:
            return prior_functions.gaussian_prior(self.mean, self.stddev, theta)
        except ForMoSAError as e:
            raise ForMoSAError(e, self.logger)

    def get_params_dict(self) -> dict:
        '''
        Return a dictionary representation of the uniform prior.

        Returns
        -------
        dict: Dictionary of the parameter

        Authors: Allan Denis
        '''

        return {"mean": self.mean, "stddev": self.stddev}
