import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from ForMoSA.core.enums import ParameterKind
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.loggings import setup_logging
from ForMoSA.parameter.parameter import Parameter
from ForMoSA.config.global_config import ConfigParameters

class ParameterSet(object):
    """
    Container for nested sampling parameters.

    Parameters
    ----------
    logger : logging.Logger
        Logger
    log_level : str
        Level of the logging

    Notes
    -----
    Authors: Allan Denis
    """

    def __init__(self, logger: logging.Logger | None = None, log_level: str = 'INFO') -> None:
        self._parameters: list[Parameter] = []

        self._logger = logger or setup_logging(level=log_level, name='ParameterSet')

    # ===========================
    # Representation
    # ===========================

    def __repr__(self):
        return f"<ParameterSet: {len(self.parameters)} parameters>"

    # ===========================
    # Properties
    # ===========================

    @property
    def logger(self) -> logging.Logger:
        """Logger."""
        return self._logger

    @property
    def parameters(self) -> list[Parameter]:
        """List of parameters."""
        return self._parameters

    @property
    def free_parameters(self) -> list[Parameter]:
        """list of free parameters."""
        return [p for p in self.parameters if not p.is_fixed]

    @property
    def fixed_parameters(self) -> list[Parameter]:
        """list of fixed parameters."""
        return [p for p in self.parameters if p.is_fixed]

    @property
    def names(self) -> list[str]:
        """list of parameter names."""
        return [p.name for p in self.parameters]

    @property
    def free_names(self) -> list[str]:
        """list of parameter names."""
        return [p.name for p in self.free_parameters]

    @property
    def titles(self) -> list[str]:
        """List of titles."""
        return [p.title for p in self.parameters]

    @property
    def free_titles(self) -> list[str]:
        """List of titles."""
        return [p.title for p in self.free_parameters]

    @property
    def kinds(self) -> list[ParameterKind]:
        """list of parameter kinds."""
        return [p.kind for p in self.parameters]

    @property
    def n_free_parameters(self) -> int:
        """Number of free parameters."""
        return len(self.free_parameters)

    @property
    def to_dict(self) -> dict:
        """Dictionary representation of the set of parameters."""
        data = {}
        for i, name in enumerate(self.names):
            data[name] = self.parameters[i].to_dict
        return data

    # ==========================
    # Class methods
    # ==========================

    @classmethod
    def from_config(cls, config_params: ConfigParameters, logger: logging.Logger | None = None, log_level: str = 'INFO') -> "ParameterSet":
        '''
        Generate instance of ParameterSet from an instance of ConfigParameters.

        Parameters
        ----------
        config_params : ConfigParameters
            Instance of ConfigParameters
        logger : logging.Logger
            Logger
        log_level : str
            Level of the logging

        Returns
        -------
        "ParameterSet"
            As instance of ParameterSet

        Notes
        -----
        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name='ParameterSet')

        if not isinstance(config_params, ConfigParameters):
            raise ForMoSAError(f'Wrong type for config_params: {type(config_params)}. Expected a ConfigParameters', logger)

        logger.debug('Generating set of parameters from configuration file')

        params = cls(logger=logger)

        for name, value in config_params.to_dict.items():
            try:
                param = config_params._parse_param(name, value, logger=params.logger)
            except ForMoSAError as e:
                raise ForMoSAError(e, logger)

            if param is not None:
                logger.info(f'      Detected {param.kind.kind} Parameter with name {param.name} from config')
                params.add_parameter(param)

        logger.info('    Set of parameters generated')

        return params

    @classmethod
    def from_dict(cls, data: dict, logger: logging.Logger | None = None, log_level: str = 'INFO') -> 'ParameterSet':
        '''
        Reconstruct a ParameterSet from a dictionary of ParameterSet.

        Parameters
        ----------
        data : dict
            Dictionary containing ParameterSet parameters
        logger : logging.Logger
            Logger
        log_level : str
            Level of the logging

        Returns
        -------
        'ParameterSet'
            An instance of class ParameterSet

        Notes
        -----
        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name='ParameterSet')
        param_set = cls(logger=logger)

        logger.debug('Building instance of ParameterSet from dictionary')

        for name in data.keys():
            param = Parameter.from_dict(data=data[name], logger=logger)
            param_set.add_parameter(param)

        return param_set

    @classmethod
    def from_json(cls, path: str | os.PathLike, logger: logging.Logger | None = None, log_level: str = 'INFO') -> 'ParameterSet':
        '''
        Reconstruct a ParameterSet from a json file.

        Parameters
        ----------
        path : str | os.PathLike
            Path to the json file
        logger : logging.Logger
            Logger
        log_level : str
            Level of the logging

        Returns
        -------
        'ParameterSet'
            An instance of class ParameterSet

        Notes
        -----
        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name='ParameterSet')

        if not isinstance(path, (str, os.PathLike)):
            raise ForMoSAError(f'Wrong type for path: {type(path)}. Expected a string or os.PathLike', logger)

        logger.debug(f'Build instance of ParameterSet from json file {str(path) + "parameters.json"}')

        filepath = Path(str(path) + 'parameters.json')
        if not filepath.exists():
            raise ForMoSAError(f'{filepath} does not exist')

        with open(filepath, "r") as f:
            data = json.load(f)

        return cls.from_dict(data, logger=logger, log_level=log_level)

    # ==========================
    # Methods
    # ==========================

    def prior_transform(self, theta: np.ndarray[float]) -> list[float]:
        '''
        Transform a list/array theta in [0,1]^N into physical values using each free parameter prior.

        Parameters
        ----------
        theta : list[float]
            list/array of floats in [0,1]^N where N is the number of free parameters

        Returns
        -------
        list[float] : physical values in the same ordering as self.free_parameters

        Notes
        -----
        Authors: Allan Denis
        '''

        # Initial checks
        if not isinstance(theta, np.ndarray):
            raise ForMoSAError(f" Wrong type for theta: expected array of floats, got {type(theta)}>", self.logger)

        if len(theta) != self.n_free_parameters:
            raise ForMoSAError(f"Theta length ({len(theta)}) does not match number of free parameters ({self.n_free_parameters})")

        # Transform
        values = []
        for val, p in zip(theta, self.free_parameters):
            # Each prior should implement sample(u) -> physical value
            values.append(p.prior.sample(val))
        return values

    def add_parameter(self, parameter: Parameter) -> None:
        '''
        Add a Parameter to the set.

        Parameters
        ----------
        parameter : Parameter
            Instance of class Parameter to add

        Notes
        -----
        Authors: Allan Denis
        '''

        if not isinstance(parameter, Parameter):
            raise ForMoSAError("Parameter must be a Parameter instance")

        if any(p.name == parameter.name for p in self.parameters):
            raise ForMoSAError(f"Parameter '{parameter.name}' already exists")

        self.logger.info(f'        Adding {parameter.kind.kind} Parameter with name {parameter.name} to the set of parameters')

        self.parameters.append(parameter)

    def summary(self, as_dataframe: bool = True) -> pd.DataFrame | str:
        '''
        Return a summary of parameters.

        Parameters
        ----------
        as_dataframe : bool
            If True, return pandas.DataFrame, else formatted string

        Returns
        -------
        pandas.DataFrame | str
            Summary of parameters

        Notes
        -----
        Authors: Allan Denis
        '''

        rows = []
        for p in self.parameters:
            rows.append({
                    "name": p.name,
                    "kind": p.kind.name,
                    "prior": p.prior.prior_type.value,
                    "value": (f"{p.prior.value}" if p.is_fixed else f"{p.prior.bounds}"),
                    "fixed": p.is_fixed,
                    "scope": p.scope,
                    "obs_index": str(p.obs_index)
                    })

        if as_dataframe:
            return pd.DataFrame(rows)

        # Text summary
        header = f"{'Name':<15} {'Kind':<15} {'Prior':<15} {'Value':<20} {'Fixed':<15} {'Scope':<15} {'Obs_index':<15}"
        sep = "-" * len(header)
        lines = [header, sep]

        for r in rows:
            lines.append(f"{r['name']:<15} {r['kind']:<15} {r['prior']:<15} {r['value']:<20} {r['fixed']:<15} {r['scope']:<15} {r['obs_index']:<15}")

        return ('').join(lines)

    def to_json(self, path: str | os.PathLike) -> None:
        '''
        Save the set of parameters to a given path as a json file.

        Parameters
        ----------
        path : str | os.PathLike
            Path to save the set of parameters

        Notes
        -----
        Authors: Allan Denis
        '''

        if not isinstance(path, (str, os.PathLike)):
            raise ForMoSAError(f'Wrong type for path: {type(path)}. Expected a string or os.PathLike', self.logger)

        self.logger.info(f'Save set of parameters to {path}')

        if not os.path.exists(path):
            self.logger.warning(f'{path} does not exist. Creating it.')
            Path(path).mkdir(exist_ok=True, parents=True)

        with open(str(path) + 'parameters.json', 'w') as f:
            json.dump(self.to_dict, f, indent=4)