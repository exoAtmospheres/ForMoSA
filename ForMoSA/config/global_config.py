import os
import ast
import logging
import numpy as np
from pathlib import Path
from configobj import ConfigObj
from typing import Any, List, Union
from scipy.interpolate import interp1d
from dataclasses import dataclass, field, asdict

import ForMoSA.utils.misc as um
import ForMoSA.parameter.prior as Prior
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.grid.model_grid import ModelGrid
from ForMoSA.core.loggings import setup_logging
from ForMoSA.parameter.parameter import Parameter
from ForMoSA.observation.observation_set import ObservationSet
from ForMoSA.core.enums import PriorType, VsiniFunction, ParameterKind, ObservationType

# ----------------------------
# Dataclasses for each section
# ----------------------------

@dataclass
class ConfigPath:
    observation_path: list[str | os.PathLike] = field(default_factory=list[str])
    adapt_store_path: str | os.PathLike = field(default_factory=str)
    result_path: str | os.PathLike = field(default_factory=str)
    model_path: str | os.PathLike = field(default_factory=str)

    def __post_init__(self) -> None:
        '''
        Check paths types.

        Authors: Allan Denis
        '''

        for name in ['adapt_store_path', 'result_path', 'model_path']:
            value = getattr(self, name)
            if not isinstance(value, (str, os.PathLike)):
                raise ForMoSAError(f"{name} must be str or os.PathLike, got {type(value)}")

        if isinstance(self.observation_path, (str, os.PathLike)):
            setattr(self, 'observation_path', [self.observation_path])

        if not isinstance(self.observation_path, list):
            raise ForMoSAError(f"Wront type for observation_path: {type(self.observation_path)}. Expected a list")

        if not all(isinstance(obs_path, (str | os.PathLike)) for obs_path in self.observation_path):
            raise ForMoSAError("observation_path must be a list of str or os.PathLike")

@dataclass
class ConfigAdapt:
    method: str = "linear"
    emulator: list[str] = field(default_factory=lambda: ["NA"])
    target_res_obs: list[Union[str, float]] = field(default_factory=lambda: ["obs"])
    target_res_mod: list[Union[str, float]] = field(default_factory=lambda: ["obs"])
    res_cont: list[Union[str, float]] = field(default_factory=lambda: ["NA"])

    def __post_init__(self) -> None:
        '''
        Check adapt configuration parameters and normalize types.

        Authors: Allan Denis
        '''

        # Check method
        if not isinstance(self.method, str):
            raise ForMoSAError(f" method must be a string, got {type(self.method)}")

        # Normalize fields
        self.emulator = um.normalize_list(self.emulator, "emulator")

        self.target_res_obs = um.normalize_list(self.target_res_obs, "target_res_obs", um.to_float_if_possible)

        self.target_res_mod = um.normalize_list(self.target_res_mod, "target_res_mod", um.to_float_if_possible)

        self.res_cont = um.normalize_list(self.res_cont, "res_cont", um.to_float_if_possible)

    # =======================
    # Methods
    # =======================

    def _check_with_n_obs(self, n_obs: int) -> None:
        '''
        Check consistency between lengths of list parameters and n_obs

        Parameters
        ----------
        n_obs (int): Number to be tested against lengths of list parameters

        Authors: Allan Denis
        '''

        if (not isinstance(n_obs, int)) or (n_obs < 1):
            raise ForMoSAError(f' n_obs ({n_obs}) must be an integer greater than 0')

        for name in ['target_res_obs', 'target_res_mod', 'res_cont']:
            value = getattr(self, name)
            if len(value) > 1:
                if len(value) != n_obs:
                    raise ForMoSAError(f' Number of observations ({n_obs}) and length of {name} ({len(value)}) are inconsistent')

            elif len(value) == 1 and n_obs > 1:
                setattr(self, name, n_obs * value)

    def _compute_obs_target_resolution(self, observations: ObservationSet, grid: ModelGrid) -> list[np.ndarray]:
        '''
        Compute the target spectral resolution of the observation with respect to the model grid.

        Parameters
        ----------
        observations  (ObservationSet): Instance of class Observtation
        grid               (ModelGrid): Instance of class ModelGrid

        Returns
        -------
        list[np.ndarray]: Target resolution for each observation

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        # Initial check
        if not isinstance(observations, ObservationSet):
            raise ForMoSAError('Wrong type for observations: {type(observations)}. Expected an instance of ObservationSet')

        if not isinstance(grid, ModelGrid):
            raise ForMoSAError('Wrong type for grid: {type(grid)}. Expected an isntance of ModelGrid')

        # Check that len(config_adapt.target_res_obs) is consistent with the number of observations
        self._check_with_n_obs(observations.n_observations)

        # Compute target resolution
        output_target_res = []

        # Loop in observations
        for target_res, obs in zip(self.target_res_obs, observations.observations):
            if obs.ObsType == ObservationType.PHOTOMETRIC.obstype:
                output_target_res.append(0)
            elif obs.ObsType == ObservationType.SPECTROSCOPIC.obstype:
                # Interpolate model resolution onto observation wavelength grid
                interp_model_to_obs = interp1d(grid.wave, grid.res, fill_value="extrapolate")

                res_model_obs = interp_model_to_obs(obs.wave)

                if target_res == "obs":
                    target_res = np.minimum(obs.res, res_model_obs)
                else:
                    res_custom = np.full_like(obs.res, float(target_res))
                    target_res = np.minimum.reduce([obs.res, res_model_obs, res_custom])

                output_target_res.append(target_res)

        return output_target_res

    def _compute_model_target_wavelength_and_resolution(self, observations: ObservationSet, grid: ModelGrid) -> tuple[list[np.ndarray], list[np.ndarray]]:
        '''
        Determine the target wavelength and resolution for the grids.

        Parameters
        ----------
        observations  (ObservationSet): Instance of class Observtation
        grid               (ModelGrid): Instance of class ModelGrid

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]: Targets wavelength and resolution lists

        Authors: Allan Denis
        '''

        # Initial checks
        if not isinstance(observations, ObservationSet):
            raise ForMoSAError('Wrong type for observations: {type(observations)}. Expected an instance of ObservationSet')

        if not isinstance(grid, ModelGrid):
            raise ForMoSAError('Wrong type for grid: {type(grid)}. Expected an isntance of ModelGrid')

        # Check that len(config_adapt.target_res_mod) is consistent with the number of observations
        self._check_with_n_obs(observations.n_observations)

        # Compute targets wavelength and resolution
        output_target_wave, output_target_res = [], []

        # Loop in observations
        for target_res, obs in zip(self.target_res_mod, observations.observations):
            if target_res == "obs":
                target_wave = obs.wave.copy()
                target_res = obs.res.copy()
            else:
                target_wave = grid.wave.copy()

                if target_res == "mod":
                    target_res = grid.res.copy()
                else:
                    # target_res is a float
                    target_res = np.full(target_wave.shape, float(target_res), dtype=float)

            output_target_wave.append(target_wave)
            output_target_res.append(target_res)


        return output_target_wave, output_target_res

    def _determine_remove_continuum(self, observations: ObservationSet) -> list[bool]:
        '''
        Determine whether the continuum should be removed for each observation.

        Parameters
        ----------
        observations  (ObservationSet): Instance of class Observtation

        Returns
        -------
        list[bool]: Whether the continuum should be removed for each observation

        Authors: Allan Denis
        '''

        # Initial checks
        if not isinstance(observations, ObservationSet):
            raise ForMoSAError('Wrong type for observations: {type(observations)}. Expected an instance of ObservationSet')

        # Check that len(config_adapt.res_cont) is consistent with the number of observations
        self._check_with_n_obs(observations.n_observations)

        # Compute remove continuum
        remove_continuum = []
        # Loop in observations
        for res_cont, obs in zip(self.res_cont, observations.observations):
            # No continuum windows → no continuum removal
            if res_cont == "NA":
                remove_cont = False

            # High-contrast mode → never remove continuum from models
            elif len(obs.star_flux) > 0:
                remove_cont = True

            # Not high-contrast mode → remove continuum from models
            else:
                remove_cont = True

            remove_continuum.append(remove_cont)

        return remove_continuum

@dataclass
class ConfigInversion:
    logL_type: List[str] = field(default_factory=lambda: ["chi2"])
    wav_fit: List[str] = field(default_factory=lambda: ["0.9, 5.0"])
    ns_algo: str = field(default_factory=lambda: "pymultinest")
    npoints: int = field(default_factory=lambda: 50)
    hc_lower_bounds_lsq: list[float | str] = field(default_factory=lambda: ["NA"])
    hc_higher_bounds_lsq: list[float | str] = field(default_factory=lambda: ["NA"])

    def __post_init__(self) -> None:
        '''
        Check inversion configuration parameters and normalize types.

        Authors: Allan Denis
        '''

        # Check npoints
        if isinstance(self.npoints, str):
            self.npoints = int(self.npoints)

        if not isinstance(self.npoints, int) or self.npoints <= 0:
            raise ForMoSAError('npoints must be a strictly positive integer')

        # Check ns_algo
        if not isinstance(self.ns_algo, str):
            raise ForMoSAError(f'ns_algo must be a string, got {type(self.ns_algo)}')

        # Normalize fields
        self.logL_type = um.normalize_list(self.logL_type, "logL_type")
        self.wav_fit = um.normalize_list(self.wav_fit, "wav_fit")
        self.hc_lower_bounds_lsq = um.normalize_list(self.hc_lower_bounds_lsq, "hc_lower_bounds_lsq", um.to_float_if_possible)
        self.hc_higher_bounds_lsq = um.normalize_list(self.hc_higher_bounds_lsq, "hc_higher_bounds_lsq", um.to_float_if_possible)

        # Check lower and higher hc bounds
        if len(self.hc_higher_bounds_lsq) != len(self.hc_higher_bounds_lsq):
            raise ForMoSAError('hc_lower_bounds_lsq and hc_higher_bounds_lsq must have same lengths')

        self._hc_bounds = None

    # =======================
    # Properties
    # =======================

    @property
    def hc_bounds(self) -> list[tuple[float, float]]:
        bounds = []
        for lower, higher in zip(self.hc_lower_bounds_lsq, self.hc_higher_bounds_lsq):
            # Replace 'NA' for lower bound
            if isinstance(lower, str) and lower.upper() == "NA":
                lower = -float('inf')

            # Replace 'NA' for higher bound
            if isinstance(higher, str) and higher.upper() == "NA":
                higher = float('inf')

            bounds.append((lower, higher))

        return bounds

    # =======================
    # Methods
    # =======================

    def _check_with_n_obs(self, n_obs: int) -> None:
        '''
        Check consistency between lengths of list parameters and n_obs

        Parameters
        ----------
        n_obs (int): Number to be tested against lengths of list parameters

        Authors: Allan Denis
        '''

        if (not isinstance(n_obs, int)) or (n_obs < 1):
            raise ForMoSAError(f' n_obs ({n_obs}) must be an integer greater than 0')

        for name in ['logL_type', 'wav_fit', 'hc_lower_bounds_lsq', 'hc_higher_bounds_lsq']:
            value = getattr(self, name)
            if len(value) > 1:
                if len(value) != n_obs:
                    raise ForMoSAError(f' Number of observations ({n_obs}) and length of {name} ({len(value)}) are inconsistent')

            elif len(value) == 1 and n_obs > 1:
                setattr(self, name, n_obs * value)

@dataclass
class ConfigParameters:
    par1: list[str] = field(default_factory=lambda: ["NA"])
    par2: list[str] = field(default_factory=lambda: ["NA"])
    par3: list[str] = field(default_factory=lambda: ["NA"])
    par4: list[str] = field(default_factory=lambda: ["NA"])
    r: list[str] = field(default_factory=lambda: ["NA"])
    d: list[str] = field(default_factory=lambda: ["NA"])
    alpha: list[str] = field(default_factory=lambda: ["NA"])
    bb_T: list[str] = field(default_factory=lambda: ["NA"])
    rv: list[str] = field(default_factory=lambda: ["NA"])
    vsini: list[str] = field(default_factory=lambda: ["NA"])
    ld: list[str] = field(default_factory=lambda: ["NA"])

    def __post_init__(self) -> None:
        '''
        Check parameters configuration.

        Authors: Allan Denis
        '''

        for name, value in self.__dict__.items():
            if not isinstance(value, list):
                setattr(self, name, ast.literal_eval(value))

    # =======================
    # Representation
    # =======================

    def __repr__(self) -> str:
        params = ", ".join(f"{key}={value!r}" for key, value in self.to_dict.items() if key.startswith('par')) + ", " + ", ".join(f"{key}={value!r}" for key, value in self.__dict__.items() if not key.startswith('par'))
        return f"ConfigParameters({params})"

    # =======================
    # Properties
    # =======================

    @property
    def to_dict(self) -> dict:                              # Dictionary representation of ConfigParameters
        return self.__dict__

    # =======================
    # Methods
    # =======================

    def _add_parameter(self, param: str, value: list[str]) -> None:
        '''
        Add a parameter to the instance.

        Parameters
        ----------
        param (str): Name of the parameter
        value (str): Value of the parameter

        Authors: Allan Denis
        '''

        setattr(self, str(param), value)

    def _parse_param_name(self, name: str) -> tuple[str, str, list | None]:
        '''
        Parse parameter name from the current instance.

        Parameters
        ----------
        name (str): Name of the parameter in the config file

        Returns
        -------
        tuple[str, str, list | None]: (name, scope, obs_index) of the parameter

        Authors: Allan Denis
        '''

        if name not in self.to_dict.keys():
            raise ForMoSAError(f' Please chose a name amongst {self.to_dict.keys()}')

        if "_" not in name or name in ('bb_T', 'bb_R'):
            if name.startswith('par'):
                kind = ParameterKind.GRID
            else:
                kind = ParameterKind[name.upper()]
            return name, kind, "global", None

        base, obs = name.split("_", 1)
        obs_index = [int(i) for i in obs.split("_")]
        kind = ParameterKind[base.upper()]
        return name, kind, "local", obs_index

    def _parse_param_value(self, value: list[str] | str) -> Prior.Prior | str:
        '''
        Parse parameter value from the current instance.

        Parameters
        ----------
        value (list[str]): Value of the parameter in the config file

        Returns
        -------
        Prior.Prior | str: Instance of Prior.Prior or 'NA'

        Authors: Allan Denis
        '''

        if value not in self.to_dict.values():
            raise ForMoSAError(f' Please chose a value amongst {self.to_dict.value()}')

        if value[0] == 'NA' or value == 'NA':
            return 'NA'
        else:
            prior_type = PriorType[value[0].strip().upper()]

            if prior_type == PriorType.CONSTANT:
                prior = Prior.ConstantPrior(value[1].strip())
            elif prior_type == PriorType.UNIFORM:
                prior = Prior.UniformPrior(value[1].strip(), value[2].strip())
            elif prior_type == PriorType.LOG_UNIFORM:
                prior = Prior.UniformPrior(value[1].strip(), value[2].strip())
            else:
                prior = Prior.GaussianPrior(value[1].strip(), value[2].strip())

        return prior

    def _parse_param(self, name: str, value: str, **kwargs) -> Parameter | None:
        '''
        Parse parameter name and value from the current instance.

        Parameters
        ----------
        name  (str): Name of the parameter in the config file
        value (str): Value of the parameter in the config file
        **kwargs   : Additional arguments (logger, log_level, ...)

        Returns
        -------
        Parameter | None: Instance of class Parameter or None if parameter has no prior

        Authors: Allan Denis
        '''

        name, kind, scope, obs_index = self._parse_param_name(name)
        prior = self._parse_param_value(value)

        if prior != 'NA':
            vsini_function = None

            if kind == ParameterKind.VSINI:
                vsini_functions = [str(func.function) for func in VsiniFunction]
                vsini_function = value[-1].strip()

                if not vsini_function in vsini_functions:
                    raise ForMoSAError(f'Wrong value for vsini_function: {vsini_function}. You must provide a valid vsini_function amongst {vsini_functions}')

                vsini_function = VsiniFunction[vsini_function]

            return Parameter(name, prior, kind, scope=scope, obs_index=obs_index, vsini_function=vsini_function, **kwargs)

        # Prior is NA
        return None

@dataclass
class ConfigNestle:
    method: str = field(default_factory=lambda: "single")
    update_interval: float = field(default_factory=lambda: None)
    npdim: int = field(default_factory=lambda: None)
    maxiter: int = field(default_factory=lambda: None)
    maxcall: int = field(default_factory=lambda: None)
    dlogz: float = field(default_factory=lambda: None)
    decline_factor: float = field(default_factory=lambda: None)
    rstate: Any = field(default_factory=lambda: None)  # seed or RNG, can be int, list, or numpy.random.Generator

    def __post_init__(self) -> None:
        '''
        Check Nestle configuration parameters and normalize types.

        Authors: Allan Denis
        '''

        # Check method
        if not isinstance(self.method, str):
            raise ForMoSAError(f" method must be a string, got {type(self.method)}")

        # Convert string numbers to float/int where appropriate
        int_fields = ["update_interval", "npdim", "maxiter", "maxcall"]
        float_fields = ["dlogz", "decline_factor"]

        for name in int_fields:
            value = getattr(self, name)
            if not isinstance(value, int):
                if isinstance(value, str):
                    try:
                        setattr(self, name, int(value))
                    except ValueError:
                        if value == 'None':
                            setattr(self, name, None)

                        else:
                            raise ForMoSAError(f" {name} must be int or str convertible to int, got '{value}'")

        for name in float_fields:
            value = getattr(self, name)
            if not isinstance(value, float):
                if isinstance(value, str):
                    try:
                        setattr(self, name, float(value))
                    except ValueError:
                        if value == 'None':
                            setattr(self, name, None)

                        else:
                            raise ForMoSAError(f" {name} must be float or str convertible to float, got '{value}'")

        # rstate can be None, int, or RNG, no need to convert
        if self.rstate != 'None' and not isinstance(self.rstate, (int, list)):
            pass  # allow user to pass np.random.Generator or custom RNG
        if self.rstate == 'None':
            setattr(self, 'rstate', None)

    # =======================
    # Properties
    # =======================

    @property
    def to_dict(self) -> dict:                             # Dictionary representation of ConfigNestle
        return self.__dict__

    # =======================
    # Methods
    # =======================

    @classmethod
    def from_dict(cls, data: dict) -> 'ConfigNestle':
        '''
        Build an instance of ConfigNestle from dictionary of Nestle parameters.

        Parameters
        ----------
        data (dict): Dictionary of Nestle parameters

        Returns
        -------
        'ConfigNestle': An instance of class ConfigNestle

        Authors: Allan Denis
        '''

        if not isinstance(data, dict):
            raise ForMoSAError(f'Wrong type for data: {type(data)}. Expected a dictionary')

        return cls(**data['nestle'])

@dataclass
class ConfigPyMultiNest:
    importance_nested_sampling: bool = field(default_factory=lambda: True)
    multimodal: bool = field(default_factory=lambda: True)
    const_efficiency_mode: bool = field(default_factory=lambda: False)
    evidence_tolerance: float = field(default_factory=lambda: 0.5)
    sampling_efficiency: float = field(default_factory=lambda: 0.8)
    n_iter_before_update: int = field(default_factory=lambda: 100)
    null_log_evidence: float = field(default_factory=lambda: -1e90)
    max_modes: int = field(default_factory=lambda: 100)
    mode_tolerance: float = field(default_factory=lambda: -1e90)
    seed: int = field(default_factory=lambda: -1)
    verbose: bool = field(default_factory=lambda: True)
    resume: bool = field(default_factory=lambda: False)
    context: int = field(default_factory=lambda: 0)
    log_zero: float = field(default_factory=lambda: -1e100)
    max_iter: int = field(default_factory=lambda: 0)
    init_MPI: bool = field(default_factory=lambda: False)
    wrapped_params: int = field(default_factory=lambda: None)
    dump_callback: int = field(default_factory=lambda: None)
    use_MPI: bool = field(default_factory=lambda: True)

    def __post_init__(self) -> None:
        '''
        Check PyMultiNest configuration parameters and normalize types.

        Authors: Allan Denis
        '''

        # Bool fields
        bool_fields = (
            "importance_nested_sampling",
            "multimodal",
            "const_efficiency_mode",
            "verbose",
            "resume",
            "init_MPI",
            "use_MPI",
        )

        for name in bool_fields:
            value = getattr(self, name)
            if isinstance(value, str):
                val_lower = value.lower()
                if val_lower == "true":
                    setattr(self, name, True)

                elif val_lower == "false":
                    setattr(self, name, False)

                else:
                    raise ForMoSAError(f" {name} must be a boolean or 'true'/'false' string, got '{value}'")

            elif not isinstance(value, bool):
                raise ForMoSAError(f" {name} must be a boolean, got {type(value)}")

        # Float fields
        float_fields = ("evidence_tolerance", "sampling_efficiency", "null_log_evidence",
                        "mode_tolerance", "log_zero")

        for name in float_fields:
            value = getattr(self, name)
            if isinstance(value, str):
                try:
                    setattr(self, name, float(value))
                except ValueError:
                    if value == 'None':
                        setattr(self, name, None)
                    else:
                        raise ForMoSAError(f" {name} must be float or string convertible to float, got '{value}'")

            elif not isinstance(value, (float, int)) and value is not None:
                raise ForMoSAError(f" {name} must be float, got {type(value)}")

        # Integer fields
        int_fields = ("n_iter_before_update", "max_modes", "seed", "context", "max_iter", "wrapped_params")
        for name in int_fields:
            value = getattr(self, name)
            if isinstance(value, str):
                try:
                    setattr(self, name, int(value))
                except ValueError:
                    if value == 'None':
                        setattr(self, name, None)

                    else:
                        raise ForMoSAError(f" {name} must be int or string convertible to int, got '{value}'")

            elif (not (isinstance(value, int)) and (value is not None)):
                raise ForMoSAError(f" {name} must be int, got {type(value)}")

    # =======================
    # Properties
    # =======================

    @property
    def to_dict(self) -> dict:                            # Dictionary representation of ConfigPyMultiNest
        return self.__dict__

    # =======================
    # Methods
    # =======================

    @classmethod
    def from_dict(cls, data: dict) -> 'ConfigPyMultiNest':
        '''
        Build an instance of ConfigPyMultiNest from dictionary of PyMultiNest parameters.

        Parameters
        ----------
        data (dict): Dictionary of PyMultiNest parameters

        Returns
        -------
        'ConfigPyMultiNest': An instance of class ConfigPyMultiNest

        Authors: Allan Denis
        '''

        if not isinstance(data, dict):
            raise ForMoSAError(f'Wrong type for data: {type(data)}. Expected a dictionary')

        return cls(**data['pymultinest'])

@dataclass
class ConfigUltraNest:

    # Arguments for ReactiveNestedSampler
    wrapped_params: bool = field(default_factory=lambda: None)
    vectorized: bool = field(default_factory=lambda: False)
    resume: bool = field(default_factory=lambda: True)
    run_num: int = field(default_factory=lambda: None)
    num_bootstraps: int = field(default_factory=lambda: 30)
    storage_backend: str = field(default_factory=lambda: "hdf5")
    warmstart_max_tau: int = field(default_factory=lambda: -1)

    # Arguments for run
    dlogz: float = field(default_factory=lambda: 0.5)
    max_iters: int = field(default_factory=lambda: None)
    max_ncalls: int = field(default_factory=lambda: None)
    min_ess: int = field(default_factory=lambda: 400)
    frac_remain: float = field(default_factory=lambda: 0.01)
    cluster_num_live_points: int = field(default_factory=lambda: 40)
    Lepsilon: float = field(default_factory=lambda: 0.001)

    def __post_init__(self) -> None:
        '''
        Check UltraNest configuration parameters and normalize types.

        Authors: Allan Denis
        '''

        # Float fields
        float_fields = (
            "dlogz",
            "frac_remain",
            "Lepsilon",
        )
        for name in float_fields:
            value = getattr(self, name)
            if isinstance(value, str):
                try:
                    setattr(self, name, float(value))
                except ValueError:
                    if value == 'None':
                        setattr(self, name, None)
                    else:
                        raise ForMoSAError(f" {name} must be float or string convertible to float, got '{value}'")
            elif not isinstance(value, (int, float)) and value is not None:
                raise ForMoSAError(f" {name} must be float, got {type(value)}")

        # Integer fields
        int_fields = (
            "num_bootstraps",
            "warmstart_max_tau",
            "max_iters",
            "max_ncalls",
            "min_ess",
            "cluster_num_live_points",
        )
        for name in int_fields:
            value = getattr(self, name)
            if isinstance(value, str):
                if value == 'None':
                    setattr(self, name, None)

                else:
                    try:
                        setattr(self, name, int(value))
                        continue
                    except ValueError:
                        raise ForMoSAError(f"{name} must be int or string convertible to int, got '{value}'")

            elif not isinstance(value, int) and value is not None:
                raise ForMoSAError(f" {name} must be int, got {type(value)}")

        # Boolean fields
        bool_fields = ("wrapped_params", "vectorized")
        for name in bool_fields:
            value = getattr(self, name)
            if isinstance(value, str):
                if value == 'None':
                    setattr(self, name, None)

                else:
                    val_lower = value.lower()
                    if val_lower == "true":
                        setattr(self, name, True)

                    elif val_lower == "false":
                        setattr(self, name, False)

                    else:
                        raise ForMoSAError(f" {name} must be boolean or 'true'/'false' string, got '{value}'")

            elif not isinstance(value, bool) and value is not None:
                raise ForMoSAError(f" {name} must be boolean, got {type(value)}")

        if self.wrapped_params == 'None':
            self.wrapped_params = None

        float_fields = ("dlogz", "Lepsilon", "frac_remain")
        for name in float_fields:
            value = getattr(self, name)
            if isinstance(value, str):
                if value == 'None':
                    setattr(self, name, None)

                else:
                    try:
                        setattr(self, name, float(value))
                        continue
                    except ValueError:
                        raise ForMoSAError(f"{name} must be a float or string convertible to float, got '{value}'")

            elif not isinstance(value, float) and value is not None:
                raise ForMoSAError(f"{name} must be float, got {type(value)}")

    # =======================
    # Properties
    # =======================

    @property
    def ReactiveNSParams(self) -> dict:
        return {
            'wrapped_params': self.wrapped_params,
            'vectorized': self.vectorized,
            'resume': self.resume,
            'run_num': self.run_num,
            'num_bootstraps': self.num_bootstraps,
            'storage_backend': self.storage_backend,
            'warmstart_max_tau': self.warmstart_max_tau
            }

    @property
    def runNSParams(self) -> dict:
        return {
            'dlogz': self.dlogz,
            'max_iters': self.max_iters,
            'max_ncalls': self.max_ncalls,
            'min_ess': self.min_ess,
            'frac_remain': self.frac_remain,
            'cluster_num_live_points': self.cluster_num_live_points,
            'Lepsilon': self.Lepsilon
            }

    @property
    def to_dict(self) -> dict:            # Dictionary representation of ConfigUltranest
        return {'ReactiveNS': self.ReactiveNSParams, 'runNS': self.runNSParams}

    # =======================
    # Methods
    # =======================

    @classmethod
    def from_dict(cls, data: dict) -> 'ConfigUltraNest':
        '''
        Build an instance of ConfigUltraNest from dictionary of UltraNest parameters.

        Parameters
        ----------
        data (dict): Dictionary of UltraNest parameters

        Returns
        -------
        'ConfigUltraNest': An instance of class ConfigUltraNest

        Authors: Allan Denis
        '''

        if not isinstance(data, dict):
            raise ForMoSAError(f'Wrong type for data: {type(data)}. Expected a dictionary')

        return cls(**data['ultranest']['ReactiveNS'], **data['ultranest']['runNS'])

@dataclass
class Config_NS:
    nestle: ConfigNestle = field(default_factory=lambda: ConfigNestle())
    pymultinest: ConfigPyMultiNest = field(default_factory=lambda:ConfigPyMultiNest())
    ultranest: ConfigUltraNest = field(default_factory=lambda:ConfigUltraNest())

    def __post_init__(self) -> None:
        '''
        Check Config_NS configuration parameters.

        Authors: Allan Denis
        '''

        # Check nestle type
        if not isinstance(self.nestle, ConfigNestle):
            raise ForMoSAError(f" nestle must be a ConfigNestle, got {type(self.nestle)}")

        # Check pymultinest type
        if not isinstance(self.pymultinest, ConfigPyMultiNest):
            raise ForMoSAError(f" pymultinest must be a ConfigPyMultinest, got {type(self.pymultinest)}")

        # Check ultranest type
        if not isinstance(self.ultranest, ConfigUltraNest):
            raise ForMoSAError(f" nestle must be a ConfigUltranest, got {type(self.ultranest)}")

    # =======================
    # Properties
    # =======================

    @property
    def to_dict(self) -> dict:
        return {
            'nestle': self.nestle.to_dict,
            'pymultinest': self.pymultinest.to_dict,
            'ultranest': self.ultranest.to_dict
            }

    # =======================
    # Methods
    # =======================

    @classmethod
    def from_dict(cls, data: dict) -> "Config_NS":
        '''
        Build an instance of Config_NS from a dictionary of data.

        Parameters
        ----------
        data (dict): Dictionary representation of data

        Returns
        -------
        "Config_NS": An instance of class Config_NS

        Authors: Allan Denis
        '''

        if not isinstance(data, dict):
            raise ForMoSAError(f'Wrong type for data: {type(data)}. Expected a dictionary')

        return cls(
            nestle = ConfigNestle.from_dict(data),
            pymultinest = ConfigPyMultiNest.from_dict(data),
            ultranest=ConfigUltraNest.from_dict(data)
            )


# ----------------------------
# Config file generator
# ----------------------------

class ConfigGenerator:
    '''
    Config file generator.

    Parameters
    ----------
    sections          (dict): Dictionary containing sections of the config file
    logger  (logging.Logger): Logger
    log_level          (str): Level of the Logger

    Authors: Mathieu Ravet and Allan Denis
    '''

    def __init__(self, sections: dict = None, logger: logging.Logger | None = None, log_level: str = 'INFO') -> None:
        self.logger = logger or setup_logging(level=log_level, name='ConfigGenerator')
        if sections is not None:
            # If we provide the sections
            self.config = sections
        else:
            # Otherwise, we generate default sections
            self.config = {
                "config_path": ConfigPath('unknown', 'unknown', 'unknown', 'unknown'),
                "config_adapt": ConfigAdapt(),
                "config_inversion": ConfigInversion(),
                "config_parameters": ConfigParameters(),
                "config_nestle": ConfigNestle(),
                "config_pymultinest": ConfigPyMultiNest(),
                "config_ultranest": ConfigUltraNest(),
            }
        self.comments = self._init_comments()


    def __repr__(self):
        return "ConfigGenerator()"

    def _init_comments(self):
        c = {}

        # ---------------- config_path ----------------
        c["config_path"] = {
            "observation_path": ["    # Path to the observed spectrum file"],
            "adapt_store_path": ["    # Path to store your interpolated grid"],
            "result_path": ["    # Path to store your results"],
            "model_path": ["    # Path to the model"]
        }

        # ---------------- config_adapt ----------------
        c["config_adapt"] = {
            "method": [
                "    # Adaptation method. /!\\ For safety reasons, this will also be the interpolation method",
                "    # Format : 'linear' or 'nearest' or 'zero' or 'slinear' or 'quadratic' or 'cubic' or 'quintic' or 'pchip' or 'barycentric' or 'krogh' or 'akima' or 'makima'",
                "    # MOSAIC : No"
            ],
            "emulator": [
                "    # If you want to use an emulator to fit your grid (smooth out the grid).",
                "    # Format : 'NA' or 'PCA, ncomp' or 'NMF, ncomp'",
                "    # MOSAIC : No"
            ],
            "target_res_obs": [
                "    # Target resolution to reach for the observation(s).",
                "    # Format : float or 'obs' (if you want to keep the original obs resolution)",
                "    # MOSAIC : Yes"
            ],
            "target_res_mod": [
                "    # Target resolution to reach for the model.",
                "    # Format : float or 'obs' or 'mod' (if you want to keep the model's resolution during inversion)",
                "    # MOSAIC : Yes"
            ],
            "res_cont": [
                "    # Resolution used to estimate the continuum.",
                "    # Format : 'NA' or float",
                "    # MOSAIC : Yes"
            ]
        }

        # ---------------- config_inversion ----------------
        c["config_inversion"] = {
            "logL_type": [
                "    # Method to calculate the loglikelihood function used in the nested sampling procedure.",
                "    # Format : 'chi2' or 'chi2_covariance' or 'chi2_noisescaling' or 'chi2_noisescaling_covariance' or 'CCF_Brogi' or 'CCF_Zucker' or 'CCF_custom'",
                "    # MOSAIC : Yes"
            ],
            "wav_fit": [
                "    # Wavelength range(s) used during the nested sampling procedure.",
                "    # Format : 'window1_min / window1_max, window2_min / ... / windowN_max'",
                "    # MOSAIC : Yes"
            ],
            "ns_algo": [
                "    # Nested sampling algorithm used.",
                "    # Format : 'nestle' or 'pymultinest' or 'ultranest'",
                "    # MOSAIC : No"
            ],
            "npoints": [
                "    # Number of living points during the nested sampling procedure.",
                "    # Format : int",
                "    # MOSAIC : No"
            ],
            "hc_lower_bounds_lsq": [
                "    # Least-square bounds.",
                "    # Format : 'NA' or 'lower, upper'",
                "    # MOSAIC : Yes"
            ],
            "hc_higher_bounds_lsq": [
                "    # Least-square bounds.",
                "    # Format : 'NA' or 'lower, upper'",
                "    # MOSAIC : Yes"
            ]
        }

        # ---------------- config_parameters ----------------
        c["config_parameters"] = {
            "par1": [
                "    # Definition of the prior function of each parameter explored by the grid. Please refer to the documentation to check",
                "    # the parameter space explore by each grid. Check prior functions for more infos",
                "    # Format : 'function', function_param1, function_param2",
                "    # MOSAIC : No"
            ],
            "par2": [],
            "par3": [],
            "par4": [],
            "r": [
                "    # Definition of the prior function of each extra-grid parameter. Check prior functions for more infos",
                "    # Format : 'function', function_param1, function_param2",
                "    # MOSAIC : Yes and No, check the doc !"
            ],
            "d": []
        }

        # ---------------- config_nestle ----------------
        c["config_nestle"] = {
            "method": [
                "    # Nestle configuration parameters. For more details, please see: http://kylebarbary.com/nestle/index.html",
                "    # Format : _",
                "    # MOSAIC : No"
            ],
            "update_interval": [],
            "npdim": [],
            "maxiter": [],
            "maxcall": [],
            "dlogz": [],
            "decline_factor": [],
            "rstate": []
        }

        # ---------------- config_pymultinest ----------------
        c["config_pymultinest"] = {
            "n_clustering_params": [
                "    # Pymultinest configuration parameters. For more details, please see: https://github.com/JohannesBuchner/PyMultiNest/blob/master/pymultinest/run.py",
                "    # Format : _",
                "    # MOSAIC : No"
            ],
            "wrapped_params": [],
            "importance_nested_sampling": [],
            "multimodal": [],
            "const_efficiency_mode": [],
            "evidence_tolerance": [],
            "sampling_efficiency": [],
            "n_iter_before_update": [],
            "null_log_evidence": [],
            "max_modes": [],
            "mode_tolerance": [],
            "seed": [],
            "verbose": [],
            "resume": [],
            "context": [],
            "log_zero": [],
            "max_iter": [],
            "init_MPI": [],
            "dump_callback": [],
            "use_MPI": []
        }

        # ---------------- config_ultranest ----------------
        c["config_ultranest"] = {
            "resume": [
                "    # Ultranest configuration parameters. For more details, please see: https://johannesbuchner.github.io/UltraNest/readme.html",
                "    # Format : _",
                "    # MOSAIC : No"
            ]
        }
        # Other keys are added without comments
        for key in asdict(ConfigUltraNest()).keys():
            if key not in c["config_ultranest"]:
                c["config_ultranest"][key] = []

        return c

    def save(self, path: str | os.PathLike, name: str = 'new_config.ini'):
        '''
        Save the ConfigGenerator to a given path.

        Parameters
        ----------
        path (str | os.PathLike): Path where to save the config file
        name               (str): Name of the config file

        Authors: Allan Denis
        '''

        self.logger.info(f'    Save config to path {path}')

        path = Path(path) / name
        config = ConfigObj(indent_type='    ', list_values=True)
        for sec_name, sec_obj in self.config.items():
            config[sec_name] = {}
            sec_dict = asdict(sec_obj)
            for key, val in sec_dict.items():
                config[sec_name][key] = val
                if sec_name in self.comments and key in self.comments[sec_name]:
                    config[sec_name].comments[key] = self.comments[sec_name][key]
            config.comments[sec_name] = [""] + config.comments.get(sec_name, [])
        config.filename = path
        config.write()

class ConfigLoader:
    '''
    Class handling the loading of a config file.

    Parameters
    ----------
    path (str | os.PathLike): Path to the config file
    logger  (logging.Logger): Logger
    log_level          (str): Level of the Logger

    Auhors: Allan Denis
    '''

    def __init__(self, path: str | os.PathLike, logger: logging.Logger | None = None, log_level: str = 'INFO') -> None:
        self.path = path
        self.config_ini = ConfigObj(self.path, list_values=True, encoding='utf-8', file_error=False)
        self.logger = logger or setup_logging(level=log_level, name='ConfigLoader')

        self.defaults = ConfigGenerator().config
        self.config = {}

    def load(self):
        '''
        Load all the sections of .ini file in dataclasses.

        Authors: Allan Denis
        '''

        self.logger.debug(f' Load config file {self.path}')
        # Default config file
        self._fill_defaults()
        # mapping section name -> dataclass
        mapping = {
            "config_path": ConfigPath,
            "config_adapt": ConfigAdapt,
            "config_inversion": ConfigInversion,
            "config_parameters": ConfigParameters,
            "config_nestle": ConfigNestle,
            "config_pymultinest": ConfigPyMultiNest,
            "config_ultranest": ConfigUltraNest,
        }

        for section, cls in mapping.items():
            if section in self.config_ini:
                data = {}
                for key, val in self.config_ini[section].items():
                    if key not in self.defaults[section].__dict__.keys():
                        continue
                    data[key] = val
                if section == "config_parameters":
                    params = ConfigParameters()
                    for key, val in self.config_ini[section].items():
                        params._add_parameter(key, val)
                    self.config[section] = params
                    continue
                self.config[section] = cls(**data)
            else:
                self.config[section] = cls()  # Default values if not present

        self.logger.info('    Config file loaded')
        return self.config

    def _fill_defaults(self):
        '''
        Add missing sections / keys using defaults without overwriting existing values.

        Authors: Allan Denis
        '''

        for section, default_obj in self.defaults.items():
            if section not in self.config_ini:
                self.config_ini[section] = {}

            default_dict = asdict(default_obj)

            for key, value in default_dict.items():
                if key not in self.config_ini[section]:
                    self.config_ini[section][key] = value