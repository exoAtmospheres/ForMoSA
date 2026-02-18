import os
import json
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.axes._axes import Axes

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.enums import ObservationType
from ForMoSA.core.loggings import setup_logging
from ForMoSA.observation.observation_base import Observation
from ForMoSA.core.config import SpectralPlotConfig, PhotometricPlotConfig
from ForMoSA.observation.observation_spectroscopy import SpectralObservation
from ForMoSA.observation.observation_photometry import PhotometryObservation

class ObservationSet(object):
    '''
    Container for a set of Observation objects.

    Parameters
    ----------
    logger                            (logging.Logger): Logger
    log_level                                    (str): Level of the logging

    Authors: Allan Denis
    '''

    def __init__(self, logger: logging.Logger | None = None, log_level: str = "INFO") -> None:

        self._logger = logger or setup_logging(level=log_level, name="ObservationSet")
        self._observations: list[Observation] = []

    # ==================================================
    # Representation
    # ==================================================

    def __repr__(self) -> str:
        return f' ObservationSet : {self.n_observations} observations'

    def __format__(self) -> str:
        return self.__repr__()

    # ==================================================
    # Collection protocol
    # ==================================================

    def __len__(self) -> int:
        return len(self._observations)

    def __iter__(self):
        return iter(self._observations)

    def __getitem__(self, idx: int) -> Observation:
        return self._observations[idx]

    # ==================================================
    # Properties
    # ==================================================

    @property
    def is_empty(self) -> bool:                                                # Whether ObservationSet is empty
        return len(self.observations) == 0

    @property
    def logger(self) -> logging.Logger:                                        # Logger
        return self._logger

    @property
    def observation_names(self) -> list[str]:                                  # List of observation names
        return [obs.name for obs in self.observations]

    @property
    def observations(self) -> list[Observation]:                               # List of observations
        return self._observations

    @property
    def n_observations(self) -> int:                                           # Number of observations
        return len(self)

    @property
    def has_spectroscopy(self) -> bool:                                        # Whether the observations set has spectroscopy
        for obs in self._observations:
            if obs.ObsType == ObservationType.SPECTROSCOPIC.obstype:
                return True
        return False

    @property
    def has_photometry(self) -> bool:                                          # Whether the observations set has photometry
        for obs in self._observations:
            if obs.ObsType == ObservationType.PHOTOMETRIC.obstype:
                return True
        return False

    @property
    def spectral_observations(self) -> list[SpectralObservation]:              # List of spectroscopic observations
        return [obs for obs in self._observations if obs.ObsType == ObservationType.SPECTROSCOPIC]

    @property
    def photometry_observations(self) -> list[PhotometryObservation]:          # List of photometric observations
        return [obs for obs in self._observations if obs.ObsType == ObservationType.PHOTOMETRIC]

    @property                                                                  # Maximum resolution (None if no spectroscopic observation)
    def max_resolution(self) -> float | None:
        specs = self.spectral_observations
        if not specs:
            return None

        return None if not self.spectral_observations else max(obs.max_resolution for obs in specs)

    @property                                                                  # Minimum resolution (None if no spectroscopic observation)
    def min_resolution(self) -> float | None:
        specs = self.spectral_observations
        if not specs:
            return None

        return None if not self.spectral_observations else min(obs.min_resolution for obs in specs)

    @property
    def wavelength_range(self) -> tuple[float, float]:                         # Global wavelength range
        wmins = [obs.wavelength_range[0] for obs in self._observations]
        wmaxs = [obs.wavelength_range[1] for obs in self._observations]
        return min(wmins), max(wmaxs)

    @property
    def to_dict(self) -> dict:                                                 # Dictionary representation of the set of observations
        data = {}
        for i, name in enumerate(self.observation_names):
            data[name] = self.observations[i].to_dict
        return data

    # ==================================================
    # Class methods
    # ==================================================

    @classmethod
    def from_npz(cls, path: str | os.PathLike, logger: logging.Logger | None = None, log_level: str = 'INFO') -> "ObservationSet":
        '''
        Create an instance of ObservationSet from a path containing observation fits files.

        Parameters
        ----------
        path (str | os.PathLike): Path containing all the observations
        logger  (logging.Logger): Logger
        log_level          (str): Level of the Logger

        Returns
        -------
        "ObservationSet": An instance of ObservationSet

        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name='ObservationSet')

        logger.debug(f'Generating a set of observations from path {path}')

        # Initial checking
        if not isinstance(path, (str, os.PathLike)):
            raise ForMoSAError(f' Wrong type for path: {type(path)}. Expected a str or os.PathLike', logger)

        obs_set = cls(logger=logger)
        obs_path = Path(path).expanduser() / 'Observations'

        # Initial checks
        if not obs_path.exists():
            raise ForMoSAError(f'{obs_path} does not exist', logger)

        obs_files = [obs_file for obs_file in os.listdir(obs_path) if obs_file.endswith('.npz')]

        if len(obs_files) == 0:
            raise ForMoSAError(f'Wrong path extension for files: {obs_files}. Require a .npz')

        # Generate ordered observations
        order_file = obs_path / "observation_order.json"

        if order_file.exists():
            with open(order_file, "r") as f:
                ordered_names = json.load(f)
                obs_files = [f"Observation_{name}.npz" for name in ordered_names]
        else:
            logger.warning("No order_file found. The extracted observation order likely won't the initial observation order")

        for obs_file in obs_files:
            file_path = obs_path / obs_file
            obs = Observation.from_file(file_path, logger=logger)
            obs_set.add_observation(obs)

        logger.info('    Set of Observations generated')
        return obs_set

    @classmethod
    def from_fits(cls, path: str | os.PathLike, logger: logging.Logger | None = None, log_level: str = 'INFO') -> "ObservationSet":
        '''
        Create an instance of ObservationSet from a path containing observation fits files.

        Parameters
        ----------
        path (str | os.PathLike): Path containing all the observations
        logger  (logging.Logger): Logger
        log_level          (str): Level of the Logger

        Returns
        -------
        "ObservationSet": An instance of ObservationSet

        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name='ObservationSet')

        logger.debug(f'Generating a set of observations from path {path}')

        # Initial checking
        if not isinstance(path, (str, os.PathLike)):
            raise ForMoSAError(f' Wrong type for path: {type(path)}. Expected a str or os.PathLike', logger)

        obs_set = cls(logger=logger)
        obs_path = Path(path).expanduser()

        # Initial checks
        if not obs_path.exists():
            raise ForMoSAError(f'{obs_path} does not exist', logger)

        obs_files = [obs_file for obs_file in os.listdir(obs_path) if obs_file.endswith('.fits')]

        if len(obs_files) == 0:
            raise ForMoSAError(f'Wrong path extension for files: {obs_files}. Require a .fits')

        for obs_file in obs_files:
            file_path = obs_path / obs_file
            obs = Observation.from_file(file_path, logger=logger)
            obs_set.add_observation(obs)

        logger.info('    Set of Observations generated')
        return obs_set

    @classmethod
    def from_dict(cls, data: dict, logger: logging.Logger | None = None, log_level: str = 'INFO') -> 'ObservationSet':
        '''
        Reconstruct an ObservationSet from a dictionary of ObservationSet.

        Parameters
        ----------
        data                (dict): Dictionary containing ObservationSet parameters
        logger    (logging.Logger): Logger
        log_level            (str): Level of the logging

        Returns
        -------
        'ParameterSet': An instance of class ParameterSet

        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name='ParameterSet')

        if not isinstance(data, dict):
            raise ForMoSAError(f'Wrong type for data: {type(data)}. Expected a dictionary', logger)

        obs_set = cls(logger=logger)

        logger.debug('Build instance of ObservationSet from dictionary')

        for name in data.keys():
            obs = Observation.from_dict(data=data[name], logger=logger)
            obs_set.add_observation(obs)

        return obs_set

    @classmethod
    def from_json(cls, path: str | os.PathLike, logger: logging.Logger | None = None, log_level: str = 'INFO') -> 'ObservationSet':
        '''
        Reconstruct an ObservationSet from a json file.

        Parameters
        ----------
        path   (str | os.PathLike): Path to the json file
        logger    (logging.Logger): Logger
        log_level            (str): Level of the logging

        Returns
        -------
        'ParameterSet': An instance of class ParameterSet

        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name='ParameterSet')

        if not isinstance(path, (str, os.PathLike)):
            raise ForMoSAError(f'Wrong type for path: {type(path)}. Expected a string or os.PathLike', logger)

        logger.debug(f'Building instance of ObservationSet from json file {str(path) + "observations.json"}')

        filepath = Path(str(path) + 'observations.json')
        if not filepath.exists():
            raise ForMoSAError(f'{filepath} does not exist')

        with open(filepath, "r") as f:
            data = json.load(f)

        return cls.from_dict(data, logger=logger, log_level=log_level)

    # ==================================================
    # Methods
    # ==================================================

    def add_observation(self, *args, **kwargs):
        '''
        Add an observation to the set based on the type of data provided.

        Parameters
        ----------
        args :
            - If a Observation object is provided, directly add the observation
            - If a `.fits` file is provided, provide a single argument `path` (str | Path)
            - If a dictionary of data is provided, provide a single argument `data` (dict)
            - If attributes are provided, provide the necessary arguments to create the observation (Spectral or Photometric)

        kwargs : Additional attributes for the observations if necessary.

        Example:
        - self.add_observation(path="path/to/file.fits")
        - self.add_observation(data={"wavelength": ..., "flux": ...})
        - self.add_observation(name="spectral_obs", wavelength=..., flux=..., ...)

        Authors: Allan Denis
        '''

        if len(args) == 1:
            if isinstance(args[0], (Observation, SpectralObservation, PhotometryObservation)):
                # If the argument is an Observation
                obs = args[0]
            elif isinstance(args[0], (str, os.PathLike)):
                # If the argument is a path (FITS file)
                obs = Observation.from_file(args[0], logger=self.logger, **kwargs)
            elif isinstance(args[0], dict):
                # If the argument is a dictionary of data
                obs = Observation.from_dict(args[0], logger=self.logger, **kwargs)
            else:
                raise ForMoSAError(f"Unrecognized input type {type(args[0])}", self.logger)

        elif len(kwargs) > 1:
            # If multiple arguments are provided, we assume they are attributes
            obs = Observation.from_attributes(**kwargs)

        else:
            raise ForMoSAError('No valid data provided to add an observation', self.logger)

        self.logger.info(f'      Adding {obs.ObsType} Observation with name {obs.name} to the set of observations')
        self._observations.append(obs)

    def save_all(self, path: str | os.PathLike, to_json: bool = False) -> None:
        '''
        Save all observations to disk as .npz files.

        Parameters
        ----------
        path       (str | os.PathLike): Directory where to save the observations
        prefix                  (str): Prefix for the saved files
        to_json                (bool): Whether to save all observations in a json file

        Authors: Allan Denis
        '''

        path = Path(path).expanduser() / 'Observations'

        self.logger.info(f'    Saving all the observations {self.observation_names} to path {path}')

        if to_json is True:
            self.to_json(path)

        else:
            for obs in self.observations:
                obs.save_observation(path)

            # Save order
            order_file = path / "observation_order.json"
            with open(order_file, "w") as f:
                json.dump(self.observation_names, f, indent=4)

    def adapt_all(self, target_resolution: list[np.ndarray], wave_cont: list[str] | None = None, res_cont: list[float] | None = None) -> None:
        '''
        Adapt all observations to the target resolution.

        Parameters
        ----------
        target_resolution: (list[np.ndarray]): List of target resolution to reach for the observations
        wave_cont                 (list[str]): List of wavelengths used for the continuum
        res_cont                (list[float]): List os resolutions used for the continuum

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        # Initial checks
        if not isinstance(target_resolution, list):
            raise ForMoSAError(' Wrong type for target_resolution: {type(target_resolution)}. Expected a list', self.logger)

        if len(target_resolution) != self.n_observations:
            raise ForMoSAError(' Wrong length for target_resolution: {len(target_resolution)}. Expected {self.n_observations}', self.logger)

        if wave_cont is None:
            wave_cont = [None] * self.n_observations

        elif not isinstance(wave_cont, list):
            raise ForMoSAError(f' Wrong type for wave_cont: {type(wave_cont)}. Expected a list or None', self.logger)

        if res_cont is None:
            res_cont = [None] * self.n_observations

        elif not isinstance(res_cont, list):
            raise ForMoSAError(f' Wrong type for res_cont: {type(res_cont)}. Expected a list or None', self.logger)

        elif len(res_cont) != self.n_observations:
            raise ForMoSAError(f' Wrong length for res_cont: {len(res_cont)}. Expected {self.n_observations}', self.logger)

        self.logger.debug(f'    Adapting all the observations {self.observation_names}')

        # Adaptation to observations
        for i, obs in enumerate(self._observations):
            self._logger.info(f'    Adapting Observation: {obs.name}')
            self.observations[i] = obs._adapt_to_resolution(target_resolution[i], wave_cont=wave_cont[i], res_cont=res_cont[i])

        self.logger.info(f'    Observations {self.observation_names} adapted')

    def to_json(self, path: str | os.PathLike) -> None:
        '''
        Save the set of observations to a given path as a json file.

        Parameters
        ----------
        path (str | os.PathLike): Path to save the set of parameters

        Authors: Allan Denis
        '''

        if not isinstance(path, (str, os.PathLike)):
            raise ForMoSAError(f'Wrong type for path: {type(path)}. Expected a string or os.PathLike', self.logger)

        self.logger.info(f'    Saving set of observations to json path {Path(path) / "observations.json"}')

        path = Path(path)
        if not path.exists():
            self.logger.warning(f'{path} does not exist. Creating it.')
            path.mkdir(exist_ok=True, parents=True)

        with open(path / 'observations.json', 'w') as f:
            json.dump(self.to_dict, f, indent=4)

    def plot_all(self, figsize=(10,7), fig: Figure | None = None, ax: Axes | None = None, ax_filt: Axes | None = None) -> None:
        '''
        Plot all the observations and photometric filters.

        Parameters
        ----------
        figsize (tuple[float, float]): Size of the figure to plot if fig is None
        fig        (matplotlib.figure.Figure): Figure (used to overplot on an existing figure)
        ax         (matplotlib.axes._axes.Axes): Ax (used to overplot the observations)
        ax_filt    (matplotlib.axes._axes.Axes): Ax used to overplot the transmission filter

        Returns
        -------
        None

        Authors: Allan Denis
        '''

        self.logger.info(f'Plot all the observations {self.observation_names}')

        # Create figure if not provided
        if fig is None:
            fig = plt.figure(figsize=figsize)

        # Create main axis if not provided
        if ax is None:
            gs = gridspec.GridSpec(9, 10)
            ax = fig.add_subplot(gs[0:7, 0:10])

        # Create photometric filter axis only if not provided
        if self.has_photometry and ax_filt is None:
            gs = gridspec.GridSpec(9, 10)
            ax_filt = fig.add_subplot(gs[0:2, 0:10], sharex=ax)

        # Plot each observation
        for i, obs in enumerate(self.observations):
            if obs.ObsType == ObservationType.SPECTROSCOPIC.obstype:
                plot_config = SpectralPlotConfig(color=f'C{i}')

            elif obs.ObsType == ObservationType.PHOTOMETRIC.obstype:
                plot_config = PhotometricPlotConfig(color=f'C{i}')

            else:
                raise ForMoSAError(f' Unknown observation type: {obs.ObsType}. Expected {[type.obstype for type in ObservationType]}')

            # Plot on the appropriate axes
            obs.plot_data(fig=fig, ax=ax, ax_filt=ax_filt, plot_config=plot_config)