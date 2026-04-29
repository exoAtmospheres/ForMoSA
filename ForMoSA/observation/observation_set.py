import os
import json
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.axes._axes import Axes
from matplotlib import colors as mcolors
from matplotlib.ticker import AutoMinorLocator

from ForMoSA.core.config import MAIN_PLOT
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.enums import ObservationKeys
from ForMoSA.core.loggings import setup_logging
from ForMoSA.observation.observation_base import Observation
from ForMoSA.observation.observation_spectroscopy import SpectralObservation
from ForMoSA.observation.observation_photometry import PhotometryObservation

class ObservationSet(object):
    '''
    Container for a set of Observation objects.

    Parameters
    ----------
    logger : logging.Logger
        Logger
    log_level : str
        Level of the logging

    Notes
    -----
    Authors: Allan Denis
    '''

    def __init__(self, logger: logging.Logger | None = None, log_level: str = "INFO") -> None:

        self._logger = logger if logger is not None else setup_logging(level=log_level, name="ObservationSet")
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
    def is_empty(self) -> bool:
        """Whether ObservationSet is empty."""
        return len(self.observations) == 0

    @property
    def logger(self) -> logging.Logger:
        """Logger."""
        return self._logger

    @property
    def observation_names(self) -> list[str]:
        """List of observation names."""
        return [obs.name for obs in self.observations]

    @property
    def observations(self) -> list[Observation]:
        """List of observations."""
        return self._observations

    @property
    def n_observations(self) -> int:
        """Number of observations."""
        return len(self)

    @property
    def has_spectroscopy(self) -> bool:
        """Whether the observation set has spectroscopy."""
        for obs in self.observations:
            if obs.is_spectroscopic:
                return True
        return False

    @property
    def has_photometry(self) -> bool:
        """Whether the observation set has photometry."""
        for obs in self.observations:
            if obs.is_photometric:
                return True
        return False

    @property
    def has_high_contrast(self) -> bool:
        """Whether the observation set has high-contrast observations."""
        for obs in self.observations:
            if obs.hc_mode:
                return True
        return False

    @property
    def spectral_observations(self) -> list[SpectralObservation]:
        """List of spectroscopic observations."""
        return [obs for obs in self.observations if obs.is_spectroscopic]

    @property
    def photometry_observations(self) -> list[PhotometryObservation]:
        """List of photometric observations."""
        return [obs for obs in self.observations if obs.is_photometric]

    @property
    def high_contrast_observations(self) -> list[Observation]:
        """List of high-contrast observation."""
        return [obs for obs in self.observations if obs.hc_mode]

    @property
    def max_resolution(self) -> float | None:
        """Maximum resolution (None if no spectroscopic observation)."""
        specs = self.spectral_observations
        if not specs:
            return None

        return None if not self.spectral_observations else max(obs.max_resolution for obs in specs)

    @property
    def min_resolution(self) -> float | None:
        """Minimum resolution (None if no spectroscopic observation)."""
        specs = self.spectral_observations
        if not specs:
            return None

        return None if not self.spectral_observations else min(obs.min_resolution for obs in specs)

    @property
    def wavelength_range(self) -> tuple[float, float]:
        """Global wavelength range."""
        wmins = [obs.wavelength_range[0] for obs in self._observations]
        wmaxs = [obs.wavelength_range[1] for obs in self._observations]
        return min(wmins), max(wmaxs)

    @property
    def to_dict(self) -> dict:
        """Dictionary representation of the set of observations."""
        data = {}
        for i, name in enumerate(self.observation_names):
            data[name] = self.observations[i].to_dict
        return data

    @property
    def mcolors_normalize(self) -> mcolors.Normalize:
        """Color normalization (for plotting)."""
        return plt.Normalize(vmin=self.wavelength_range[0], vmax=self.wavelength_range[1])

    # ==================================================
    # Class methods
    # ==================================================

    @classmethod
    def from_npz(cls, path: str | os.PathLike, logger: logging.Logger | None = None, log_level: str = 'INFO') -> "ObservationSet":
        '''
        Create an instance of ObservationSet from a path containing observation fits files.

        Parameters
        ----------
        path : str | os.PathLike
            Path containing all the observations
        logger : logging.Logger
            Logger
        log_level : str
            Level of the Logger

        Returns
        -------
        "ObservationSet"
            An instance of ObservationSet

        Notes
        -----
        Authors: Allan Denis
        '''

        logger = logger if logger is not None else setup_logging(level=log_level, name='ObservationSet')

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
    def from_fits(cls, path: list[str | os.PathLike], logger: logging.Logger | None = None, log_level: str = 'INFO') -> "ObservationSet":
        '''
        Create an instance of ObservationSet from a path containing observation fits files.

        Parameters
        ----------
        path : list[str | os.PathLike
            Paths to the observations
        logger : logging.Logger
            Logger
        log_level : str
            Level of the Logger

        Returns
        -------
        "ObservationSet"
            An instance of ObservationSet

        Notes
        -----
        Authors: Allan Denis
        '''

        logger = logger if logger is not None else setup_logging(level=log_level, name='ObservationSet')

        logger.debug(f'Generating a set of observations from path {path}')

        # Initial checking
        if not isinstance(path, list):
            raise ForMoSAError(f'Wrong type for path: {type(path)}. Expected a list', logger)

        if not all(isinstance(obs_path, (str | os.PathLike)) for obs_path in path):
            raise ForMoSAError('path must be a list of str or os.PathLike')

        obs_set = cls(logger=logger)
        obs_files: list[Path] = []

        # Iterate over provided paths
        for p in path:

            p = Path(p).expanduser()

            if not p.exists():
                raise ForMoSAError(f'{p} does not exist', logger)

            # Case 1: it's a file
            if p.is_file():

                if p.suffix.lower() != '.fits':
                    raise ForMoSAError(f'{p} is not a .fits file', logger)

                obs_files.append(p)

            # Case 2: it's a directory
            elif p.is_dir():

                fits_in_dir = list(p.glob('*.fits'))

                if len(fits_in_dir) == 0:
                    logger.warning(f'No .fits files found in {p}')

                obs_files.extend(fits_in_dir)

        if len(obs_files) == 0:
            raise ForMoSAError('No .fits files found in provided paths', logger)

        # Load observations
        for file_path in obs_files:

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
        data : dict
            Dictionary containing ObservationSet parameters
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

        logger = logger if logger is not None else setup_logging(level=log_level, name='ParameterSet')

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

        logger = logger if logger is not None else setup_logging(level=log_level, name='ParameterSet')

        if not isinstance(path, (str, os.PathLike)):
            raise ForMoSAError(f'Wrong type for path: {type(path)}. Expected a string or os.PathLike', logger)

        logger.debug(f'Building instance of ObservationSet from json file {str(path) + "observations.json"}')

        filepath = Path(str(path) + 'observations.json')
        if not filepath.exists():
            raise ForMoSAError(f'{filepath} does not exist')

        with open(filepath, "r") as f:
            data = json.load(f)

        return cls.from_dict(data, logger=logger)

    @classmethod
    def from_list(cls, obs_list: list[Observation], logger: logging.Logger | None = None, log_level: str = 'INFO') -> 'ObservationSet':
        '''
        Reconstruct an ObservationSet from a list of observations.

        Parameters
        ----------
        obs_list : list[Observation]
            List containing observations
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

        logger = logger if logger is not None else setup_logging(level=log_level, name='ParameterSet')

        if not isinstance(obs_list, list):
            raise ForMoSAError(f'Wrong type for obs_list: {type(obs_list)}. Expected a list', logger)

        if not all(isinstance(obs, Observation) for obs in obs_list):
            raise ForMoSAError('Expected a list of observations for obs_list', logger)

        obs_set = cls(logger=logger)

        logger.debug('Build instance of ObservationSet from dictionary')

        for obs in obs_list:
            obs_set.add_observation(obs)

        return obs_set

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

        Notes
        -----
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
        path : str | os.PathLike
            Directory where to save the observations
        prefix : str
            Prefix for the saved files
        to_json : bool
            Whether to save all observations in a json file

        Notes
        -----
        Authors: Allan Denis
        '''

        path = Path(path).expanduser() / 'Observations'

        self.logger.info(f'    Saving all the observations {self.observation_names} to path {path}')
        if not path.exists():
            self.logger.warning(f'path {path} does not exist. Creating it')
            path.mkdir(exist_ok=True, parents=True)

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
        wave_cont : list[str]
            List of wavelengths used for the continuum
        res_cont : list[float]
            List os resolutions used for the continuum

        Notes
        -----
        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        # Initial checks
        if not isinstance(target_resolution, list):
            raise ForMoSAError(f' Wrong type for target_resolution: {type(target_resolution)}. Expected a list', self.logger)

        if len(target_resolution) != self.n_observations:
            raise ForMoSAError(f' Wrong length for target_resolution: {len(target_resolution)}. Expected {self.n_observations}', self.logger)

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
        path : str | os.PathLike
            Path to save the set of parameters

        Notes
        -----
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

    def plot_all(self, fig: Figure | None = None, ax: Axes | None = None, ax_hc: Axes | None = None, ax_filt: Axes | None = None) -> tuple[Figure, Axes, Axes | None]:
        '''
        Plot all the observations and photometric filters.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure (used to overplot on an existing figure)
        ax : matplotlib.axes._axes.Axes
            Ax (used to overplot non-high-contrast observations)
        ax_hc : matplotlib.axes._axes.Axes
            Ax used to overplot high-contrast observations. When None and ax is
            provided, HC observations fall back to ax (e.g. inside plot_fit).
        ax_filt : matplotlib.axes._axes.Axes
            Ax used to overplot the transmission filter

        Returns
        -------
        fig : matplotlib.figure.Figure
            New Figure object
        ax : matplotlib.axes._axes.Axes
            New Ax object
        ax_filt : matplotlib.axes._axes.Axes
            New ax object for photometric filters

        Notes
        -----
        Authors: Allan Denis
        '''

        self.logger.info(f'    Plotting all the observations {self.observation_names}')

        main_plot_config = MAIN_PLOT

        # Create figure if not provided
        if fig is None:
            fig = plt.figure(figsize=main_plot_config.figsize)

        # Create axes for observations if not provided
        if ax is None and ax_hc is None:

            # Create a gridspec to have more control over the layout of the axes
            gs = gridspec.GridSpec(9, 10)

            # If we have both high-contrast and non-high-contrast observations, we create two separate axes for them
            if self.has_high_contrast and len(self.high_contrast_observations) != self.n_observations:
                ax = fig.add_subplot(gs[2:6, 0:10])
                ax_hc = fig.add_subplot(gs[6:9, 0:10], sharex=ax)

            # If we only have high-contrast observations, we use the whole space for the high-contrast axis
            else:
                if self.has_high_contrast:
                    ax_hc = fig.add_subplot(gs[2:9, 0:10])
                else:
                    ax = fig.add_subplot(gs[2:9, 0:10])

        # Create photometric filter axis only if not provided
        if self.has_photometry and ax_filt is None:
            gs = gridspec.GridSpec(9, 10)
            ax_filt = fig.add_subplot(gs[0:2, 0:10], sharex=(ax if ax is not None else ax_hc))

        # Plot each observation — legend is suppressed here;
        # plot_all renders a single consolidated legend below.
        for obs in self.observations:

            if not obs.hc_mode:
                fig, ax, ax_filt = obs.plot_data(fig=fig, ax=ax, ax_filt=ax_filt, draw_legend=False)

            else:
                # When ax_hc is not set (e.g. called with a pre-existing ax from
                # plot_fit), fall back to ax so data lands on the main axes.
                _target_hc = ax_hc if ax_hc is not None else ax
                fig, _target_hc, ax_filt = obs.plot_data(fig=fig, ax=_target_hc, ax_filt=ax_filt, draw_legend=False)
                if ax_hc is not None:
                    ax_hc = _target_hc
                else:
                    ax = _target_hc

        # Use whichever axis was effectively used for spectral plotting.
        plot_axis = ax if ax is not None else ax_hc
        if plot_axis is None:
            raise ForMoSAError('No plotting axis available for observations', self.logger)

        if self.has_high_contrast and len(self.high_contrast_observations) == self.n_observations:
            ncol = max(1, int(main_plot_config.legend_hc_ncol))
        else:
            ncol = max(1, int(main_plot_config.legend_ncol))

        handles, labels = plot_axis.get_legend_handles_labels()
        if handles:
            plot_axis.legend(ncol=ncol, frameon=False, loc='upper right', fontsize=main_plot_config.legend_fontsize)

        # Add legend for photometric filters if we have photometry and an axis for the filters
        if ax_filt is not None:
            filt_handles, filt_labels = ax_filt.get_legend_handles_labels()
            if filt_handles:
                ax_filt.legend(ncol=max(1, int(main_plot_config.legend_filt_ncol)), frameon=False)

        # Minor ticks
        if main_plot_config.minor_ticks:
            # Principal axis
            plot_axis.xaxis.set_minor_locator(AutoMinorLocator(main_plot_config.nb_minor_ticks))
            plot_axis.yaxis.set_minor_locator(AutoMinorLocator(main_plot_config.nb_minor_ticks))

            if ax_filt is not None:
                # Filter axis
                ax_filt.xaxis.set_minor_locator(AutoMinorLocator(main_plot_config.nb_minor_ticks))
                ax_filt.yaxis.set_minor_locator(AutoMinorLocator(main_plot_config.nb_minor_ticks))

        # Rescale y axis with a power of 10
        ymin, ymax = plot_axis.get_ylim()
        ymax_abs = max(abs(ymin), abs(ymax))
        if ymax_abs > 0:
            exponent = int(np.floor(np.log10(ymax_abs)))
            plot_axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{y/10**exponent:.1f}"))
            plot_axis.set_ylabel(rf'Flux ($10^{{{exponent}}}$  W.m$^{{-2}}$.$\mu$m$^{{-1}}$)')
        else:
            plot_axis.set_ylabel(r'Flux (W.m$^{-2}$.$\mu$m$^{-1}$)')

        return fig, plot_axis, ax_filt

    def _stack(self, ind_obs: list[int] | None = None, print_logger: bool = False) -> dict:
        '''
        Stack all observations using the dictionary representation.

        Parameters
        ----------
        ind_obs : list[int]
            List of index of observations to stack. If None, stack all observations

        Returns
        -------
        dict
            Stacked observations sorted by wavelength

        Notes
        -----
        Authors: Allan Denis
        '''

        if print_logger:
            self.logger.info("    Stacking observations")

        wavelength_all = []
        flux_all = []
        error_all = []
        res_all = []

        if ind_obs is None:
            ind_obs = np.arange(self.n_observations)

        for iobs in ind_obs:

            obs_data = self.observations[iobs]

            wave = np.atleast_1d(obs_data.wave)
            flux = np.atleast_1d(obs_data.flux)
            err = np.atleast_1d(obs_data.err)
            res = np.atleast_1d(obs_data.res)

            wavelength_all.append(wave)
            flux_all.append(flux)
            error_all.append(err)
            res_all.append(res)

        # concatenate
        wavelength_all = np.concatenate(wavelength_all)
        flux_all = np.concatenate(flux_all)
        error_all = np.concatenate(error_all)
        res_all = np.concatenate(res_all)

        # sort by wavelength
        idx = np.argsort(wavelength_all)

        stacked = {
            ObservationKeys.WAVELENGTH.canonical : wavelength_all[idx],
            ObservationKeys.FLUX.canonical: flux_all[idx],
            ObservationKeys.ERROR.canonical: error_all[idx],
            ObservationKeys.RESOLUTION.canonical: res_all[idx],
        }

        return stacked