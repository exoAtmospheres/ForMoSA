import os
import logging
import numpy as np
from pathlib import Path
import astropy.units as u
from abc import ABC, abstractmethod
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes

from ForMoSA.core.config import ObsPlotConfig
from ForMoSA.core.loggings import setup_logging
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.enums import WavelengthUnit, ObservationType


class Observation(ABC):
    '''
    Base class for any observation (photometric or spectroscopic).

    Parameters
    ----------
    wave : np.ndarray
        Wavelength array
    flux : np.ndarray
        Flux array
    err : np.ndarray
        Error array
    native_unit : WavelengthUnit
        Native unit of the wavelength array
    facility : str
        Facility name
    instrument : str
        Instrument name
    logger : logging.Logger
        Logger
    log_level : str
        Level of the logging
    display_unit : WavelengthUnit
        Display unit of the wavelength array
    plot_config : ObsPlotConfig
        Plot configuration for the observation

    Notes
    -----
    Authors: Allan Denis
    '''

    def __init__(self, wave: np.ndarray, flux: np.ndarray, err: np.ndarray, native_unit: WavelengthUnit, facility: str, instrument: str, logger: logging.Logger | None = None, log_level:str = 'INFO', display_unit: WavelengthUnit = WavelengthUnit.MICROMETER, plot_config: ObsPlotConfig = ObsPlotConfig()) -> None:

        self._logger = logger or setup_logging(log_level)

        self._wave = np.atleast_1d(np.asarray(wave, dtype=float))
        self._flux = np.atleast_1d(np.asarray(flux, dtype=float))
        self._err  = np.atleast_1d(np.asarray(err, dtype=float))
        self._native_unit = native_unit
        self._display_unit = display_unit
        self._facility  = np.atleast_1d(np.asarray(facility, dtype=str))
        self._instrument  = np.atleast_1d(np.asarray(instrument, dtype=str))

        self._plot_config = plot_config

        self._validate()

    # ==================================================
    # Abstract methods
    # (force the subclasses to implement these methods)
    # ==================================================

    @property
    @abstractmethod
    def ObsType(self) -> ObservationType.obstype:
        """Observation type."""
        pass

    @property
    @abstractmethod
    def to_dict(self) -> dict[str, np.ndarray]:
        """Dictionary representation of the observations."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Observation name."""
        pass

    @property
    @abstractmethod
    def wavelength_range(self) -> tuple[float, float]:
        """Wavelength range."""
        pass

    @property
    @abstractmethod
    def res(self) -> np.ndarray[float]:
        """Resolution."""
        pass

    @property
    @abstractmethod
    def hc_mode(self) -> bool:
        """Whether observation is in high-contrast mode."""
        pass

    @abstractmethod
    def _adapt_to_resolution(self, target_resolution: np.ndarray, wave_cont: str | None = None, res_cont: float | None = None) -> "Observation":
        '''
        Adapt the spectral observation to the target resolution.

        Notes
        -----
        Authors: Allan Denis
        '''
        pass

    @abstractmethod
    def plot_data(self, fig: Figure | None = None, ax: Axes | None = None, ax_filt: Axes | None = None, draw_legend: bool = True) -> tuple[Figure, Axes, Axes]:
        '''
        Plot the observation.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure (used to overplot on an existing figure)
        ax : matplotlib.axes._axes.Axes
            Ax (used to overplot on an existing ax)
        ax_filt : matplotlib.axes._axes.Axes
            Ax used to overplot the transmission filter on an existing ax
        draw_legend : bool
            Whether to draw the legend. Set to False when called from a
            parent function (e.g. plot_all) that manages the legend itself.

        Notes
        -----
        Authors: Allan Denis
        '''

    @abstractmethod
    def _restricted_observation(self, windows: str | None = None, print_logger: bool=True) -> "Observation":
        '''
        Restrict the observation to wavelength windows.

        Parameters
        ----------
        windows : str
            Windows in the format 'wmin1,wmax1 / wmin2,wmax2 / ...'

        Returns
        -------
        dict
            Restricted observation data

        Notes
        -----
        Authors: Allan Denis
        '''

        pass

    # ==================================================
    # Common properties
    # ==================================================

    @property
    def is_spectroscopic(self) -> bool:
        """Whether observation is spectroscopic."""
        return self.ObsType == ObservationType.SPECTROSCOPIC.obstype

    @property
    def is_photometric(self) -> bool:
        """Whether observation is photometric."""
        return self.ObsType == ObservationType.PHOTOMETRIC.obstype

    @property
    def native_unit(self) -> u.core.Unit:
        """Native unit of the wavelength array."""
        return self._native_unit.unit

    @property
    def unit(self) -> u.core.PrefixUnit:
        """Display unit of the wavelength array."""
        return self._display_unit.unit

    @property
    def wave(self) -> np.ndarray:
        """Wavelength array."""
        return ((self._wave * self.native_unit).to(self.unit)).value

    @property
    def central_wavelength(self) -> float:
        """Central wavelength."""
        return (self.wavelength_range[0] + self.wavelength_range[1]) / 2

    @property
    def flux(self) -> np.ndarray[float]:
        """Flux array."""
        return self._flux

    @property
    def err(self) -> np.ndarray[float]:
        """Error array."""
        return self._err

    @property
    def facility(self) -> np.ndarray[str]:
        """Facility (e.g. 'JWST', 'Keck', 'Paranal')."""
        return self._facility

    @property
    def instrument(self) -> np.ndarray[str]:
        """Instrument (e.g. 'NIRCam', 'NIRC2', 'SPHERE')."""
        return self._instrument

    @property
    def n_points(self) -> int:
        """Number of points."""
        return len(self.wave)

    @property
    def logger(self) -> logging.Logger:
        """Logger."""
        return self._logger

    @property
    def path(self) -> Path:
        """Path of the observation (if any)."""
        return Path(self._path) if self._path is not None else 'in-memory observation'

    @property
    def plot_config(self) -> ObsPlotConfig:
        """Configuration plotting."""
        return self._plot_config

    @plot_config.setter
    def plot_config(self, config: ObsPlotConfig):
        """Configuration plotting setter."""
        self._plot_config = config

    # ================================================
    # Class methods
    # ================================================

    @classmethod
    def from_dict(cls, data: dict, logger: logging.Logger | None = None, log_level: str = 'INFO', **kwargs) -> "Observation":
        '''
        Generate Observation from dictionary of data.

        Parameters
        ----------
        data : dict
            Dictionary of data
        logger : logging.Logger
            Logger
        log_level : str
            Level of the Logger
        **kwargs               : Additional arguments

        Returns
        -------
        Obervation
            An instance of class Observation

        Examples
        --------
        >>> obs = Observation.from_dict(data, logger, log_level)

        Notes
        -----
        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name="Observation")
        logger.debug('Extracting Observation from dictionary')

        from ForMoSA.observation.observation_loader import ObservationLoader
        try:
            return ObservationLoader._from_data(data, logger=logger, **kwargs)
        except ForMoSAError as e:
            raise ForMoSAError(e, logger)

    @classmethod
    def from_file(cls, path: str | os.PathLike, logger: logging.Logger | None = None, log_level: str = 'INFO', **kwargs) -> "Observation":
        '''
        Generate Observation from a fits file.

        Parameters
        ----------
        path : str | os.PathLike
            Path to the observation
        logger : logging.Logger
            Logger
        log_level : str
            Level of the Logger
        **kwargs                : Additional keyword arguments

        Returns
        -------
        "Observation"
            Instance of class Observation

        Examples
        --------
        >>> obs = Observation._from_file(path, logger, log_level)

        Notes
        -----
        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name="Observation")
        logger.debug(f'Extracting observation from file {path}')

        from ForMoSA.observation.observation_loader import ObservationLoader

        try:
            # fits file
            if Path(path).suffix == '.fits':
                return ObservationLoader._from_fits(path, logger=logger, **kwargs)
            # npz file
            elif Path(path).suffix == '.npz':
                data = dict(np.load(path, allow_pickle=True))
                return cls.from_dict(data, logger=logger)
            else:
                raise ForMoSAError(f'Unknown path extension: {Path(path).suffix[1:]}. Require a fits or npz extension', logger)

        except ForMoSAError as e:
            raise ForMoSAError(f'Error for observation path {path}: {e}', logger)

    @classmethod
    def from_attributes(cls, logger: logging.Logger | None = None, log_level: str = 'INFO', **kwargs) -> "Observation":
        '''
        Generation Observation from attributes.

        Parameters
        ----------
        **kwargs : Keyword attributes

        Returns
        -------
        "Observation"
            Instance of class Observation

        Examples
        --------
        >>> obs = Observation._from_attributes(**attributes, logger, log_level)

        Notes
        -----
        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name="Observation")
        logger.debug('Extractin observation from attributes')

        from ForMoSA.observation.observation_loader import ObservationLoader

        try:
            return ObservationLoader._from_attributes(logger=logger, **kwargs)
        except ForMoSAError as e:
            raise ForMoSAError(f'Error for observation with attributes {kwargs}: {e}', logger)

    # ==================================================
    # Methods
    # ==================================================

    def _validate(self) -> None:
        '''
        Check consistency in wavelength, flux and error

        Notes
        -----
        Authors: Allan Denis
        '''

        if not (len(self._wave) == len(self._flux) == len(self._err) == len(self.instrument) == len(self.facility)):
            raise ForMoSAError(f'wave ({len(self.wave)}), flux ({len(self.flux)}), err ({len(self.err)}), instrument ({len(self.instrument)}) and facility ({len(self.facility)}) must have same length', self.logger)

        if not isinstance(self._native_unit, WavelengthUnit):
            raise ForMoSAError(f'Wrong type for native_unit: {type(self._native_unit)}. Expected a WavelengthUnit', self.logger)

        if not isinstance(self._display_unit, WavelengthUnit):
            raise ForMoSAError(f'Wrong type for display_unit: {type(self._display_unit)}. Expected a WavelengthUnit', self.logger)

        valid_units = [unit.unit for unit in WavelengthUnit]
        for unit in [self.native_unit, self.unit]:
            if unit not in valid_units:
                raise ForMoSAError(f'Wrong unit: {unit}. Chose amongst {valid_units}', self.logger)

        if np.any(self.err <= 0):
            raise ForMoSAError('Error must be strictly positive', self.logger)

    def _set_unit(self, unit: WavelengthUnit) -> None:
        '''
        Set the display unit of the wavelength array.

        Parameters
        ----------
        unit : WavelengthUnit
            Desired display unit

        Notes
        -----
        Authors: Allan Denis
        '''

        if not(isinstance(unit, WavelengthUnit)):
            raise ForMoSAError(f'unit must be an instance of WavelengthUnit enum. Instead got {type(unit)}', self.logger)
        self._display_unit = unit

    def save_observation(self, store_path: str | os.PathLike) -> None:
        '''
        Save observation to disk as .npz files.

        Parameters
        ----------
        store_path : str | os.PathLike
            Path where to store the observation file

        Notes
        -----
        Authors: Allan Denis
        '''

        self.logger.debug(f'Save observation {self.name} to path {store_path}')
        # Get the saving path and automatically create it if it does not exist
        if not isinstance(store_path, str | os.PathLike):
            raise ForMoSAError(f'Wrong type for store_path: {type(store_path)}. Expected a string or os.PathLike', self.logger)

        path = Path(store_path).expanduser()
        filename = f"Observation_{self.name}.npz"
        self.logger.info(f"      Saving Observation Observation_{self.name}.npz")

        if not path.exists():
            self.logger.warning(f'{path} does not exist. Creating it')
            path.mkdir(exist_ok=True, parents=True)

        # Save dictionnary of observation to path
        np.savez(path / filename, **self.to_dict)