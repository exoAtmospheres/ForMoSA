import copy
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.axes._axes import Axes

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.filter.filter import PhotometryFilter
from ForMoSA.observation.observation_base import Observation
from ForMoSA.core.config import  PhotometricPlotConfig, PHOTOMETRIC_PLOT
from ForMoSA.core.enums import ObservationType, ObservationKeys, WavelengthUnit


class PhotometryObservation(Observation):
    '''
    Photometric observation class.

    Parameters
    ----------
    wave                (np.ndarray): Wavelength array
    flux                (np.ndarray): Flux array
    err                 (np.ndarray): Error array
    instrument                 (str): Instrument
    facility                   (str): Facility
    filter_id                  (str): Filter ID
    native_unit     (WavelengthUnit): native unit of the wavelength
    logger          (logging.Logger): Logger
    log_level                  (str): Level of the logger
    display_unit    (WavelengthUnit): Unit of the wavelength to display

    Authors: Allan Denis
    '''

    def __init__(self, wave: np.ndarray, flux: np.ndarray, err: np.ndarray, instrument: str, facility: str, filter_id: str, native_unit: WavelengthUnit, logger: logging.Logger | None = None, log_level: str = 'INFO', display_unit: WavelengthUnit = WavelengthUnit.MICROMETER) -> None:

        self._filter_id = filter_id
        # Inherit from Observation class
        super().__init__(wave=wave, flux=flux, err=err, facility=facility, instrument=instrument, native_unit=native_unit, logger=logger, log_level=log_level, display_unit=display_unit)

        self._Filter = PhotometryFilter(facility, instrument, filter_id)

        self._validate_photometry()
        self._Filter._set_unit(display_unit)

    # ==================================================
    # Representation
    # ==================================================

    def __repr__(self) -> str:
        return f' PhotometricObservation : {self.facility} - {self.instrument} - {self.filter_id}'

    def __format__(self) -> str:
        return self.__repr__()

    # ==================================================
    # Properties
    # ==================================================

    @property
    def ObsType(self) -> ObservationType:                         # Observation type
        return ObservationType.PHOTOMETRIC.obstype

    @property
    def res(self) -> np.ndarray[float]:                           # Resolution
        return np.array([0.0])

    @property
    def hc_mode(self) -> bool:                                    # Whether observation is in high-contrast mode
        return False

    @property
    def to_dict(self) -> dict[str, np.ndarray]:                   # Dictionary representation of photometric observations
        return {
            ObservationKeys.WAVELENGTH.canonical: self.wave.tolist(),
            ObservationKeys.FLUX.canonical: self.flux.tolist(),
            ObservationKeys.ERROR.canonical: self.err.tolist(),
            ObservationKeys.FACILITY.canonical: self.facility,
            ObservationKeys.INSTRUMENT.canonical: self.instrument,
            ObservationKeys.FILTER_ID.canonical: self.filter_id,
            ObservationKeys.WAVELENGTH_UNIT.canonical: str(WavelengthUnit[str(self.unit)].value),
        }

    @property
    def Filter(self) -> PhotometryFilter:                         # Photometric filter
        return self._Filter

    @Filter.setter
    def Filter(self, new_filter: PhotometryFilter) -> None:       # Setter for photometric filter
        if not isinstance(new_filter, PhotometryFilter):
            raise ForMoSAError(f'Wrong type for Filter: {type(new_filter)}. Expected an instance of PhotometryFilter', self.logger)
        self._Filter = new_filter
        self._validate_photometry()
        self._Filter._set_unit(self.display_unit)

    @property
    def filter_id(self) -> str:                                   # Filter ID
        return self._filter_id

    @property
    def name(self) -> str:                                        # Name of the observation
        return f"{self.facility}_{self.instrument}_{self.filter_id}"

    @property
    def wavelength_range(self) -> tuple:                          # Wavelength range of the observation
        return self.Filter.wavelength_range

    # ==================================================
    # Methods
    # ==================================================

    def _validate_photometry(self) -> None:
        '''
        Do some checks on photometric observations.

        Authors: Allan Denis
        '''

        if not isinstance(self.Filter, PhotometryFilter):
            raise ForMoSAError(f'Wrong type for Filter: {type(self.Filter)}. Expected an instance of PhotometryFilter', self.logger)

        if (self.wave[0] < self.Filter.wavelength_min) or (self.wave[0] > self.Filter.wavelength_max):
            raise ForMoSAError(f'Wrong value for wave: {self.wave}. Expected a value between {list(self.Filter.wavelength_range)}', self.logger)

    def _adapt_to_resolution(self, target_resolution: float | None = None, wave_cont: str | None = None, res_cont: float | None = None) -> "PhotometryObservation":
        '''
        For photometry, this function does not implement anything.

        Authors: Allan Denis
        '''

        self.logger.info(f'      Observation {self.name} is photometric. No adaptation')

        return self

    def plot_data(self, fig: Figure | None = None, ax: Axes | None = None, ax_filt: Axes | None = None, plot_config: PhotometricPlotConfig = PHOTOMETRIC_PLOT) -> tuple[Figure, Axes, Axes]:
        '''
        Plot photometric data.

        Parameters
        ----------
        figure     (matplotlib.figure.Figure): Figure (used to overplot on an existing figure)
        ax       (matplotlib.axes._axes.Axes): Ax (used to overplot on an existing ax)
        ax_filt  (matplotlib.axes._axes.Axes): Ax used to overplot the transmission filter on an existing ax
        plot_config   (PhotometricPlotConfig): Instance of class PhotometricPlotConfig

        Returns
        -------
        fig        (matplotlib.figure.Figure): Updated figure
        ax       (matplotlib.axes._axes.Axes): Updated ax
        ax_filt  (matplotlib.axes._axes.Axes): Updated ax_filt

        Authors: Allan Denis
        '''

        self.logger.info(f'      Plotting data for observation {self.name}')

        # Create figure and axes if not provided
        if ax is None or ax_filt is None:
            fig = plt.figure(figsize=plot_config.figsize)
            gs = gridspec.GridSpec(9, 10)
            ax = fig.add_subplot(gs[3:9, 0:10])
            ax_filt = fig.add_subplot(gs[0:3, 0:10], sharex=ax)

        # Plot filter transmission
        ax_filt.plot(self.Filter.wavelength, self.Filter.transmission, label=getattr(self.Filter, "name"), color=plot_config.color)
        ax_filt.legend()
        ax_filt.set_ylabel("Transmission")

        # Plot data points
        label = None
        if plot_config.label:
            label = self.facility + '/' + self.instrument + '.' + self.filter_id
        ax.scatter(self.wave, self.flux, color=plot_config.color, edgecolors=plot_config.edgecolor, marker=plot_config.marker, s=plot_config.markersize, linewidths=plot_config.linewidth, zorder=plot_config.zorder_data, label = label)

        # Plot error bars
        ax.errorbar(self.wave, self.flux, yerr=self.err, xerr=self.Filter.width, fmt=plot_config.errorbar_fmt, ecolor=plot_config.color, alpha=plot_config.errorbar_alpha, capsize=plot_config.errorbar_capsize, zorder=plot_config.zorder_error)

        # Axis labels
        ax.set_xlabel(f"Wavelength ({getattr(self, 'unit', '')})")
        ax.set_ylabel("Flux")

        return fig, ax, ax_filt

    def _restricted_observation(self, windows: str | None = None, print_logger: bool = True) -> "PhotometryObservation":
        '''
        Restrict the observation to wavelength windows.

        Parameters
        ----------
        windows        (str): Windows in the format 'wmin1,wmax1 / wmin2,wmax2 / ...'
        print_logger  (bool): Whether to print logger

        Returns
        -------
        PhotometryObservation: Restricted observation

        Authors: Allan Denis
        '''

        # Dictionary of the observation
        restricted = copy.deepcopy(self)

        if windows is None:
            windows = f'{self.wave[0]}, {self.wave[-1]}'

        if print_logger:
            self.logger.debug(f'Restricting observation {self.name} onto wavelengths windows {windows}')

        ind = np.array([], dtype=int)
        for window in windows.split("/"):
            wmin, wmax = map(float, window.split(","))
            indices = np.where((self.wave >= wmin) & (self.wave <= wmax))[0]
            ind = np.concatenate((ind, indices))
        ind = np.unique(ind)

        for name, value in zip(['_wave', '_flux', '_err'], [self.wave, self.flux, self.err]):
            if value is not None:
                setattr(restricted, name, value[ind])

        if print_logger:
            self.logger.info(f'Length of former Observation: {len(self.wave)}. Lengths of restricted obervation: {len(restricted.wave)}')

        return restricted