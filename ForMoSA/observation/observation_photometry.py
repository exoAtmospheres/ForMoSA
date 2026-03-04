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
    wave : np.ndarray
        Wavelength array
    flux : np.ndarray
        Flux array
    err : np.ndarray
        Error array
    instrument : np.ndarray
        Instrument
    facility : np.ndarray
        Facility
    filter_id : np.ndarray
        Filter ID
    native_unit : WavelengthUnit
        native unit of the wavelength
    logger : logging.Logger
        Logger
    log_level : str
        Level of the logger
    display_unit : WavelengthUnit
        Unit of the wavelength to display

    Authors
    -------
    Allan Denis
    '''

    def __init__(self, wave: np.ndarray, flux: np.ndarray, err: np.ndarray, instrument: np.ndarray, facility: np.ndarray, filter_id: np.ndarray, native_unit: WavelengthUnit, logger: logging.Logger | None = None, log_level: str = 'INFO', display_unit: WavelengthUnit = WavelengthUnit.MICROMETER) -> None:

        self._filter_id = np.atleast_1d(np.asarray(filter_id, dtype=str))
        # Inherit from Observation class
        super().__init__(wave=wave, flux=flux, err=err, facility=facility, instrument=instrument, native_unit=native_unit, logger=logger, log_level=log_level, display_unit=display_unit)

        self._Filter = np.array([])

        self._validate_photometry()

    # ==================================================
    # Representation
    # ==================================================

    def __repr__(self) -> str:
        return f' PhotometricObservation : {self.name}'

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
        return np.array([0.0] * len(self.wave))

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
    def Filter(self) -> np.ndarray[PhotometryFilter]:             # Photometric filters
        return self._Filter

    @property
    def filter_id(self) -> np.ndarray[str]:                       # Filter ID
        return self._filter_id

    @property
    def name(self) -> str:                                        # Observation name
        # ---- Facilities
        facilities = sorted(set(self.facility.astype(str)))
        facility_str = f'[{"+".join(facilities)}]'

        # ---- Instruments
        instruments = sorted(set(self.instrument.astype(str)))
        instrument_str = f'[{"+".join(instruments)}]'

        # ---- Filters
        filters = sorted(set(self.filter_id.astype(str)))
        nfilters = len(filters)

        # Condense if too many filters
        if nfilters <= 6:
            filter_str = f'[{"+".join(filters)}]'
        else:
            filter_str = f"[{nfilters}filters]"

        return f"{facility_str}_{instrument_str}_{filter_str}"

    @property
    def wavelength_range(self) -> tuple:                          # Wavelength range of the observation
        wmin = np.min([filt.wavelength_min for filt in self.Filter])
        wmax = np.max([filt.wavelength_max for filt in self.Filter])
        return wmin, wmax

    @property
    def filter_idxs(self) -> np.ndarray:                          # Indexes of occurence of new filters
        idxs = np.array([0])
        last_filt_id = self.filter_id[0]

        for idx, filt_id in enumerate(self.filter_id):
            if filt_id != last_filt_id:
                idxs = np.append(idxs, idx)
                last_filt_id = filt_id

        idxs = np.append(idxs, len(self.filter_id))
        return idxs

    @property
    def nb_filters(self) -> int:                                  # Number of filters
        return len(self.filter_idxs) - 1

    # ==================================================
    # Methods
    # ==================================================

    def _validate_photometry(self) -> None:
        '''
        Do some checks on photometric observations.

        Authors
        -------
        Allan Denis
        '''

        if not len(self.filter_id) == len(self.instrument):
            raise ForMoSAError('filter_id and instrument must have same lengths', self.logger)

        for i, (filt_id, facility, instrument) in enumerate(zip(self.filter_id, self.facility, self.instrument)):
            self._Filter = np.append(self._Filter, PhotometryFilter(self.facility[i], self.instrument[i], filt_id))
            self._Filter[i]._set_unit(WavelengthUnit[str(self.unit)])

        if (self.wave[0] < self.wavelength_range[0]) or (self.wave[0] > self.wavelength_range[1]):
            raise ForMoSAError(f'Wrong value for wave: {self.wave}. Expected a value between {list(self.wavelength_range)}', self.logger)

    def _adapt_to_resolution(self, target_resolution: float | None = None, wave_cont: str | None = None, res_cont: float | None = None) -> "PhotometryObservation":
        '''
        For photometry, this function does not implement anything.

        Authors
        -------
        Allan Denis
        '''

        self.logger.info(f'      Observation {self.name} is photometric. No adaptation')

        return self

    def plot_data(self, fig: Figure | None = None, ax: Axes | None = None, ax_filt: Axes | None = None, plot_config: PhotometricPlotConfig = PHOTOMETRIC_PLOT) -> tuple[Figure, Axes, Axes]:
        '''
        Plot photometric data.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            Figure (used to overplot on an existing figure)
        ax : matplotlib.axes._axes.Axes
            Ax (used to overplot on an existing ax)
        ax_filt : matplotlib.axes._axes.Axes
            Ax used to overplot the transmission filter on an existing ax
        plot_config : PhotometricPlotConfig
            Instance of class PhotometricPlotConfig

        Returns
        -------
        fig : matplotlib.figure.Figure
            Updated figure
        ax : matplotlib.axes._axes.Axes
            Updated ax
        ax_filt : matplotlib.axes._axes.Axes
            Updated ax_filt

        Authors
        -------
        Allan Denis
        '''

        self.logger.info(f'      Plotting data for observation {self.name}')

        # --------------------------------------------------
        # Figure / axes creation
        # --------------------------------------------------
        if ax is None or ax_filt is None:
            fig = plt.figure(figsize=plot_config.figsize)
            gs = gridspec.GridSpec(9, 10)
            ax = fig.add_subplot(gs[3:9, 0:10])
            ax_filt = fig.add_subplot(gs[0:3, 0:10], sharex=ax)

        ax_filt.set_ylabel("Transmission")
        ax.set_xlabel(f"Wavelength ({getattr(self, 'unit', '')})")
        ax.set_ylabel("Flux")

        # --------------------------------------------------
        # Plot data
        # --------------------------------------------------

        filt_handles = []
        filt_labels = []

        for i, filt in enumerate(self.Filter):
            idx0 = self.filter_idxs[i]
            idx1 = self.filter_idxs[i + 1]

            # Use central wavelength to determine color
            plot_config.set_photometric_plot_config(color = plot_config.cmap(plot_config.norm((filt.central_wavelength))))

            # ------------------------
            # Transmission (TOP PANEL)
            # ------------------------
            fig, ax_filt = filt._plot_transmission_curve(fig=fig, ax=ax_filt, plot_config=plot_config)

            # ------------------------
            # Photometric points (MID PANEL)
            # ------------------------

            ax.scatter(
                self.wave[idx0:idx1],
                self.flux[idx0:idx1],
                color=plot_config.color,
                edgecolors=plot_config.edgecolor,
                marker=plot_config.marker,
                s=plot_config.markersize,
                linewidths=plot_config.linewidth,
                zorder=plot_config.zorder_data,
                label=f'{filt.name}'
            )

            ax.errorbar(
                self.wave[idx0:idx1],
                self.flux[idx0:idx1],
                yerr=self.err[idx0:idx1],
                xerr=filt.width,
                fmt=plot_config.errorbar_fmt,
                ecolor=plot_config.color,
                alpha=plot_config.errorbar_alpha,
                capsize=plot_config.errorbar_capsize,
                zorder=plot_config.zorder_error
            )


            # Get handles of ax_filt
            lines = ax_filt.get_lines()
            filt_handles.extend(lines)
            filt_labels.append(f"{filt.name}")

        # --------------------------------------------------
        # Legend upper panel (Photometric data)
        # --------------------------------------------------

        if plot_config.label_filter:
            plot_config.legend_filter_ncol = (self.nb_filters + 4) // 5
            ax_filt.legend(filt_handles, filt_labels, fontsize=plot_config.legend_fontsize, ncol=plot_config.legend_filter_ncol, frameon=False)

        # --------------------------------------------------
        # Legend mid panel (Photometric data)
        # --------------------------------------------------
        if plot_config.label_data:
            plot_config.legend_data_ncol = (self.nb_filters + 6) // 7
            ax.legend(fontsize=plot_config.legend_fontsize, ncol=plot_config.legend_data_ncol, frameon=False)

        # --------------------------------------------------
        # Axis labels
        # --------------------------------------------------
        ax.set_xlabel(f"Wavelength ({getattr(self, 'unit', '')})")
        ax.set_ylabel("Flux")

        return fig, ax, ax_filt

    def _restricted_observation(self, windows: str | None = None, print_logger: bool = True) -> "PhotometryObservation":
        '''
        Restrict the observation to wavelength windows.

        Parameters
        ----------
        windows : str
            Windows in the format 'wmin1,wmax1 / wmin2,wmax2 / ...'
        print_logger : bool
            Whether to print logger

        Returns
        -------
        PhotometryObservation
            Restricted observation

        Authors
        -------
        Allan Denis
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

        for name, value in zip(['_wave', '_flux', '_err', '_Filter'], [self.wave, self.flux, self.err, self.Filter]):
            if value is not None:
                setattr(restricted, name, value[ind])

        if print_logger:
            self.logger.info(f'    Wavelength of former Observation: {self.wavelength_range}. Wavelength of restricted obervation: {restricted.wavelength_range}')

        return restricted