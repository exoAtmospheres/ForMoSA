import copy
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from ForMoSA.utils.spec import resolution_decreasing, continuum_estimate

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.observation.observation_base import Observation
from ForMoSA.core.config import SpectralPlotConfig, SPECTRAL_PLOT
from ForMoSA.core.enums import ObservationType, ObservationKeys, WavelengthUnit


class SpectralObservation(Observation):
    '''
    Spectral observation class.

    Parameters
    ----------
    wave                (np.ndarray): Wavelength array
    flux                (np.ndarray): Flux array
    err                 (np.ndarray): Error array
    res                 (np.ndarray): Spectral resolution array
    facility                   (str): Facility name
    instrument                 (str): Instrument name
    native_unit     (WavelengthUnit): Native unit of the wavelength
    cov                 (np.ndarray): Covariance matrix
    transm              (np.ndarray): Transmission array (Atmo+inst)
    star_flux           (np.ndarray): Star flux array
    system              (np.ndarray): Systematics array
    logger          (logging.Logger): Logger
    log_level                  (str): Level of the logger
    display_unit    (WavelengthUnit): Unit of the wavelength to display

    Authors: Allan Denis
    '''

    def __init__(self, wave: np.ndarray, flux: np.ndarray, err: np.ndarray, res: np.ndarray, facility: str, instrument: str, native_unit: WavelengthUnit, cov: np.ndarray | None = None, transm: np.ndarray | None = None, star_flux: np.ndarray | None = None, system: np.ndarray | None = None, logger: logging.Logger | None = None, log_level: str = 'INFO', display_unit: WavelengthUnit = WavelengthUnit.MICROMETER) -> None:

        # Inherit from Observation class
        super().__init__(wave=wave, flux=flux, err=err, facility=facility, instrument=instrument, native_unit=native_unit, logger=logger, log_level=log_level, display_unit=display_unit)

        # Spectral-specific attributes
        self._res = np.atleast_1d(np.asarray(res, dtype=float))

        self._cov = None if cov is None else np.atleast_2d(np.asarray(cov, dtype=float))
        self._transm = None if transm is None else np.atleast_1d(np.asarray(transm, dtype=float))
        self._star_flux = None if star_flux is None else np.atleast_2d(np.asarray(star_flux, dtype=float)).reshape(max(np.shape(star_flux)), -1)
        self._system = None if system is None else np.atleast_2d(np.asarray(system, dtype=float)).reshape(max(np.shape(system)), -1)
        self._inv_cov = None
        self._flux_cont = None
        self._star_flux_cont = None
        self._flux_cont = None
        self._star_flux_cont = None
        self._res_cont = None
        self._wave_cont = None

        self._validate_spectral()
        self._clean_nans()

    # ==================================================
    # Representation
    # ==================================================

    def __repr__(self) -> str:
        return f' SpectralObservation : {self.facility} - {self.instrument} - {self.n_points} points'

    def __format__(self) -> str:
        return self.__repr__()

    # ==================================================
    # Properties
    # ==================================================

    @property
    def ObsType(self) -> ObservationType:                       # Observation type
        return ObservationType.SPECTROSCOPIC.value

    @property
    def res(self) -> np.ndarray:                                # Resolution
        return self._res

    @property
    def cov(self) -> np.ndarray | None:                         # Covariance
        return self._cov

    @property
    def inv_cov(self) -> np.ndarray | None:                     # Inverse of covariance
        return self._inv_cov

    @property
    def transm(self) -> np.ndarray | None:                      # Transmission
        return self._transm

    @property
    def star_flux(self) -> np.ndarray | None:                    # Stellar flux
        return self._star_flux

    @property
    def system(self) -> np.ndarray | None:                      # Systematics
        return self._system

    @property
    def flux_cont(self) -> np.ndarray | None:                   # Continuum of the flux
        return self._flux_cont

    @property
    def star_flux_cont(self) -> np.ndarray | None:              # Continuum of the star flux
        return self._star_flux_cont

    @property
    def wave_cont(self) -> str | None:                          # Wavelengths used for the continuum
        return self._wave_cont

    @property
    def res_cont(self) -> float | None:                         # Resolution used for the continuum
        return self._res_cont

    @property
    def hc_mode(self) -> bool:                                  # Whether the observation is in high-contrast mode
        return self._star_flux is not None

    @property
    def max_resolution(self) -> float:                          # Maximum resolution
        return float(np.max(self._res))

    @property
    def min_resolution(self) -> float:                          # Minimum resolution
        return float(np.min(self._res))

    @property
    def to_dict(self) -> dict[str, np.ndarray]:                 # Dictionary representation of spectroscopic data
        data = {
            ObservationKeys.WAVELENGTH.canonical: self.wave.tolist(),
            ObservationKeys.FLUX.canonical: self.flux.tolist(),
            ObservationKeys.ERROR.canonical: self.err.tolist(),
            ObservationKeys.RESOLUTION.canonical: self.res.tolist(),
            ObservationKeys.FACILITY.canonical: self.facility,
            ObservationKeys.INSTRUMENT.canonical: self.instrument,
            ObservationKeys.WAVELENGTH_UNIT.canonical: str(WavelengthUnit[str(self.unit)].value),
            ObservationKeys.WAVE_CONT.canonical: self.wave_cont,
            ObservationKeys.RES_CONT.canonical: self.res_cont
        }

        if self.cov is not None and self.cov.size != 0:
            data[ObservationKeys.COVARIANCE.canonical] = self.cov.tolist()

        if self.transm is not None and self.transm.size != 0:
            data[ObservationKeys.TRANSMISSION.canonical] = self.transm.tolist()

        if self.star_flux is not None and self.star_flux.size != 0:
            for i in range(self.star_flux.shape[1]):
                data[f'{ObservationKeys.STAR_FLUX.canonical}{i}'] = self.star_flux[:,i].tolist()

        if self.system is not None and self.system.size != 0:
            for i in range(self.star_system.shape[1]):
                data[f'{ObservationKeys.SYSTEMATICS.canonical}{i}'] = self.system[:,i].tolist()

        if self.star_flux_cont is not None:
            data[ObservationKeys.STAR_FLUX_CONT.canonical] = self.star_flux_cont.tolist()

        if self.flux_cont is not None:
            data[ObservationKeys.FLUX_CONT.canonical] = self.flux_cont.tolist()

        return data

    @property
    def name(self) -> str:                                      # Name of the observation
        return f'{self.facility}_{self.instrument}'

    @property
    def wavelength_range(self) -> tuple[float, float]:          # Wavelength range of the observation
        return float(self.wave.min()), float(self.wave.max())

    # ==================================================
    # Methods
    # ==================================================

    def _validate_spectral(self) -> None:
        '''
        Do some checks on spectroscopic observations.

        Authors: Allan Denis
        '''

        # Resolution
        if len(self._res) != self.n_points:
            raise ForMoSAError('res must have same length as wave', self.logger)

        if np.any(self.res <= 0):
            raise ForMoSAError('Spectral resolution must be strictly positive', self.logger)

        # Covariance
        if self._cov is not None:
            n = self.n_points
            if self._cov.shape != (n, n):
                raise ForMoSAError(f'cov must have shape {n, n}', self.logger)
            if np.any(np.diag(self.cov) <= 0):
                raise ForMoSAError('Covariance must be strictly positive', self.logger)
            self._inv_cov = np.linalg.inv(self._cov)

        # Star flux
        if self.star_flux is not None:
            if len(self.star_flux) != self.n_points:
                raise ForMoSAError('star flux must have same length as wave', self.logger)

            # Covariance is not implemented yet with high-contrast observations
            if self.cov is not None:
                self.logger.warning('Covariance is not implemented yet with high-contrast observations. Not using it')
                self.cov = None

        # Transmission
        if self.transm is not None:
            if len(self.transm) != self.n_points:
                raise ForMoSAError('Transmission must have same length as wave', self.logger)

        # Systematics
        if self.system is not None:
            if self(self.system) != self.n_points:
                raise ForMoSAError('Systematics must have same length as wave', self.logger)

    def _clean_nans(self) -> None:
        '''
        Remove non-finite values from all observation vectors
        and adjust covariance matrix accordingly.

        Authors: Allan Denis
        '''

        # --------------------------------------------------
        # Start with mandatory 1D arrays
        # --------------------------------------------------
        mask = (np.isfinite(self.wave) & np.isfinite(self.flux) & np.isfinite(self.err) & np.isfinite(self.res))

        # --------------------------------------------------
        # Optional 1D arrays
        # --------------------------------------------------
        if self.transm is not None:
            mask &= np.isfinite(self.transm)

        # --------------------------------------------------
        # Optional 2D arrays (N, M)
        # --------------------------------------------------
        if self.star_flux is not None:
            mask &= np.all(np.isfinite(self.star_flux), axis=1)

        if self.system is not None:
            mask &= np.all(np.isfinite(self.system), axis=1)

        # --------------------------------------------------
        # Covariance matrix (N, N)
        # --------------------------------------------------
        if self.cov is not None:
            mask &= (np.all(np.isfinite(self.cov), axis=0) & np.all(np.isfinite(self.cov), axis=1))

        # --------------------------------------------------
        # Apply mask
        # --------------------------------------------------
        self._wave = self.wave[mask]
        self._flux = self.flux[mask]
        self._err = self.err[mask]
        self._res = self.res[mask]

        if self.transm is not None:
            self._transm = self.transm[mask]

        if self.star_flux is not None:
            self._star_flux = self.star_flux[mask]

        if self.system is not None:
            self._system = self.system[mask]

        if self.cov is not None:
            self._cov = self.cov[np.ix_(mask, mask)]

    def _adapt_to_resolution(self, target_resolution: np.ndarray, wave_cont: str | None = None, res_cont: float | None = None) -> Observation:
        '''
        Adapt the spectral observation to the target resolution:
        - degrade spectral resolution
        - optionally estimate/remove continuum

        Parameters
        ----------
        target_resolution   (np.ndarray): Target spectral resolution array
        wave_cont                  (str): Wavelengths used for the continuum ('window1 / window2 / window3 / ...' where windowi = wave{i}, wave{i+1})
        res_cont                 (float): Resolution of the continuum

        Returns
        -------
        dict: Dictionnary representation of the adapted observation

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        self.logger.info(f'      Target resolution for observation {self.name}: {target_resolution}')

        # Deep copy to avoid modifying the original observation
        adapted_obs = copy.deepcopy(self)

        # Limit resolution to original resolution
        target_resolution = np.minimum(self.res, target_resolution)

        # ========================
        # Resolution degrading
        # ========================
        adapted_obs._flux = resolution_decreasing(self.wave, self.flux, self.res, self.wave, target_resolution)

        if self.transm is not None:
            adapted_obs._transm = resolution_decreasing(self.wave, self.transm, self.res, self.wave, target_resolution)

        # ========================
        # High contrast components
        # ========================
        if self.hc_mode:
            if self.star_flux is not None:
                adapted_obs._star_flux = np.column_stack([resolution_decreasing(self.wave, self.star_flux[:, i], self.res, self.wave, target_resolution) for i in range(self.star_flux.shape[1])])

            if self.system is not None:
                adapted_obs._system = np.column_stack([resolution_decreasing(self.wave, self.system[:, i], self.res, self.wave, target_resolution) for i in range(self.system.shape[1])])

        # Updating resolution
        adapted_obs._res = target_resolution

        # ========================
        # Continuum handling
        # ========================

        if (res_cont is not None and res_cont != 'NA'):
            if (wave_cont is None or wave_cont == 'NA'):
                self.logger.warning('Wave_cont is not defined for continuum estimation. Using the observation wavelength')
                wave_cont = str(self.wavelength_range[0]) + ',' + str(self.wavelength_range[1])

            # ========================
            # Continuum
            # ========================

            self._logger.debug('Subtracting continuum from spectrum')
            adapted_obs._flux_cont = continuum_estimate(self.wave, adapted_obs.flux, adapted_obs.res, wave_cont, res_cont)

            if not self.hc_mode:
                adapted_obs._flux -= adapted_obs._flux_cont
            else:
                adapted_obs._star_flux_cont = continuum_estimate(self.wave, self.star_flux[:, self.star_flux.shape[1] // 2], adapted_obs.res, wave_cont, res_cont)

            adapted_obs._res_cont = res_cont
            adapted_obs._wave_cont = wave_cont

        self._logger.info(f'    Spectral observation {self.name} adapted to target resolution')

        return adapted_obs

    def plot_data(self, fig: Figure | None = None, ax: Axes | None = None, ax_filt: Axes | None = None, plot_config: SpectralPlotConfig = SPECTRAL_PLOT) -> tuple[Figure, Axes, Axes]:
        '''
        Plot spectroscopic data.

        Parameters
        ----------
        fig        (matplotlib.figure.Figure): Figure (used to overplot on an existing figure)
        ax       (matplotlib.axes._axes.Axes): Ax (used to overplot on an existing ax)
        ax_filt  (matplotlib.axes._axes.Axes): Ax used to overplot the transmission filter on an existing ax
        plot_config      (SpectralPlotConfig): Instance of class ObservationPlotConfig

        Returns
        -------
        fig        (matplotlib.figure.Figure): Updated figure
        ax       (matplotlib.axes._axes.Axes): Updated ax
        ax_filt  (matplotlib.axes._axes.Axes): Non updated ax_filt

        Authors: Allan Denis
        '''

        self.logger.info(f'      Plotting data for observation {self.name}')

        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=plot_config.figsize)
        elif fig is None:
            fig = ax.figure

        if plot_config.label:
            label = self.facility + '/' + self.instrument
        else:
            label = None

        # Plot data: either as line or scatter depending on marker
        if plot_config.marker == 'None' :
            ax.plot(self.wave, self.flux, color=plot_config.color, linewidth=plot_config.linewidth, zorder=plot_config.zorder_data, label = label)
        else:
            ax.scatter( self.wave, self.flux, color=plot_config.color, edgecolors=plot_config.edgecolor, marker=plot_config.marker, s=plot_config.markersize, linewidths=plot_config.linewidth, zorder=plot_config.zorder_data, label = label)
            # Plot error bars
            ax.errorbar(self.wave, self.flux, yerr=self.err, fmt=plot_config.errorbar_fmt, ecolor=plot_config.color, alpha=plot_config.errorbar_alpha, capsize=plot_config.errorbar_capsize, zorder=plot_config.zorder_error)

        # Legend
        if plot_config.label:
            ax.legend()

        # Axis labels
        ax.set_xlabel(f" Wavelength ({getattr(self, 'unit', '')})")
        ax.set_ylabel('Flux')

        return fig, ax, ax_filt

    def _restricted_observation(self, windows: str | None = None, print_logger: bool = True, extension: float = 0.0) -> "SpectralObservation":
        '''
        Restrict the observation to wavelength windows.

        Parameters
        ----------
        windows        (str): Windows in the format 'wmin1,wmax1 / wmin2,wmax2 / ...'
        print_logger  (bool): Whether to print logger
        extension    (float): Extension factor of the windows

        Returns
        -------
        SpectralObservation: Restricted observation

        Authors: Allan Denis
        '''

        # Dictionary of the observation
        restricted = copy.deepcopy(self)

        if windows is None:
            windows = f'{self.wave[0]}, {self.wave[-1]}'

        if print_logger:
            self.logger.debug(f'Restrict observation {self.name} onto wavelengths windows {windows}')

        ind = np.array([], dtype=int)
        for window in windows.split("/"):
            wmin, wmax = map(float, window.split(","))
            indices = np.where((self.wave >= wmin * (1 - extension)) & (self.wave <= wmax * (1 + extension)))[0]
            ind = np.concatenate((ind, indices))
        ind = np.unique(ind)

        for name, value in zip(['_wave', '_flux', '_err', '_res', '_star_flux', '_transm', '_system', '_flux_cont', '_star_flux_cont'], [self.wave, self.flux, self.err, self.res, self.star_flux, self.transm, self.system, self.flux_cont, self.star_flux_cont]):
            if value is not None:
                setattr(restricted, name, value[ind])

        if self.cov is not None:
            restricted._cov = self.cov[np.ix_(ind, ind)]
            restricted._inv_cov = self.inv_cov[np.ix_(ind, ind)]

        restricted._wave_cont = self.wave_cont
        restricted._res_cont = self.res_cont

        if print_logger:
            self.logger.info(f'Length of former Observation: {len(self.wave)}. Lengths of restricted obervation: {len(restricted.wave)}')

        return restricted