import copy
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from ForMoSA.utils.spec import resolution_decreasing, continuum_estimate

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.observation.observation_base import Observation
from ForMoSA.core.config import SpectralPlotConfig, MAIN_PLOT
from ForMoSA.core.enums import ObservationType, ObservationKeys, WavelengthUnit


class SpectralObservation(Observation):
    '''
    Spectral observation class.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength array
    flux : np.ndarray
        Flux array
    err : np.ndarray
        Error array
    res : np.ndarray
        Spectral resolution array
    facility : str
        Facility name
    instrument : str
        Instrument name
    native_unit : WavelengthUnit
        Native unit of the wavelength
    name : str
        Name of the observation
    labels : list 
        Labels used for plotting the data
    cov : np.ndarray
        Covariance matrix
    transm : np.ndarray
        Transmission array (Atmo+inst)
    star_flux : np.ndarray
        Star flux array
    system : np.ndarray
        Systematics array
    logger : logging.Logger
        Logger
    log_level : str
        Level of the logger
    display_unit : WavelengthUnit
        Unit of the wavelength to display

    Notes
    -----
    Authors: Allan Denis
    '''

    def __init__(self, wave: np.ndarray, flux: np.ndarray, err: np.ndarray, res: np.ndarray, facility: str, instrument: str, native_unit: WavelengthUnit, name: str = 'unknown', labels: list | None = None, cov: np.ndarray | None = None, transm: np.ndarray | None = None, star_flux: np.ndarray | None = None, system: np.ndarray | None = None, logger: logging.Logger | None = None, log_level: str = 'INFO', display_unit: WavelengthUnit = WavelengthUnit.MICROMETER) -> None:

        # Inherit from Observation class
        super().__init__(wave=wave, flux=flux, err=err, facility=facility, instrument=instrument, name=name, labels=labels, native_unit=native_unit, logger=logger, log_level=log_level, display_unit=display_unit, plot_config=SpectralPlotConfig())

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

        self._clean_nans()
        self._validate_spectral()

    # ==================================================
    # Representation
    # ==================================================

    def __repr__(self) -> str:
        return f' SpectralObservation : {self.name} - {self.n_points} points'

    def __format__(self) -> str:
        return self.__repr__()

    # ==================================================
    # Properties
    # ==================================================

    @property
    def ObsType(self) -> ObservationType:
        """Observation type."""
        return ObservationType.SPECTROSCOPIC.value

    @property
    def res(self) -> np.ndarray[float]:
        """Resolution."""
        return self._res

    @property
    def cov(self) -> np.ndarray[float] | None:
        """Covariance."""
        return self._cov

    @property
    def inv_cov(self) -> np.ndarray[float] | None:
        """Inverse of covariance."""
        return self._inv_cov

    @property
    def transm(self) -> np.ndarray[float] | None:
        """Transmission."""
        return self._transm

    @property
    def star_flux(self) -> np.ndarray[float] | None:
        """Stellar flux."""
        return self._star_flux

    @property
    def system(self) -> np.ndarray[float] | None:
        """Systematics."""
        return self._system

    @property
    def flux_cont(self) -> np.ndarray[float] | None:
        """Continuum of the flux."""
        return self._flux_cont

    @property
    def star_flux_cont(self) -> np.ndarray[float] | None:
        """Continuum of the star flux."""
        return self._star_flux_cont

    @property
    def wave_cont(self) -> str | None:
        """Wavelengths used for the continuum."""
        return self._wave_cont

    @property
    def res_cont(self) -> float | None:
        """Resolution used for the continuum."""
        return self._res_cont

    @property
    def hc_mode(self) -> bool:
        """Whether the observation is in high-contrast mode."""
        return self._star_flux is not None

    @property
    def max_resolution(self) -> float:
        """Maximum resolution."""
        return float(np.max(self._res))

    @property
    def min_resolution(self) -> float:
        """Minimum resolution."""
        return float(np.min(self._res))

    @property
    def to_dict(self) -> dict[str, np.ndarray]:
        """Dictionary representation of spectroscopic data."""
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
            for i in range(self.system.shape[1]):
                data[f'{ObservationKeys.SYSTEMATICS.canonical}{i}'] = self.system[:,i].tolist()

        if self.star_flux_cont is not None:
            data[ObservationKeys.STAR_FLUX_CONT.canonical] = self.star_flux_cont.tolist()

        if self.flux_cont is not None:
            data[ObservationKeys.FLUX_CONT.canonical] = self.flux_cont.tolist()

        return data

    @property
    def wavelength_range(self) -> tuple[float, float]:
        """Wavelength range of the observation."""
        return float(self.wave.min()), float(self.wave.max())

    @property
    def instrument_idxs(self) -> np.ndarray:
        """Indexes of occurence of new instruments."""
        idxs = np.array([0])
        last_ins = self.instrument[0]

        for idx, ins in enumerate(self.instrument):
            if ins != last_ins:
                idxs = np.append(idxs, idx)
                last_ins = ins

        idxs = np.append(idxs, len(self.instrument))
        return idxs

    @property
    def nb_instruments(self) -> int:
        """Number of instruments."""
        return len(self.instrument_idxs) - 1
    
    @property
    def default_labels(self) -> list[str]:
        """Default labels used for plotting."""
    
        labels = []
    
        for i in range(self.nb_instruments):
            idx = self.instrument_idxs[i]
            labels.append(f'{self.facility[idx]}/{self.instrument[idx]}')
    
        return labels

    # ==================================================
    # Methods
    # ==================================================

    def _validate_spectral(self) -> None:
        '''
        Do some checks on spectroscopic observations.

        Notes
        -----
        Authors: Allan Denis
        '''

        # Resolution
        if len(self._res) != self.n_points:
            raise ForMoSAError('res must have same length as wave', self.logger)

        if np.any(self.res < 0):
            raise ForMoSAError('Spectral resolution must be positive', self.logger)

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
            if len(self.system) != self.n_points:
                raise ForMoSAError('Systematics must have same length as wave', self.logger)

    def _clean_nans(self) -> None:
        '''
        Remove non-finite values from all observation vectors
        and adjust covariance matrix accordingly.

        Notes
        -----
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
        self._facility = self.facility[mask]
        self._instrument = self.instrument[mask]

        if self.transm is not None:
            self._transm = self.transm[mask]

        if self.star_flux is not None:
            self._star_flux = self.star_flux[mask]

        if self.system is not None:
            self._system = self.system[mask]

        if self.cov is not None:
            self._cov = self.cov[np.ix_(mask, mask)]
            
    def _normalize_labels(self, labels: str | list[str] | None) -> list[str] | None:
        '''
        Normalize plotting labels.
    
        Parameters
        ----------
        labels : str | list[str] | None
            Labels to normalize
    
        Returns
        -------
        list[str] | None
            Normalized labels
    
        Notes
        -----
        Authors: Allan Denis
        '''
    
        if labels is None:
            return None
    
        if isinstance(labels, str):
            labels = [labels]
    
        if len(labels) == 1 and self.nb_instruments > 1:
            labels = labels * self.nb_instruments
    
        if len(labels) != self.nb_instruments:
            raise ForMoSAError(
                f'Expected {self.nb_instruments} labels, got {len(labels)}',
                self.logger
            )
    
        return labels

    def _adapt_to_resolution(self, target_resolution: np.ndarray, wave_cont: str | None = None, res_cont: float | None = None) -> Observation:
        '''
        Adapt the spectral observation to the target resolution:
        - degrade spectral resolution
        - optionally estimate/remove continuum

        Parameters
        ----------
        target_resolution : np.ndarray
            Target spectral resolution array
        wave_cont : str
            Wavelengths used for the continuum ('window1 / window2 / window3 / ...' where windowi = wave{i}, wave{i+1})
        res_cont : float
            Resolution of the continuum

        Returns
        -------
        dict
            Dictionnary representation of the adapted observation

        Notes
        -----
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

        # Always propagate wave_cont so that HC-mode observations can
        # compute the model continuum later in _hc_modeling, even when
        # res_cont is 'NA' (i.e. no continuum subtraction is requested).
        if wave_cont is not None and wave_cont != 'NA':
            adapted_obs._wave_cont = wave_cont

        if (res_cont is not None and res_cont != 'NA'):
            if (wave_cont is None or wave_cont == 'NA'):
                self.logger.warning('Wave_cont is not defined for continuum estimation. Using the observation wavelength')
                wave_cont = str(self.wavelength_range[0]) + ',' + str(self.wavelength_range[1])
                adapted_obs._wave_cont = wave_cont

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

        self._logger.info(f'    Spectral observation {self.name} adapted to target resolution')

        return adapted_obs

    def plot_data(self, fig: Figure | None = None, ax: Axes | None = None, ax_filt: Axes | None = None, draw_legend: bool = True) -> tuple[Figure, Axes, Axes]:
        '''
        Plot spectroscopic data.

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

        Returns
        -------
        fig : matplotlib.figure.Figure
            Updated figure
        ax : matplotlib.axes._axes.Axes
            Updated ax
        ax_filt : matplotlib.axes._axes.Axes
            Non updated ax_filt

        Notes
        -----
        Authors: Allan Denis
        '''

        self.logger.info(f'      Plotting data for observation {self.name}')

        plot_config = self.plot_config
        main_plot_config = MAIN_PLOT

        # --------------------------------------------------
        # Figure / axes creation
        # --------------------------------------------------
        if ax is None:
            fig, ax = plt.subplots(figsize=main_plot_config.figsize)
        elif fig is None:
            fig = ax.figure

        # --------------------------------------------------
        # Plotting
        # --------------------------------------------------
        for i in range(self.nb_instruments):
            idx0 = self.instrument_idxs[i]
            idx1 = self.instrument_idxs[i + 1]

            label = self.labels[i]

            if plot_config.marker == 'None':

                ax.plot(
                    self.wave[idx0:idx1],
                    self.flux[idx0:idx1],
                    color=plot_config.color,
                    linewidth=plot_config.linewidth,
                    zorder=plot_config.zorder_data,
                    label=label
                )

            else:

                ax.scatter(
                    self.wave[idx0:idx1],
                    self.flux[idx0:idx1],
                    color=plot_config.color,
                    edgecolors=plot_config.edgecolor,
                    marker=plot_config.marker,
                    s=plot_config.markersize,
                    linewidths=plot_config.linewidth,
                    zorder=plot_config.zorder_data,
                    label=label
                )

                ax.errorbar(
                    self.wave[idx0:idx1],
                    self.flux[idx0:idx1],
                    yerr=self.err[idx0:idx1],
                    fmt=plot_config.errorbar_fmt,
                    ecolor=plot_config.color,
                    alpha=plot_config.errorbar_alpha,
                    capsize=plot_config.errorbar_capsize,
                    zorder=plot_config.zorder_error
                )

        # --------------------------------------------------
        # Legend (only once; suppressed when called from plot_all)
        # --------------------------------------------------
        if draw_legend and plot_config.label:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ncol = max(1, int(main_plot_config.legend_hc_ncol if self.hc_mode else main_plot_config.legend_ncol))
                ax.legend(fontsize=main_plot_config.legend_fontsize, ncol=ncol, frameon=False)

        # --------------------------------------------------
        # Axis labels
        # --------------------------------------------------
        ax.set_xlabel(f"Wavelength ({getattr(self, 'unit', '')})")
        ax.set_ylabel("Flux")

        return fig, ax, ax_filt

    def _restricted_observation(self, windows: str | None = None, print_logger: bool = True, extension: float = 0.0) -> "SpectralObservation":
        '''
        Restrict the observation to wavelength windows.

        Parameters
        ----------
        windows : str
            Windows in the format 'wmin1,wmax1 / wmin2,wmax2 / ...'
        print_logger : bool
            Whether to print logger
        extension : float
            Extension factor of the windows

        Returns
        -------
        SpectralObservation
            Restricted observation

        Notes
        -----
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
            self.logger.info(f'    Wavelength of former Observation: {self.wavelength_range}. Wavelength of restricted obervation: {restricted.wavelength_range}')

        return restricted