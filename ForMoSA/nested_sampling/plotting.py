import corner
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.axes._axes import Axes
import matplotlib.patheffects as path_effects
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path as MplPath
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from ForMoSA.core.config import PLOTS_CONFIG, MAIN_PLOT
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.loggings import setup_logging
from ForMoSA.transform.observed import ObservedModel
from ForMoSA.nested_sampling.results import NSResults
from ForMoSA.observation.observation_set import ObservationSet

# Cache so we don't re-register the projection on every plot call
_REGISTERED_RADARS: set[int] = set()


def _radar_polygon_factory(num_vars: int) -> tuple[np.ndarray, str]:
    """
    Register (once) a polygon-frame radar projection for `num_vars` spokes.

    Authors: Bhavesh Rajpoot (Adapted from matplotlib's radar chart gallery example:
    https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html)
    """
    name  = f'_formosa_radar_{num_vars}'
    theta = np.linspace(0.0, 2.0 * np.pi, num_vars, endpoint=False)

    if num_vars in _REGISTERED_RADARS:
        return theta, name
    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return MplPath(self.transform(path.vertices), path.codes)
    class RadarAxes(PolarAxes):
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')   # first axis at top

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                x, y = line.get_data()
                if x[0] != x[-1]:
                    line.set_data(np.append(x, x[0]), np.append(y, y[0]))

        def _gen_axes_patch(self):
            return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor='none')

        def _gen_axes_spines(self):
            spine = Spine(axes=self, spine_type='circle',
                          path=MplPath.unit_regular_polygon(num_vars))
            spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
            return {'polar': spine}

    RadarAxes.name = name
    register_projection(RadarAxes)
    _REGISTERED_RADARS.add(num_vars)
    return theta, name

class Plotting(object):
    '''
    Class of visualisation of the results of the nested sampling.

    Parameters
    ----------
    results : NSResults
        Instance of class NSResults
    logger : Logger
        Logger used
    log_level : str
        Level of the Logger

    Notes
    -----
    Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
    '''

    def __init__(self, results: NSResults, logger: logging.Logger, log_level: str = 'INFO') -> None:

        self._logger = logger if logger is not None else setup_logging(log_level,  name='Plotting')
        self._ns_results = results

        if not isinstance(results, NSResults):
            raise ForMoSAError(f'<Wrong type for results: {type(results)}. Expected a NSResults>', self.logger)

    # =================
    # Representation
    # =================

    def __repr__(self):
        return '<Plotting>'

    # =================
    # Properties
    # =================

    @property
    def logger(self) -> logging.Logger:
        """Logger."""
        return self._logger

    @property
    def ns_results(self) -> NSResults:
        """Instance of classe NSResults."""
        return self._ns_results

    # =================
    # Methods
    # =================

    def plot_corner(self) -> Figure:
        '''
        Corner plot the posterior samples from the nested sampling results.

        Parameters
        ----------
        config : CornerPlotConfig
            Instance of class CornerPlotConfig

        Returns
        -------
        matplotlib.figure.Figure
            Figure containin corner plots.

        Notes
        -----
        Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        self._logger.info('    Plotting Corner plot')

        samples, weights = self.ns_results.samples[self.ns_results.burn_in:], self.ns_results.weights[self.ns_results.burn_in:]

        # Get config for Corner plot
        config = PLOTS_CONFIG.CornerPlot

        # Get corner arguments from the config
        corner_kwargs = config.to_dict
        corner_kwargs['labels'] = self.ns_results.free_parameters
        corner_kwargs['weights'] = weights
        corner_kwargs['range'] = [0.99999 for i in self.ns_results.free_parameters]

        # Create the figure
        fig = corner.corner(samples, **corner_kwargs)

        return fig

    def plot_chains(self) -> tuple[Figure, Axes]:
        '''
        Plot the chains of the samples results.

        Parameters
        ----------

        Returns:
        --------
        tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
            Tuple containing Figure and Ax objects

        Notes
        -----
        Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        self._logger.info('    Plotting posterior chains for each parameter.')

        samples, weights = self.ns_results.samples, self.ns_results.weights

        samples = self.ns_results.samples
        param_best_values = list(self.ns_results.median_parameters.values())

        n_params = samples.shape[1]
        n_rows = (n_params + 1) // 2

        # Get config for chains plot
        config = PLOTS_CONFIG.ChainsPlot

        fig, axs = plt.subplots(n_rows, 2, figsize=config.figsize)
        axs = axs.flatten()

        for idx in range(n_params):
            ax = axs[idx]
            param_name = self.ns_results.free_parameters[idx]
            ax.plot(samples[:, idx], color=config.color_chains, alpha=config.alpha_chains)
            ax.set_ylabel(param_name)
            ax.axvline(self.ns_results.burn_in, linestyle=config.linestyle_burn_in, color=config.color_plot_burn_in)
            ax.text(x = config.text_burn_in[0], y = config.text_burn_in[1], s='burn in', color=config.color_text_burn_in, transform=ax.transAxes, fontsize=config.fontsize_burn_in)

            if config.show_weights:
                ax_w = ax.twinx()
                ax_w.plot(weights, config.color_plot_weights, alpha=config.alpha_weights)
                ax_w.set_yticks([])
                ax_w.text(x=config.text_weights[0], y=config.text_weights[1], s='weights', color=config.color_text_weights, transform=ax_w.transAxes, fontsize=config.fontsize_weights)

            if config.plot_best_value:
                ax.axhline(param_best_values[idx], color=config.color_best_value, linestyle=config.linestyle_best_value)

        for idx in range(n_params, len(axs)):
            fig.delaxes(axs[idx])

        return fig, axs[:n_params]

    def plot_radars(self) -> tuple[Figure, Axes]:
        '''
        Plot spider plot of the samples results.

        Parameters
        ----------

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]

        Notes
        -----
        Authors: Bhavesh Rajpoot (adapted from Paulina Palma-Bifani, Matthieu Ravet, Allan Denis)
        '''

        self._logger.info('    Plotting spider plot of the chains')

        samples = self.ns_results.samples[self.ns_results.burn_in:]
        weights = self.ns_results.weights[self.ns_results.burn_in:]
        params  = self.ns_results.free_parameters
        N       = len(params)

        config = PLOTS_CONFIG.RadarPlot

        # Step-size registry for known parameters 
        _STEPS: dict[str, float] = {
            'rv':      5.0,    # km/s
            'vsini':   10.0,   # km/s
            'd':       1.0,    # pc
            'r':       1.0,    # Rjup
            'teff':    100.0,  # K
            'logg':    0.5,    # dex
            'feh':     0.5,
            '[m/h]':   0.5,
            'co':      0.1,
        }

        def _nice(dr: float) -> float:
            if dr <= 0.0:
                return 1.0
            raw = dr / 5.0
            mag = 10.0 ** np.floor(np.log10(abs(raw)))
            r   = raw / mag
            if   r < 1.5: return 1.0 * mag
            elif r < 3.5: return 2.0 * mag
            elif r < 7.5: return 5.0 * mag
            else:         return 10.0 * mag

        def _step(name: str, dr: float) -> float:
            return _STEPS.get(name.lower().strip(), _nice(dr))

        def _snap(lo: float, hi: float, s: float) -> tuple[float, float]:
            return np.floor(lo / s) * s, np.ceil(hi / s) * s

        def _norm(v: float, lo: float, hi: float) -> float:
            return (v - lo) / (hi - lo) if hi != lo else 0.0

        def _fmt(v: float) -> str:
            if   abs(v) >= 1000: return f'{v:.0f}'
            elif abs(v) >= 10:   return f'{v:.1f}'
            else:                return f'{v:.2f}'

        # Weighted quantiles 
        q_lo, q_med, q_hi = [], [], []
        for i in range(samples.shape[1]):
            q_lo.append( self.ns_results._weighted_quantile(samples[:, i], weights, config.quantiles[0]))
            q_med.append(self.ns_results._weighted_quantile(samples[:, i], weights, 0.5))
            q_hi.append( self.ns_results._weighted_quantile(samples[:, i], weights, config.quantiles[1]))
        q_lo, q_med, q_hi = np.asarray(q_lo), np.asarray(q_med), np.asarray(q_hi)

        # Per-axis snapped bounds: based on CI width, not sample range 
        # This is the key change that removes the large empty space around
        # tight posteriors.  K_MARGIN = 3 puts the median at ~50% radius and
        # the 1-sigma band at ~33% to ~66% radius before snapping.
        K_MARGIN = 3.0
        ax_lo, ax_hi, step_sizes = [], [], []
        for i, p in enumerate(params):
            width = q_hi[i] - q_lo[i]
            if width <= 0:
                width = max(float(np.std(samples[:, i])), 1e-6)
            raw_lo = q_med[i] - K_MARGIN * width
            raw_hi = q_med[i] + K_MARGIN * width
            s = _step(p, raw_hi - raw_lo)
            lo, hi = _snap(raw_lo, raw_hi, s)
            ax_lo.append(lo); ax_hi.append(hi); step_sizes.append(s)

        # Normalize quantiles to [0, 1]; clip prevents excursions outside the ring
        med_n = np.clip([_norm(q_med[i], ax_lo[i], ax_hi[i]) for i in range(N)], 0.0, 1.0)
        lo_n  = np.clip([_norm(q_lo[i],  ax_lo[i], ax_hi[i]) for i in range(N)], 0.0, 1.0)
        hi_n  = np.clip([_norm(q_hi[i],  ax_lo[i], ax_hi[i]) for i in range(N)], 0.0, 1.0)

        # Polygon-frame polar projection 
        theta, proj = _radar_polygon_factory(N)

        fig = plt.figure(figsize=config.figsize)
        ax  = fig.add_subplot(projection=proj)

        # Grid rings & limits
        N_RINGS     = 4
        ring_levels = np.linspace(0.0, 1.0, N_RINGS + 1)[1:]   # [0.25, 0.5, 0.75, 1.0]
        ax.set_rgrids(ring_levels, labels=[''] * len(ring_levels))   # hide default radial labels
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, color='gray', linewidth=0.8, alpha=0.5, linestyle='--')
        ax.spines['polar'].set_color('#808183')
        ax.spines['polar'].set_linewidth(1.2)

        # Uncertainty band 
        ax.fill_between(theta, lo_n, hi_n,  color=config.color_uncertainty, alpha=config.alpha_fill, zorder=2)

        # Median polygon + markers 
        ax.plot(theta, med_n, color=config.color_radar, linewidth=2.0, zorder=3)
        for k in range(N):
            ax.scatter(theta[k], med_n[k], color='white', s=config.size_quantiles + 40,
                    zorder=4, edgecolors='none')
            ax.scatter(theta[k], med_n[k], color=config.color_quantiles, s=config.size_quantiles,
                    zorder=5, edgecolors='white', linewidths=config.lw_quantiles)

        # Parameter name labels at spoke tips (native matplotlib) 
        ax.set_thetagrids(np.degrees(theta), labels=params, fontsize=config.fontsize_names)
        ax.tick_params(axis='x', pad=50, colors='#24292E')

        # Value annotations: between polygon and parameter name 
        # Fixed radius along each spoke — independent of the data value, so
        # the labels never crowd the centre or escape the outer ring.
        VAL_R = 1.12
        for k in range(N):
            dlo = q_med[k] - q_lo[k]
            dhi = q_hi[k]  - q_med[k]
            lbl = f'${_fmt(q_med[k])}_{{-{_fmt(dlo)}}}^{{+{_fmt(dhi)}}}$'
            ax.text(theta[k], VAL_R, lbl,
                    ha='center', va='center',
                    fontsize=config.fontsize_ticks, fontweight='600',
                    color=config.color_radar, zorder=10, clip_on=False,
                    bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white', edgecolor='none', alpha=0.85))

        # Per-spoke tick labels: ONLY at grid-ring radii 
        # Fixes the original "label every step" overflow by reusing the same
        # 4 radii used for the visual grid.
        if getattr(config, 'show_ticks', False):
            for k in range(N):
                for rl in ring_levels[:-1]:   # skip outermost (overlaps with VAL_R)
                    actual = ax_lo[k] + rl * (ax_hi[k] - ax_lo[k])
                    ax.text(theta[k], rl, _fmt(actual),
                            ha='center', va='center',
                            fontsize=max(config.fontsize_ticks - 2, 11),
                            color=config.color_ticks, zorder=6, clip_on=False,
                            bbox=dict(boxstyle='round,pad=0.15',
                                    facecolor='white', edgecolor='none', alpha=0.5))

        fig.tight_layout()
        return fig, ax

    def plot_fit(self, observations: ObservationSet, best_fit: list[ObservedModel], plot_native_model: bool = False, native_model: ObservedModel | None = None) -> tuple[Figure, Axes, Axes, Axes, Axes]:
        '''
        Plot best fit

        Parameters
        ----------
        observations : ObservationSet
            Instance of class ObservationSet
        best_fit : list[ObservedModel]
            List of instances of class ObservedModel corresponding to the best-fit model for each observation
        plot_native_model : bool
            Whether to plot the native model
        native_model : ObservedModel
            As instance of ObservedModel

        Returns
        -------
        tuple[Figure, Axes, Axes, Axes, Axes]
            Figure and ax objects

        Notes
        -----
        Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        self._logger.info('    Plotting best fit and residuals')

        # Initial checks

        if not isinstance(best_fit, list) or len(best_fit) != observations.n_observations:
            raise ForMoSAError(f'best_fit must be a list of length {observations.n_observations}', self.logger)

        if plot_native_model is True:
            if not isinstance(native_model, ObservedModel):
                raise ForMoSAError(f'If you want to plot the native model, native_model must be an instance of ObservedModel. Got {type(native_model)}')

        # Get config for best fit
        config = PLOTS_CONFIG.BestFitPlot
        main_config = MAIN_PLOT

        # obs_set_transformed = ObservationSet(self.logger)

        # for i, obs in enumerate(observations.observations):
        #     # Create a copy of the observations to optionally remove component estimated by high-contrast module
        #     obs_transformed = copy.deepcopy(obs)
        #     obs_transformed._flux -= best_fit[i].component

        #     obs_set_transformed.add_observation(obs_transformed)

        # Reserve top rows for filter axis only when photometry is present
        ax_row_start = 2 if observations.has_photometry else 0

        fig = plt.figure(figsize=main_config.figsize)
        gs = gridspec.GridSpec(9, 11)

        # Main axis for observations + best-fit
        ax = fig.add_subplot(gs[ax_row_start:7, 0:10])

        # Axis for photometric filters
        ax_filt = None
        if observations.has_photometry:
            ax_filt = fig.add_subplot(gs[0:2, 0:10], sharex=ax)

        # Residuals and histogram axes
        axr = fig.add_subplot(gs[7:9, 0:10], sharex=ax)
        axr2 = fig.add_subplot(gs[7:9, 10:11], sharey=axr)

        # Plot native model if required
        if plot_native_model:
            ax.plot(native_model.wave, native_model.flux,
                    color=config.color_fit, linewidth=config.linewidth,
                    zorder=config.zorder, label='Best fit native model')

        # concatenate all residuals first to compute a global standard deviation for normalization,
        # which is crucial for a consistent residuals plot across different observations
        all_residuals = []

        for i, obs in enumerate(observations.observations):
            res = best_fit[i].residuals(obs.flux)
            all_residuals.append(res)

        all_residuals = np.concatenate(all_residuals)
        global_std = np.std(all_residuals)

        # Plot observations
        # obs_set_transformed.plot_all(fig=fig, ax=ax, ax_filt=ax_filt)
        observations.plot_all(fig=fig, ax=ax, ax_filt=ax_filt)

        # Plot best-fit and residuals
        for i, obs in enumerate(observations.observations):
            # Compute residuals and normalize by global std
            res_norm = best_fit[i].residuals(obs.flux) / global_std

            # Plot best-fit and residuals for photometric data
            if obs.is_photometric:
                if not plot_native_model: # For photometric data, we only plot the best-fit as scatter points
                    ax.scatter(best_fit[i].wave,
                               best_fit[i].total_flux, #best_fit[i].flux,
                               marker='o', c = config.color_fit, zorder=config.zorder, label='Best fit')

                # Plot residuals as scatter points for photometric data
                axr.scatter(obs.wave, res_norm, c = config.color_residuals, marker='o')

            # Plot best-fit and residuals for spectroscopic data
            else:
                if not plot_native_model: # For spectroscopic data, we plot the best-fit as a line
                    ax.plot(best_fit[i].wave,
                            best_fit[i].total_flux, #best_fit[i].flux,
                            color=config.color_fit, linewidth=config.linewidth, zorder=config.zorder, label='Best fit')

                # Plot residuals as a line for spectroscopic data
                axr.plot(obs.wave, res_norm, c=config.color_residuals, linewidth=config.linewidth)

            axr2.hist(res_norm, orientation='horizontal', bins=60, color=config.color_residuals, alpha=0.8, density=True)

        axr.set_xlabel(r'Wavelength ($\mu$m)')
        ax.set_xlabel(None)
        axr.set_ylabel(r'Residuals ($\sigma$)')
        axr.axhline(y=0, linestyle='--', color='grey')

        # Minor ticks
        if main_config.minor_ticks:
            axr.xaxis.set_minor_locator(AutoMinorLocator(main_config.nb_minor_ticks))
            axr.yaxis.set_minor_locator(AutoMinorLocator(main_config.nb_minor_ticks))

        # Remove axis for axr2
        axr2.axis('off')

        # Re-render the main legend so the 'Best fit' line is included
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles=handles, labels=labels, frameon=False, loc='upper right', fontsize=MAIN_PLOT.legend_fontsize)

        fig.tight_layout()

        return fig, ax, ax_filt, axr, axr2

    def plot_ccf(self, rv_grid: np.ndarray, ccf: np.ndarray, acf: np.ndarray, ccf_star: np.ndarray | None = None, title: str = None) -> tuple[Figure, Axes]:
        '''
        Plot the Cross-Correlation Function (CCF).

        Parameters
        ----------
        rv_grid : np.ndarray
            Grid of radial velocity values (in km/s)
        ccf : np.ndarray
            Corresponding ccf (cross-correlation) values
        acf : np.ndarray
            acf (aut-correlation) values
        ccf_star : np.ndarray
            ccf values with star speckles

        Returns
        -------
        tuple[Figure, Axes]
            Figure and Axes objects

        Notes
        -----
        Authors: Bhavesh Rajpoot and Allan Denis
        '''

        self._logger.info('    Plotting CCF')

        # Find best RV
        best_idx = np.unravel_index(np.argmax(ccf), ccf.shape)
        rv_peak = rv_grid[best_idx[0]]

        # plot_ccf
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rv_grid, ccf, label='CCF', color='blue')
        ax.plot(rv_grid + rv_peak, acf, label='ACF', color='orange', linestyle='--')

        if ccf_star is not None and np.any(ccf_star != 0):
            ax.plot(rv_grid, ccf_star, label='Star CCF', color='red', alpha=0.5)

        ax.axvline(rv_peak, color='grey', linestyle=':', label=f'RV = {rv_peak:.1f} km/s')
        ax.set_xlabel('RV (km/s)')
        ax.set_ylabel('CCF (SNR)')

        if title is not None:
            ax.set_title(f'CCF - {title}')

        ax.legend()

        return fig, ax

    def plot_rv_vsini_map(self, rv_grid: np.ndarray, vsini_grid: np.ndarray, logL_map: np.ndarray, title: str = None) -> tuple[list[Figure], list[Axes]]:
        '''
        Plot the RV vs v.sin(i) loglikelihood map.

        Parameters
        ----------
        rv_grid : np.ndarray
            Grid of radial velocity values (in km/s)
        ccf : np.ndarray
            Corresponding ccf (cross-correlation) values
        acf : np.ndarray
            acf (aut-correlation) values
        ccf_star : np.ndarray
            ccf values with star speckles

        Returns
        -------
        tuple[Figure, Axes]
            Figure and Axes objects

        Notes
        -----
        Authors: Bhavesh Rajpoot (adapted from Allan Denis)
        '''

        self._logger.info('    Computing RV-vsini map')

        # Find best RV and vsini
        best_idx = np.unravel_index(np.argmax(logL_map), logL_map.shape)
        best_vsini = vsini_grid[best_idx[0]]
        best_rv = rv_grid[best_idx[1]]

        # plot rv/vsini map
        fig, ax = plt.subplots(figsize=(8, 6))
        extent = [rv_grid[0], rv_grid[-1], vsini_grid[0], vsini_grid[-1]]
        im = ax.imshow(logL_map, aspect='auto', origin='lower', extent=extent, cmap='viridis')
        ax.scatter(best_rv, best_vsini, marker='x', color='red', s=100, label=f'Best: RV={best_rv:.1f}, vsini={best_vsini:.1f}')
        ax.set_xlabel('RV (km/s)')
        ax.set_ylabel(r'v.sin(i) (km/s)')

        if title is not None:
            ax.set_title(f'RV - v.sin(i) map - {title}')

        ax.legend()
        fig.colorbar(im, ax=ax, label='log L')

        return fig, ax