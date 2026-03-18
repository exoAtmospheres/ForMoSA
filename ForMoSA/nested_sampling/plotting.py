import copy
import corner
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.axes._axes import Axes

from ForMoSA.core.config import PLOTS_CONFIG
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.loggings import setup_logging
from ForMoSA.transform.observed import ObservedModel
from ForMoSA.nested_sampling.results import NSResults
from ForMoSA.observation.observation_set import ObservationSet


class Plotting(object):
    '''
    Class of visualisation of the results of the nested sampling.

    Parameters
    ----------
    results : NSResults
        Instance of class NSResults
    logger : Logger
        Logger used
    log_level         v(str): Level of the Logger

    Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
    '''

    def __init__(self, results: NSResults, logger: logging.Logger, log_level: str = 'INFO') -> None:

        self._logger = logger or setup_logging(log_level)
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
    def logger(self) -> logging.Logger:                  # Logger
        return self._logger

    @property
    def ns_results(self) -> NSResults:                   # Instance of classe NSResults
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
        Radar plot the samples.

        Parameters
        ----------
        config : RadarPlotConfig
            Instance of class RadarPlotConfig


        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
            Tuple containing Figure and Ax objects

        Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        self._logger.info('    Plotting radar plot of the chains')

        samples, weights = self.ns_results.samples[self.ns_results.burn_in:], self.ns_results.weights[self.ns_results.burn_in:]

        samples = self.ns_results.samples

        # Get config for radar plot
        config = PLOTS_CONFIG.RadarPlot

        # Compute quantiles for each parameter
        q_low, q_med, q_high = [], [], []
        for i in range(samples.shape[1]):
            q_low.append(self.ns_results._weighted_quantile(samples[:,i], weights, config.quantiles[0]))
            q_med.append(self.ns_results._weighted_quantile(samples[:,i], weights, 0.5))
            q_high.append(self.ns_results._weighted_quantile(samples[:,i], weights, config.quantiles[1]))

        q_low = np.array(q_low)
        q_med = np.array(q_med)
        q_high = np.array(q_high)

        # Use min/max of samples to simulate prior bounds
        prior_mins = np.min(samples, axis=0)
        prior_maxs = np.max(samples, axis=0)

        # Normalize based on "prior-like" range
        q_low_norm, q_med_norm, q_high_norm = [], [], []
        for i in range(len(q_low)):
            min_val = prior_mins[i]
            max_val = prior_maxs[i]
            range_val = max_val - min_val if max_val != min_val else 1.0
            q_low_norm.append((q_low[i] - min_val) / range_val)
            q_med_norm.append((q_med[i] - min_val) / range_val)
            q_high_norm.append((q_high[i] - min_val) / range_val)

        # Close the circle
        q_low_norm.append(q_low_norm[0])
        q_med_norm.append(q_med_norm[0])
        q_high_norm.append(q_high_norm[0])
        q_med = np.append(q_med, q_med[0])
        q_low = np.append(q_low, q_low[0])
        q_high = np.append(q_high, q_high[0])
        prior_mins = np.append(prior_mins, prior_mins[0])
        prior_maxs = np.append(prior_maxs, prior_maxs[0])

        # Angles for the radar plot
        angles = np.linspace(0, 2 * np.pi, len(self.ns_results.free_parameters), endpoint=False).tolist()
        angles.append(angles[0])

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        ax.fill_between(angles, q_low_norm, q_high_norm, color=config.color_radar, alpha=config.alpha_fill)
        ax.plot(angles, q_med_norm, color=config.color_radar, linewidth=2)
        ax.scatter(angles[:-1], q_med_norm[:-1], color=config.color_quantiles, s=config.size_quantiles)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.ns_results.free_parameters, fontsize=config.fontsize_names)
        ax.set_yticklabels([])
        # ax.set_title('Radar plot', size=14, pad=20)
        ax.grid(True)

        # Display ticks
        for i, angle in enumerate(angles[:-1]):
            min_val = prior_mins[i]
            max_val = prior_maxs[i]
            ticks = np.linspace(min_val, max_val, num=5)
            range_val = max_val - min_val if max_val != min_val else 1.0
            for i in range(len(ticks)-2):
                radius = (ticks[i+1] - min_val) / range_val
                ax.text(angle, radius, f'{ticks[i+1]:.2f}', ha='center', va='center', fontsize=config.fontisze_ticks, color=config.color_ticks)

        return fig, ax

    def plot_fit(self, observations: ObservationSet, best_fit: list[ObservedModel], figsize: tuple=(12,7), plot_native_model: bool = False, native_model: ObservedModel | None = None) -> tuple[Figure, Axes, Axes, Axes, Axes]:
        '''
        Plot best fit

        Parameters
        ----------
        observations : ObservationSet
            Instance of class ObservationSet
        best_fit : list[ObservedModel]
            List of instances of class ObservedModel corresponding to the best-fit model for each observation
        figsize : tuple[float, float]
            Size of the figure
        plot_native_model : bool
            Whether to plot the native model
        native_model : ObservedModel
            As instance of ObservedModel


        Returns
        -------
        tuple[Figure, Axes, Axes, Axes, Axes]
            Figure and ax objects

        Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        self._logger.info('    Plotting best fit and residuals')

        # Initial checks

        if not isinstance(best_fit, list) or len(best_fit) != observations.n_observations:
            raise ForMoSAError(f'best_fit must be a list with {observations.n_observations}', self.logger)

        if plot_native_model is True:
            if not isinstance(native_model, ObservedModel):
                raise ForMoSAError(f'If you want to plot the native model, native_model must be an instance of ObservedModel. Got {type(native_model)}')

        # Get config for best fit
        config = PLOTS_CONFIG.BestFitPlot

        obs_set_transformed = ObservationSet(self.logger)

        for i, obs in enumerate(observations.observations):
            # Create a copy of the observations to optionally remove component estimated by high-contrast module
            obs_transformed = copy.deepcopy(obs)
            obs_transformed._flux -= best_fit[i].component

            obs_set_transformed.add_observation(obs_transformed)

        fig = plt.figure(figsize=figsize)
        fig.clf()
        gs = gridspec.GridSpec(9, 11)

        # Main axis for observations + best-fit
        ax = fig.add_subplot(gs[2:7, 0:10])

        # Axis for photometric filters
        ax_filt = None
        if observations.has_photometry:
            ax_filt = fig.add_subplot(gs[0:2, 0:10], sharex=ax)

        # Residuals and histogram axes
        axr = fig.add_subplot(gs[7:9, 0:10], sharex=ax)
        axr2 = fig.add_subplot(gs[7:9, 10:11], sharey=axr)

        # Plot native model if required
        if plot_native_model:
            ax.plot(native_model.wave, native_model.flux, color=config.color, linewidth=config.linewidth, zorder=config.zorder)

        # concatenate all residuals first
        all_residuals = []

        for i, obs in enumerate(observations.observations):
            res = best_fit[i].residuals(obs.flux)
            all_residuals.append(res)

        all_residuals = np.concatenate(all_residuals)
        global_std = np.std(all_residuals)

        # Plot observations
        obs_set_transformed.plot_all(fig=fig, ax=ax, ax_filt=ax_filt)

        # Plot best-fit and residuals
        for i, obs in enumerate(observations.observations):
            res = best_fit[i].residuals(obs.flux)

            if obs.is_photometric:
                if not plot_native_model:
                    ax.scatter(best_fit[i].wave,  best_fit[i].flux, marker='o', c = config.color, zorder=config.zorder)
                axr.scatter(obs.wave, res / global_std, c = config.color, marker='o')

            else:
                if not plot_native_model:
                    ax.plot(best_fit[i].wave, best_fit[i].flux, color=config.color, linewidth=config.linewidth, zorder=config.zorder)  # Best-fit
                axr.plot(obs.wave, res / global_std, c=config.color)               # Residuals

            axr2.hist(res/global_std, orientation='horizontal', bins=100, color=config.color, alpha=0.8, density=True)

        axr.set_xlabel(r'Wavelength ($\mu$m)')
        axr.set_ylabel(r'Residuals ($\sigma$)')
        axr.axhline(y=0, linestyle='--', color = 'lightgrey')
        axr2.axis('off')

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

