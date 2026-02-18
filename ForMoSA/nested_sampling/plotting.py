import corner
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.axes._axes import Axes

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.loggings import setup_logging
from ForMoSA.transform.observed import ObservedModel
from ForMoSA.nested_sampling.results import NSResults
from ForMoSA.observation.observation_set import ObservationSet
from ForMoSA.core.config import CORNER_PLOT, CHAINS_PLOT, RADAR_PLOT
from ForMoSA.core.config import CornerPlotConfig, ChainsPlotConfig, RadarPlotConfig


class Plotting(object):
    '''
    Class of visualisation of the results of the nested sampling.

    Parameters
    ----------
    results      (NSResults): Instance of class NSResults
    logger          (Logger): Logger used
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

    def plot_corner(self, config: CornerPlotConfig = CORNER_PLOT) -> Figure:
        '''
        Corner plot the posterior samples from the nested sampling results.

        Parameters
        ----------
        config (CornerPlotConfig): Instance of class CornerPlotConfig

        Returns
        -------
        matplotlib.figure.Figure: Figure containin corner plots.

        Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        self._logger.info('ForMoSA - Corner plot')

        samples, weights = self.ns_results.samples[self.ns_results.burn_in:], self.ns_results.weights[self.ns_results.burn_in:]

        # Get corner arguments from the config
        corner_kwargs = config.to_dict
        corner_kwargs['labels'] = self.ns_results.free_parameters
        corner_kwargs['weights'] = weights
        corner_kwargs['range'] = [0.99 for i in self.ns_results.free_parameters]

        # Create the figure
        fig = corner.corner(samples, **corner_kwargs)

        return fig

    def plot_chains(self, config: ChainsPlotConfig = CHAINS_PLOT) -> tuple[Figure, Axes]:
        '''
        Plot the chains of the samples results.

        Parameters
        ----------

        Returns:
        --------
        tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]: Tuple containing Figure and Ax objects

        Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        self._logger.info(' Plotting posterior chains for each parameter.')

        samples, weights = self.ns_results.samples, self.ns_results.weights

        samples = self.ns_results.samples
        param_best_values = list(self.ns_results.median_parameters.values())

        n_params = samples.shape[1]
        n_rows = (n_params + 1) // 2
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

    def plot_radars(self, config: RadarPlotConfig = RADAR_PLOT) -> tuple[Figure, Axes]:
        '''
        Radar plot the samples.

        Parameters
        ----------
        config (RadarPlotConfig): Instance of class RadarPlotConfig


        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]: Tuple containing Figure and Ax objects

        Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        self._logger.info('    Radar plot of the chains')

        samples, weights = self.ns_results.samples[self.ns_results.burn_in:], self.ns_results.weights[self.ns_results.burn_in:]

        samples = self.ns_results.samples

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
        observations       (ObservationSet): Instance of class ObservationSet
        best_fit           (list[ObservedModel]): List of instances of class ObservedModel corresponding to the best-fit model for each observation
        figsize            (tuple[float, float]): Size of the figure
        plot_native_model                 (bool): Whether to plot the native model
        native_model             (ObservedModel): As instance of ObservedModel


        Returns
        -------
        tuple[Figure, Axes, Axes, Axes, Axes]: Figure and ax objects

        Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        self._logger.info('ForMoSA - Best fit and residuals plot')

        # Initial checks

        if not isinstance(best_fit, list) or len(best_fit) != observations.n_observations:
            raise ForMoSAError(f'best_fit must be a list with {observations.n_observations}', self.logger)

        if plot_native_model is True:
            if not isinstance(native_model, ObservedModel):
                raise ForMoSAError(f'If you want to plot the native model, native_model must be an instance of ObservedModel. Got {type(native_model)}')

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
            ax.plot(native_model.wave, native_model.flux, color='black', linewidth=2)

        # Plot all observations and filters
        observations.plot_all(fig=fig, ax=ax, ax_filt=ax_filt)

        # Plot best-fit and residuals
        for i, obs in enumerate(observations.observations):
            res, std = best_fit[i].residuals(obs.flux), best_fit[i].std_residuals(obs.flux)

            if obs.is_photometric:
                if not plot_native_model:
                    ax.scatter(best_fit[i].wave,  best_fit[i].flux, marker='o')
                axr.scatter(obs.wave, res / std, c = 'black', marker='o')

            else:
                if not plot_native_model:
                    ax.plot(best_fit[i].wave, best_fit[i].flux, color='black', linewidth=2)  # Best-fit
                axr.plot(obs.wave, res / std, c='black')               # Residuals

            axr2.hist(res/std, orientation='horizontal', bins=100, color='black', alpha=0.8, density=True)

        axr.set_xlabel(r'Wavelength ($\mu$m)')
        axr.set_ylabel(r'Residuals ($\sigma$)')
        axr.axhline(y=0, linestyle='--', color = 'lightgrey')
        axr2.axis('off')

        # Rescale y axis with a power of 10
        ymin, ymax = ax.get_ylim()
        ymax_abs = max(abs(ymin), abs(ymax))
        exponent = int(np.floor(np.log10(ymax_abs)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{y/10**exponent:.1f}"))
        ax.set_ylabel(rf'Flux ($10^{{{exponent}}}$  W.m$^{{-2}}$.$\mu$m$^{{-1}}$)')

        return fig, ax, ax_filt, axr, axr2

