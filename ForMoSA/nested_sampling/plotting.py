import os
import corner
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from pathlib import Path
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.axes._axes import Axes

from ForMoSA.core.config import PLOTS_CONFIG
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.loggings import setup_logging
from ForMoSA.core.enums import ParameterKind
from ForMoSA.transform.observed import ObservedModel
from ForMoSA.nested_sampling.results import NSResults
from ForMoSA.grid.model_grid import ModelGrid
from ForMoSA.grid.subgrid_set import SubGridSet
from ForMoSA.observation.observation_set import ObservationSet
from ForMoSA.utils.spec import compute_ccf, vsini_fct


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

    def plot_corner(self) -> Figure:
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

        self._logger.info('    Plotting Corner plot')

        samples, weights = self.ns_results.samples[self.ns_results.burn_in:], self.ns_results.weights[self.ns_results.burn_in:]

        # Get config for Corner plot
        config = PLOTS_CONFIG.CornerPlot

        # Get corner arguments from the config
        corner_kwargs = config.to_dict
        corner_kwargs['labels'] = self.ns_results.free_parameters
        corner_kwargs['weights'] = weights
        corner_kwargs['range'] = [0.99 for i in self.ns_results.free_parameters]

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
        tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]: Tuple containing Figure and Ax objects

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
        config (RadarPlotConfig): Instance of class RadarPlotConfig


        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]: Tuple containing Figure and Ax objects

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

        self._logger.info('    Plotting best fit and residuals')

        # Initial checks

        if not isinstance(best_fit, list) or len(best_fit) != observations.n_observations:
            raise ForMoSAError(f'best_fit must be a list with {observations.n_observations}', self.logger)

        if plot_native_model is True:
            if not isinstance(native_model, ObservedModel):
                raise ForMoSAError(f'If you want to plot the native model, native_model must be an instance of ObservedModel. Got {type(native_model)}')

        # Get config for best fit plot
        config = PLOTS_CONFIG.BestFitPlot

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

        # Plot all observations and filters
        observations.plot_all(fig=fig, ax=ax, ax_filt=ax_filt)

        # concatenate all residuals first
        all_residuals = []

        for i, obs in enumerate(observations.observations):
            res = best_fit[i].residuals(obs.flux)
            all_residuals.append(res)

        all_residuals = np.concatenate(all_residuals)
        global_std = np.std(all_residuals)

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

            axr2.hist(res/global_std, orientation='horizontal', bins=100, color='black', alpha=0.8, density=True)

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

    # ==========================
    # CCF methods
    # ==========================

    def _get_native_model_template(self, grid: ModelGrid, ns: 'NestedSampling', observations: ObservationSet, rv_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, 'ObservedParameters']:
        '''
        Get the native model template interpolated at best-fit grid parameters.

        Parameters
        ----------
        grid          (ModelGrid): Parent model grid
        ns      (NestedSampling): NestedSampling instance
        observations (ObservationSet): Set of observations
        rv_grid       (np.ndarray): RV grid (used to compute wavelength margin)

        Returns
        -------
        tuple: (wav, flx, res, observed_params)

        Authors: Bhavesh Rajpoot (adapted from Allan Denis)
        '''

        best_params = list(self.ns_results.median_parameters.values())
        observed_params = ns._build_params_for_obs(best_params, 0)
        grid_params = observed_params.grid.values_by_name

        # Restrict parent grid to observation wavelength range with Doppler margin
        wrange = observations.wavelength_range
        c_km_s = 299792.458
        margin = max(np.abs(rv_grid)) / c_km_s + 0.05
        restricted = grid._restricted_grid(
            f'{wrange[0] * (1 - margin)},{wrange[1] * (1 + margin)}',
            print_logger=False
        )

        model = restricted._interpolate_between_gridpoints(grid_params, method=ns.interp_method)
        wav = model.coords['wavelength'].values
        flx = model['grid'].values
        res = model.attrs['res']

        if np.isscalar(res) or (isinstance(res, np.ndarray) and res.ndim == 0):
            res = np.full_like(wav, float(res))

        return wav, flx, res, observed_params

    def plot_ccf(self, observations: ObservationSet, grid: ModelGrid, subgrids: SubGridSet, ns: 'NestedSampling', rv_grid: np.ndarray, plot: bool = True, save_path: str | os.PathLike | None = None) -> dict:
        '''
        Compute and optionally plot the Cross-Correlation Function (CCF).

        Parameters
        ----------
        observations (ObservationSet): Set of observations
        grid              (ModelGrid): Parent model grid
        subgrids         (SubGridSet): Set of subgrids
        ns          (NestedSampling): NestedSampling instance
        rv_grid           (np.ndarray): Grid of radial velocity values (in km/s)
        plot                    (bool): Whether to display the plot
        save_path (str | os.PathLike | None): Path to save results (None to skip saving)

        Returns
        -------
        dict: Dictionary of CCF results keyed by observation name

        Authors: Bhavesh Rajpoot (adapted from Allan Denis)
        '''

        self._logger.info('    Computing CCF')

        wav_mod, flx_mod, res_mod, observed_params = self._get_native_model_template(grid, ns, observations, rv_grid)

        results = {}

        for index in range(observations.n_observations):
            obs = observations.observations[index]
            subgrid = subgrids.subgrids[index]

            if not obs.is_spectroscopic:
                continue

            wav_fit = ns.wave_fit[index]
            bounds = ns.bounds_lsq[index]

            star_flx = obs.star_flux if obs.star_flux is not None else np.array([])
            transm = obs.transm if obs.transm is not None else np.array([])
            system = obs.system if obs.system is not None else np.array([])

            ccf, acf, ccf_star, rv_peak, logL, ccf_raw = compute_ccf(
                wav_mod, flx_mod,
                obs.wave, obs.flux, obs.err,
                res_mod, obs.res,
                subgrid.res_cont, wav_fit,
                star_flx_obs_spectro=star_flx,
                transm_obs_spectro=transm,
                system_obs_spectro=system,
                rv_grid=rv_grid,
                rv_sini_map=False,
                bounds=bounds,
                normalize=True
            )

            file_tag = f'{obs.name}_{index}'

            results[file_tag] = {
                'rv_grid': rv_grid,
                'ccf': ccf,
                'acf': acf,
                'ccf_star': ccf_star,
                'rv_peak': rv_peak,
                'logL': logL
            }

            if save_path is not None:
                path = Path(save_path) / f'ccf_{file_tag}.npz'
                np.savez(path, **results[file_tag])
                self._logger.info(f'    CCF data saved to {path}')

            if plot:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(rv_grid, ccf, label='CCF', color='blue')
                ax.plot(rv_grid, acf, label='ACF', color='orange', linestyle='--')
                if ccf_star is not None and np.any(ccf_star != 0):
                    ax.plot(rv_grid, ccf_star, label='Star CCF', color='red', alpha=0.5)
                ax.axvline(rv_peak, color='grey', linestyle=':', label=f'RV = {rv_peak:.1f} km/s')
                ax.set_xlabel('RV (km/s)')
                ax.set_ylabel('CCF (SNR)')
                ax.set_title(f'CCF - {file_tag}')
                ax.legend()

                if save_path is not None:
                    fig_path = Path(save_path) / f'ccf_{file_tag}.pdf'
                    fig.savefig(fig_path)
                    self._logger.info(f'    CCF plot saved to {fig_path}')

        return results

    def plot_rv_vsini_map(self, observations: ObservationSet, grid: ModelGrid, subgrids: SubGridSet, ns: 'NestedSampling', rv_grid: np.ndarray, vsini_grid: np.ndarray, plot: bool = True, save_path: str | os.PathLike | None = None) -> dict:
        '''
        Compute and optionally plot the RV vs v.sin(i) loglikelihood map.

        Parameters
        ----------
        observations (ObservationSet): Set of observations
        grid              (ModelGrid): Parent model grid
        subgrids         (SubGridSet): Set of subgrids
        ns          (NestedSampling): NestedSampling instance
        rv_grid           (np.ndarray): Grid of radial velocity values (in km/s)
        vsini_grid        (np.ndarray): Grid of v.sin(i) values (in km/s)
        plot                    (bool): Whether to display the plot
        save_path (str | os.PathLike | None): Path to save results (None to skip saving)

        Returns
        -------
        dict: Dictionary of RV-vsini map results keyed by observation name

        Authors: Bhavesh Rajpoot (adapted from Allan Denis)
        '''

        self._logger.info('    Computing RV-vsini map')

        wav_mod, flx_mod, res_mod, observed_params = self._get_native_model_template(grid, ns, observations, rv_grid)

        # Get ld value and vsini function type from parameters
        if observed_params.has_kind(ParameterKind.LD):
            ld_value = observed_params.get_kind(ParameterKind.LD)
        else:
            ld_value = 0.6

        vsini_params = [p for p in ns.parameters.parameters if p.kind == ParameterKind.VSINI]
        vsini_type = vsini_params[0].vsini_function.value if vsini_params else 'AccurateFast'

        results = {}

        for index in range(observations.n_observations):
            obs = observations.observations[index]
            subgrid = subgrids.subgrids[index]

            if not obs.is_spectroscopic:
                continue

            wav_fit = ns.wave_fit[index]
            bounds = ns.bounds_lsq[index]

            star_flx = obs.star_flux if obs.star_flux is not None else np.array([])
            transm = obs.transm if obs.transm is not None else np.array([])
            system = obs.system if obs.system is not None else np.array([])

            file_tag = f'{obs.name}_{index}'

            logL_map = np.zeros((len(vsini_grid), len(rv_grid)))

            for j, vsini_val in enumerate(tqdm(vsini_grid, desc=f'RV-vsini map ({file_tag})', leave=False)):
                flx_broadened, res_broadened = vsini_fct(
                    wav_mod, flx_mod, res_mod, ld_value, vsini_val, vsini_type
                )

                logL = compute_ccf(
                    wav_mod, flx_broadened,
                    obs.wave, obs.flux, obs.err,
                    res_broadened, obs.res,
                    subgrid.res_cont, wav_fit,
                    star_flx_obs_spectro=star_flx,
                    transm_obs_spectro=transm,
                    system_obs_spectro=system,
                    rv_grid=rv_grid,
                    rv_sini_map=True,
                    bounds=bounds,
                    normalize=False
                )

                logL_map[j] = logL

            # Find best RV and vsini
            best_idx = np.unravel_index(np.argmax(logL_map), logL_map.shape)
            best_vsini = vsini_grid[best_idx[0]]
            best_rv = rv_grid[best_idx[1]]

            self._logger.info(f'    Best RV = {best_rv:.1f} km/s, best vsini = {best_vsini:.1f} km/s')

            results[file_tag] = {
                'rv_grid': rv_grid,
                'vsini_grid': vsini_grid,
                'logL_map': logL_map,
                'best_rv': best_rv,
                'best_vsini': best_vsini
            }

            if save_path is not None:
                path = Path(save_path) / f'rv_vsini_map_{file_tag}.npz'
                np.savez(path, **results[file_tag])
                self._logger.info(f'    RV-vsini map saved to {path}')

            if plot:
                fig, ax = plt.subplots(figsize=(8, 6))
                extent = [rv_grid[0], rv_grid[-1], vsini_grid[0], vsini_grid[-1]]
                im = ax.imshow(logL_map, aspect='auto', origin='lower', extent=extent, cmap='viridis')
                ax.scatter(best_rv, best_vsini, marker='x', color='red', s=100, label=f'Best: RV={best_rv:.1f}, vsini={best_vsini:.1f}')
                ax.set_xlabel('RV (km/s)')
                ax.set_ylabel(r'v.sin(i) (km/s)')
                ax.set_title(f'RV - v.sin(i) map - {file_tag}')
                ax.legend()
                fig.colorbar(im, ax=ax, label='log L')

                if save_path is not None:
                    fig_path = Path(save_path) / f'rv_vsini_map_{file_tag}.pdf'
                    fig.savefig(fig_path)
                    self._logger.info(f'    RV-vsini map plot saved to {fig_path}')

        return results

