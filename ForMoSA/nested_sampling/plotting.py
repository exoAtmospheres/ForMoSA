import copy
import corner
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.axes._axes import Axes
import matplotlib.patheffects as path_effects

from ForMoSA.core.config import PLOTS_CONFIG, MAIN_PLOT
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
    log_level : str
        Level of the Logger

    Notes
    -----
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
        Radar plot the samples.

        Parameters
        ----------
        config : RadarPlotConfig
            Instance of class RadarPlotConfig


        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
            Tuple containing Figure and Ax objects

        Notes
        -----
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

        # Plot uncertainty band with gradient effect
        ax.fill_between(angles, q_low_norm, q_high_norm, color=config.color_uncertainty, alpha=config.alpha_fill, zorder=2)

        # Plot main line with enhanced styling
        ax.plot(angles, q_med_norm, color=config.color_radar, linewidth=2.5, zorder=3, solid_capstyle='round')

        # Add larger, styled markers at each point
        for i in range(len(angles[:-1])):
            # Outer white ring for contrast
            ax.scatter(angles[i], q_med_norm[i], color='white', s=config.size_quantiles+40, zorder=4,
                    edgecolors='none')
            # Main point
            ax.scatter(angles[i], q_med_norm[i], color=config.color_quantiles, s=config.size_quantiles, zorder=5,
                    edgecolors='white', linewidths=config.lw_quantiles)

        # Set parameter labels with improved styling - positioned further out
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.ns_results.free_parameters, fontsize=config.fontsize_names, fontweight='600', color='#24292E')

        # Remove default radial labels
        ax.set_yticklabels([])

        # Customize gridlines for cleaner look
        ax.grid(True, color='gray', linewidth=1.2, alpha=0.5, linestyle='--', zorder=1)

        # Style the radial gridlines
        ax.spines['polar'].set_color("#808183")
        ax.spines['polar'].set_linewidth(1.5)

        # Display ticks
        # for i, angle in enumerate(angles[:-1]):
        #     min_val = prior_mins[i]
        #     max_val = prior_maxs[i]
        #     ticks = np.linspace(min_val, max_val, num=5)
        #     range_val = max_val - min_val if max_val != min_val else 1.0
        #     for i in range(len(ticks)-2):
        #         radius = (ticks[i+1] - min_val) / range_val
        #         ax.text(angle, radius, f'{ticks[i+1]:.2f}', ha='center', va='center', fontsize=config.fontisze_ticks, color=config.color_ticks)

        # Add value annotations with improved positioning and styling
        # Only show values at the median points, positioned outside the plot
        for i, angle in enumerate(angles[:-1]):
            # Position the value label slightly offset from the data point
            # We'll offset it radially outward from the median point
            data_radius = q_med_norm[i]

            # Calculate offset: place label slightly outside the data point
            label_radius = data_radius + 0.14  # Offset by a small amount

            # If the point is too close to center, push label further out
            if data_radius < 0.15:
                label_radius = 0.45

            # Get the median and quantile values for annotation
            med = q_med[i]
            low = med - q_low[i]
            high = q_high[i] - med

            # Format the values nicely
            if abs(med) >= 1000:
                med_str = f'{med:.0f}'
            elif abs(med) >= 10:
                med_str = f'{med:.1f}'
            else:
                med_str = f'{med:.2f}'

            # Format the quantile values nicely
            if abs(low) >= 1000:
                q_low_str = f'{low:.0f}'
            elif abs(low) >= 10:
                q_low_str = f'{low:.1f}'
            else:
                q_low_str = f'{low:.2f}'

            if abs(high) >= 1000:
                q_high_str = f'{high:.0f}'
            elif abs(high) >= 10:
                q_high_str = f'{high:.1f}'
            else:
                q_high_str = f'{high:.2f}'

            # Create text with shadow effect for better readability
            text = ax.text(angle+0.15, label_radius, f'${med_str}_{{-{q_low_str}}}^{{+{q_high_str}}}$',
                        ha='center', va='center',
                        fontsize=config.fontisze_ticks, fontweight='600',
                        color=config.color_ticks,
                        zorder=10,
                        bbox=dict(boxstyle='round,pad=0.4',
                                facecolor='white',
                                edgecolor='none',
                                alpha=0.85))

            # Add subtle shadow effect
            text.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground='#E1E4E8', alpha=0.5),
                path_effects.Normal()
            ])

        return fig, ax

    def plot_fit(self, observations: ObservationSet, best_fit: list[ObservedModel], figsize: tuple[float, float] = (18, 8), plot_native_model: bool = False, native_model: ObservedModel | None = None) -> tuple[Figure, Axes, Axes, Axes, Axes]:
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

        # obs_set_transformed = ObservationSet(self.logger)

        # for i, obs in enumerate(observations.observations):
        #     # Create a copy of the observations to optionally remove component estimated by high-contrast module
        #     obs_transformed = copy.deepcopy(obs)
        #     obs_transformed._flux -= best_fit[i].component

        #     obs_set_transformed.add_observation(obs_transformed)

        # Reserve top rows for filter axis only when photometry is present
        ax_row_start = 2 if observations.has_photometry else 0

        fig = plt.figure(figsize=figsize)
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
            ax.plot(native_model.wave, native_model.flux, color=config.color_fit, linewidth=config.linewidth, zorder=config.zorder)

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
        axr.set_ylabel(r'Residuals ($\sigma$)')
        axr.axhline(y=0, linestyle='--', color='grey')
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

    def _get_PT_chem(self, path_PT_chem_grid, theta, grid_used='original'):
        '''
        Function to extract the pressure, temperature and chemical profiles from the PT grid.
        This function interpolates the PT grid at the best-fit parameters and computes the brightness temperature of the photosphere.
        It also extracts the P/T profile of the photosphere based on the brightness temperature.
        This is useful for plotting the P/T profile and the photosphere in the final figure.

        Parameters
        ----------
            path_PT_chem_grid : str
                path to the PT grid in xarray format
            theta : list
                parameter values to use to compute the photosphere
            grid_used : str
                (default = 'original') Path to the grid from where to extract the spectrum. If 'original', the current grid will be used.
     
        Returns
        -------
            PT_chem : dict
                Dictionary containing the pressure, temperature and chemical arrays
            photosphere : dict 
                Dictionary containing the P/T profile of the photosphere

        Notes
        -----
        Authors: Matthieu Ravet (adapted from Nathan Zimniak)
        '''

        ds = xr.open_dataset(path_PT_chem_grid, decode_cf=False, engine='netcdf4')
        var_names = list(ds.data_vars.keys())  # Save original keys
        P = ds.coords['pressure']

        # Prepare storage per variable
        best_fit_vars = {var: [] for var in var_names}

        for j in tqdm(range(len(self.ns_results.samples)), desc=f"Interpolating grid", unit="sample"):
            sample = self.ns_results.samples[j]
            interp_kwargs = {f"par{k+1}": sample[k] for k in range(len(ds.coords)-1)}
            interp = ds.interp(**interp_kwargs, method=self.config_adapt.method, kwargs={"fill_value": "extrapolate"}).to_array()
            
            for v_idx, var in enumerate(var_names):
                best_fit_vars[var].append(interp[v_idx].values)
        ds.close()

        # Convert lists to arrays and compute percentiles
        PT_chem = {}
        PT_chem["pressure"] = P.values

        for var in var_names:
            grid = np.array(best_fit_vars[var])
            PT_chem[var + '_q2'] = np.percentile(grid, 2, axis=0)
            PT_chem[var + '_q16'] = np.percentile(grid, 16, axis=0)
            PT_chem[var + '_q50'] = np.percentile(grid, 50, axis=0)
            PT_chem[var + '_q84'] = np.percentile(grid, 84, axis=0)
            PT_chem[var + '_q98'] = np.percentile(grid, 98, axis=0)

        # - - - - 

        # Recover the original grid
        if grid_used == 'original':
            path_grid = self.global_params.model_path
        else:
            path_grid = grid_used

        # Extract spectrum for photosphere estimation
        ds = xr.open_dataset(path_grid, decode_cf=False, engine='netcdf4')
        grid = ds['grid']
        wav = grid["wavelength"].values
        interp_kwargs = {f"par{k+1}": theta[k] for k in range(len(ds.coords)-1)}
        flx = np.array(grid.interp(**interp_kwargs, method=self.config_adapt.method, kwargs={"fill_value": "extrapolate"}))
        ds.close()

        # Photosphere range (in µm) and conversion
        photosphere_wav = np.asarray([0, 10])
        mask = (wav > photosphere_wav[0]) & (wav < photosphere_wav[1])
        wav = wav[mask] * 1e-6  # m
        flx = flx[mask] * 1e6   # W/m²/m

        # Compute brightness temperatures
        brightness_temperature = np.zeros_like(wav)
        for j in range(len(wav)):
            brightness_temperature[j] = (
                cst.h.value * cst.c.value / (cst.k_B.value * wav[j]) /
                np.log(1 + (2 * cst.h.value * cst.c.value ** 2) /
                        (wav[j] ** 5 * (flx[j] / np.pi)))
            )

        brightness_temperature = brightness_temperature[np.isfinite(brightness_temperature)]
        T_min = brightness_temperature.min()
        T_max = brightness_temperature.max()

        # Match to thermal profile
        P, T = PT_chem["pressure"], PT_chem["temperature_q50"] # Taking the 'best' profile as reference
        mask = (T >= T_min) & (T <= T_max)

        # Store photosphere info
        photosphere = {
            "pressure": P[mask],
            "temperature": T[mask],
            "brightness_temperature_range": [T_min, T_max]
        }
    
        return PT_chem, photosphere

    def plot_PT_chem(self, PT_chem, photosphere={}, par_to_plot=['temperature'], figsize=(10,5)):
        '''
        Function to plot the Pressure-Temperature profiles and associated vmr/molecular profiles.

        Parameters
        ----------
            PT_chem : dict
                Dictionary containing the pressure, temperature and chemical arrays
            photosphere : dict
                Dictionary containing the P/T profile of the photosphere
            par_to_plot : list(str)
                Key list of the parameters from the PT_chem you want to plot 

        Returns
        -------
            fig : object
                matplotlib figure object
            ax : object
                matplotlib axes objects
            ax_twin : object
                matplotlib axes objects

        Notes
        -----
        Authors: Matthieu Ravet (adapted from Nathan Zimniak)
        '''

        self._logger.info('    Plotting PT and chemistry')

        # Initialize plot
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax_twin = ax.twiny()

        # Iterate on each parameter
        for i_par, par in enumerate(par_to_plot):

            # Temperature
            if par == 'temperature':
                ax.plot(PT_chem["temperature_q50"], PT_chem["pressure"], color=config.color_fit, label='Best-fit')
                ax.fill_betweenx(PT_chem["pressure"], PT_chem["temperature_q16"], PT_chem["temperature_q84"], color=config.color_fit, alpha=0.1, label=r'2 $\sigma$')
                ax.fill_betweenx(PT_chem["pressure"], PT_chem["temperature_q2"], PT_chem["temperature_q98"], color=config.color_fit, alpha=0.2, label=r'1 $\sigma$')
            else:
                ax_twin.plot(PT_chem[par + '_q50'], PT_chem["pressure"], label=par)

        # Add photosphere if necessary
        if len(photosphere) != 0:
            ax.plot(photosphere["temperature"], photosphere["pressure"], color='red', linestyle='--', label='photosphere')

        # Plot the legend

        # First ax
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Pressure (bar)')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='both')

        # Second ax
        ax_twin.set_xlabel('abundance/vmr')
        ax_twin.set_xscale('log')

        # ask matplotlib for the plotted objects and their labels
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax_twin.legend(lines + lines2, labels + labels2)
        ax_twin.tick_params(axis='both', which='both')

        return fig, ax, ax_twin