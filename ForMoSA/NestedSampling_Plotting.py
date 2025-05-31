import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import corner

import ForMoSA.utils as u

class ForMoSAError(Exception):
    pass

class NestedSampling_Plotting(object):
    '''
    Class of visualisation of the results of the nested sampling.

    Parameters
    ----------
    logger               (Logger): Logger used
    plotting_config_dict   (dict): Dictionary of plotting configurations {'color': colors, 'edgecolor': edgecolors, 'marker': markers, 'size': sizes}

    Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
    '''

    def __init__(self, logger, plotting_config_dict: dict):

        self._logger = logger
        self._color_out = 'magenta'
        self._plotting_config = plotting_config_dict


    ##################################################
    # Properties
    ##################################################

    @property
    def logger(self):                  # Logger
        return self._logger

    @property
    def color_out(self):              # Color used for the corner, radar and chains plot
        return self._color_out

    @property
    def plotting_config(self):         # Configuration for the best fit plotting
        return self._plotting_config

    @property
    def color(self):                   # Color used for the best fit plotting
        return self.plotting_config['color']

    @property
    def edgecolor(self):               # Edgecolor used for the best fit plotting
        return self.plotting_config['edgecolor']

    @property
    def marker(self):                  # Marker used for the best fit plotting
        return self.plotting_config['marker']

    @property
    def size(self):                    # Size used for the best fit plotting
        return self.plotting_config['size']


    ##################################################
    # Methods
    ##################################################


    @staticmethod
    def _plot_data_point(ax: matplotlib.axes.Axes, axr: matplotlib.axes.Axes, obs_wav: np.ndarray, obs_flx: np.ndarray, mod_flx: np.ndarray, std_global: float, color: str, edgecolor: str, marker: str, size, label: str=None, yerr: np.ndarray=None, xerr: np.ndarray=None, plot_model: str='plot') -> None:
        '''
        Plot a single data point (either spectroscopic or photometric) with optional error bars and model curve.

        Parameters:
        ax    (matplotlib.axes.Axes): Main axes for the plot
        axr   (matplotlib.axes.Axes): Residuals axes for the plot
        obs_wav              (array): Observed wavelengths
        obs_flx              (array): Observed flux
        mod_flx              (array): Model flux
        std_global           (float): Global standard deviation for residuals normalization
        color                  (str): Color of the data points
        edgecolor              (str): Edge color of the data points
        marker                 (str): Marker style for the data points
        size                 (float): Size of the markers
        label                  (str): Label for the data points
        err                  (array): Error values for the data
        plot_model             (str): Type of plot for the model ('plot' or 'scatter')

        Authors: Allan Denis
        '''

        # Plot the observed data with error bars
        if marker == 'NA':
            ax.plot(obs_wav, obs_flx, c=color, label=label)
            axr.plot(obs_wav, (obs_flx - mod_flx) / std_global, c=color, alpha=0.7)
        else:
            ax.scatter(obs_wav, obs_flx, c=color, edgecolors=edgecolor, marker=marker, s=size, label=label, linewidths=2)
            axr.scatter(obs_wav, (obs_flx - mod_flx) / std_global, c=color, edgecolors=edgecolor, marker=marker, s=size, alpha=0.7, linewidths=2)

        # Add error bars if applicable
        if yerr is not None:
            ax.errorbar(obs_wav, obs_flx, yerr=yerr, fmt='none', ecolor=edgecolor, alpha=0.8)
        if xerr is not None:
            ax.errorbar(obs_wav, obs_flx, xerr=xerr, fmt='None', ecolor=edgecolor, alpha=0.8)

        # Optionally plot the model flux
        if plot_model == 'plot':
            ax.plot(obs_wav, mod_flx, c='black')
        elif plot_model == 'scatter':
            ax.scatter(obs_wav, mod_flx, marker='o', c='black', s=20)

        # Add residuals horizontal line at y=0
        axr.axhline(0, color='k', linestyle='--', alpha=0.5)

        return ax, axr


    @staticmethod
    def _get_label(ins: str, label_type: str, used_labels: set, default_label: str) -> str:
        '''
        Determine the label for the data based on the instrument and whether label display is enabled.

        Parameters:
        -----------
            ins            (str): Instrument name
            label_type     (str): Flag to determine if the instrument label should be shown
            used_labels    (set): Set of already used labels to avoid duplicates
            default_label  (str): Default label text

        Returns:
            - label (str): The appropriate label to use.

        Authors: Allan Denis
        '''

        if label_type == 'no':
            label = default_label if default_label not in used_labels else None
        else:
            label = f"{ins}"

        if label:
            used_labels.add(label)

        return label


    def plot_corner(self, results: dict, param_names: list, levels_sig: list=[0.997, 0.95, 0.68], bins: int=100, quantiles: tuple=(0.16, 0.5, 0.84), figsize: tuple=(15, 15)) -> matplotlib.figure.Figure:
        '''
        Method to corner plot the results samples

        Parameters
        ----------
        results       (dict): Dictionary of the results {'samples': samples, 'weights': weights}
        param_names   (list): Names of the parameters
        levels_sig    (list): 1, 2 and 3 sigma contour levels of the corner plot
        bins           (int): Number of bins for the posteriors
        quantiles    (tuple): Mean +- sigma to report the posterior values
        figsize      (tuple): Size of the figure to plot

        Returns:
            - fig (matplotlib.figure.Figure): Matplotlib Figure object

        '''

        self._logger.info('ForMoSA - Corner plot')

        if not results:
            msg = ' Results are empty. Please first run the nested sampling algorithm'
            self._logger.error(msg)
            raise ForMoSAError(msg)

        samples, weights = results['samples'], results['weights']
        rangee = [(np.min(results['samples'][:, i]), np.max(results['samples'][:, i])) for i in range(results['samples'].shape[1])]


        if len(param_names) != samples.shape[1]:
            msg = f' Param_names has a len ({len(param_names)}) different than the sampling shape ({samples.shape[1]}).'
            self._logger.error(msg)
            raise ForMoSAError(msg)

        fig = plt.figure(figsize=figsize)
        fig = corner.corner(samples,
                            weights=weights,
                            labels=param_names,
                            range=rangee,
                            levels=levels_sig,
                            bins=bins,
                            smooth=1,
                            quantiles=quantiles,
                            top_ticks=False,
                            plot_datapoints=False,
                            plot_density=True,
                            plot_contours=True,
                            fill_contours=True,
                            show_titles=True,
                            title_fmt='.2f',
                            title_kwargs=dict(fontsize=14),
                            contour_kwargs=dict(colors=self.color_out, linewidths=0.7),
                            pcolor_kwargs=dict(color='red'),
                            fig=fig,
                            label_kwargs=dict(fontsize=14))

        return fig


    def plot_chains(self, results: dict, param_names: list, param_best_values: dict, figsize:tuple=(12, 15), show_weights: bool=True) -> tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
        '''
        Method to plot the chains of the samples results.

        Parameters
        ----------
        results             (dict): Dictionary of results {'samples': samples, 'results': results}
        param_names         (list): List of parameter names
        param_best_values   (dict): Dictionary of best results of nested sampling {param_name: best_value}
        figsize            (tuple): Size of the figure to plot
        show_weights        (bool): Whether to overplot the weights

        Returns:
            - fig    (matplotlib.figure.Figure): Matplotlib Figure object
            - ax   (matplotlib.axes._axes.Axes): Matplotlib Axes object

        Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        self._logger.info(' Plotting posterior chains for each parameter.')

        if not results:
            msg = ' Results are empty. Please first run the nested sampling algorithm'
            self._logger.error(msg)
            raise ForMoSAError(msg)

        samples, weights = results['samples'], results['weights']

        if len(param_names) != samples.shape[1]:
            msg = f' Param_names has a len ({len(param_names)}) different than the sampling shape ({samples.shape[1]}).'
            self._logger.error(msg)
            raise ForMoSAError(msg)

        n_params = samples.shape[1]
        n_rows = (n_params + 1) // 2
        fig, axs = plt.subplots(n_rows, 2, figsize=figsize)
        axs = axs.flatten()

        for param_idx in range(n_params):
            ax = axs[param_idx]
            param_name = param_names[param_idx]
            ax.plot(samples[:, param_idx], color=self.color_out, alpha=0.8)
            ax.set_ylabel(param_name)

            if show_weights:
                ax_w = ax.twinx()
                ax_w.plot(weights, color='black', alpha=0.4)
                ax_w.set_yticks([])
                ax_w.text(x=0, y=0.00005, s='weights', fontsize=8)

            if param_name != 'log(L/L$\\mathrm{_{\\odot}}$)':
                ax.axhline(param_best_values[param_name], color='k', linestyle='--')

        for idx in range(n_params, len(axs)):
            fig.delaxes(axs[idx])

        fig.tight_layout()

        return fig, axs[:n_params]


    def plot_radar(self, results: dict, param_names: list, quantiles=[0.16, 0.5, 0.84], alpha_fill=0.2) -> tuple[plt.Figure, plt.Axes]:
        '''
        Method to radar plot the samples with normalized scaling based on prior-like ranges, and raw value annotations.

        Parameters
        ----------
        results       (dict): Dictionary of results {'samples': samples, 'weights': weights}
        param_names   (list): List of parameter names
        quantiles    (tuple): Mean +- sigma to report the posterior values
        alpha_fill   (float): Filling factor for the uncertainty

        Returns
        -------
        fig    (matplotlib.figure.Figure): Matplotlib Figure object
        ax   (matplotlib.axes._axes.Axes): Matplotlib Axes object
        '''

        self._logger.info(' Radar plot of the chains.')

        if not results:
            msg = ' Results are empty. Please first run the nested sampling algorithm.'
            self._logger.error(msg)
            raise ForMoSAError(msg)

        samples, weights = results['samples'], results['weights']

        if len(param_names) != samples.shape[1]:
            msg = f' param_names has a len ({len(param_names)}) different than the sampling shape ({samples.shape[1]}).'
            self._logger.error(msg)
            raise ForMoSAError(msg)

        # Compute quantiles for each parameter
        q_low, q_med, q_high = [], [], []
        for i in range(samples.shape[1]):
            q = u.weighted_quantile(samples[:, i], quantiles, weights=weights)
            q_low.append(q[0])
            q_med.append(q[1])
            q_high.append(q[2])

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
        angles = np.linspace(0, 2 * np.pi, len(param_names), endpoint=False).tolist()
        angles.append(angles[0])

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        ax.fill_between(angles, q_low_norm, q_high_norm, color=self.color_out, alpha=alpha_fill)
        ax.plot(angles, q_med_norm, color=self.color_out, linewidth=2)
        ax.scatter(angles[:-1], q_med_norm[:-1], color='black', s=20, zorder=3)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(param_names, fontsize=12)
        ax.set_yticklabels([])
        ax.set_title('Radar plot', size=14, pad=20)
        ax.grid(True)

        # Display ticks
        for i, angle in enumerate(angles[:-1]):
            min_val = prior_mins[i]
            max_val = prior_maxs[i]
            ticks = np.linspace(min_val, max_val, num=5)
            range_val = max_val - min_val if max_val != min_val else 1.0
            for i in range(len(ticks)-2):
                radius = (ticks[i+1] - min_val) / range_val
                ax.text(angle, radius, f'{ticks[i+1]:.2f}', ha='center', va='center', fontsize=8, color='black')

        return fig, ax


    def _get_plot_style(self, indobs: int, default_color: str='magenta', default_edge: str='darkmagenta', default_size: float=50, default_marker: str='NA') -> tuple[str, str, str, float]:
        '''
        Method to extract the plotting style (color, edgecolor, marker, size) for the given observation index.

        Parameters:
        ----------
        indobs          (int): Index of the observation.
        default_color   (str): Default color if the value is 'NA'.
        default_edge    (str): Default edge color if the value is 'NA'.
        default_size    float): Default marker size if the value is 'NA'.
        default_marker  (str): Default marker type if the value is 'NA'.

        Returns:
            - tuple: (color, edgecolor, marker, size)

        Authors: Allan Denis
        '''

        color = self.color[indobs]
        edgecolor = self.edgecolor[indobs]
        marker = self.marker[indobs]
        size = self.size[indobs]

        if color == 'NA':
            color = default_color
        if edgecolor == 'NA':
            edgecolor = default_edge
        if size == 'NA':
            size = default_size
        else:
            size = float(size)
        if marker == 'NA':
            marker = default_marker

        return color, edgecolor, marker, size


    def plot_fit(self, modif_data: dict, best_model: dict, figsize=(10, 7), uncert: str='yes', trans: str='yes', logx: str='no', logy: str='no', norm: str='no', label_ins: str='no') -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes, matplotlib.axes.Axes]:
        '''
        Method to plot the best fit compared with the data, including residuals and filter transmissions.

        Parameters:
        modif_data (dict): Modified data {indobs: {'spectro': dict, 'photo': dict}}.
        best_model (dict): Best model {indobs: {'spectro': dict, 'photo': dict}}.
        figsize   (tuple): Figure size.
        uncert      (str): Plot uncertainties if 'yes'.
        trans       (str): Plot transmission filters if 'yes'.
        logx        (str): Use logarithmic x-axis if 'yes'.
        logy        (str): Use logarithmic y-axis if 'yes'.
        norm        (str): Normalize spectra if 'yes'.
        label_ins   (str): Show instrument labels if 'yes'.

        Returns:
        tuple: (fig, ax, axr, axr2) where:
            - fig is the figure object,
            - ax is the main axes for spectra,
            - axr is the axes for residuals,
            - axr2 is the axes for residuals histogram.

        Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        self._logger.info('ForMoSA - Best fit and residuals plot')

        filter_ax = 'no'

        if not modif_data or not best_model:
            msg = 'Results are empty. Please run the sampling or compute the best model first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)

        fig = plt.figure(figsize=figsize)
        fig.tight_layout()
        ax = plt.subplot2grid((9, 11), (2, 0), rowspan=4, colspan=10)
        axr = plt.subplot2grid((9, 11), (6, 0), rowspan=1, colspan=10, sharex=ax)
        axr2 = plt.subplot2grid((9, 11), (6, 10), rowspan=1, colspan=1)

        # First pass: collect all residuals globally
        global_residuals = []
        global_flux = []
        for obs, mod in zip(modif_data.values(), best_model.values()):
            obs_spectro, obs_photo = obs['spectro'], obs['photo']
            mod_spectro, mod_photo = mod['spectro'], mod['photo']

            ck_spectro = mod_spectro.get('ck', 1) if norm == 'yes' else 1
            ck_photo = mod_photo.get('ck', 1) if norm == 'yes' else 1

            if len(obs_spectro['wav']) > 0:
                obs_flx = np.array(obs_spectro['flx']) / ck_spectro
                mod_flx = np.array(mod_spectro['flx']) / ck_spectro
                global_residuals.append(obs_flx - mod_flx)
                global_flux.append(obs_flx)

            if len(obs_photo['wav']) > 0:
                obs_flx = np.array(obs_photo['flx']) / ck_photo
                mod_flx = np.array(mod_photo['flx']) / ck_photo
                global_residuals.append(obs_flx - mod_flx)
                global_flux.append(obs_flx)

        global_flux = np.concatenate(global_flux)
        global_flux, factor = u.scale_to_one_significant_digit(obs_flx)
        global_residuals = np.concatenate(global_residuals) / (10 ** factor)
        std_global = np.nanstd(global_residuals)
        if std_global == 0 or np.isnan(std_global):
            std_global = 1.0  # Avoid division by zero

        used_labels = set()

        # Second pass: plot data (spectro and photo)
        for indobs, (obs, mod) in enumerate(zip(modif_data.values(), best_model.values())):
            obs_spectro, obs_photo = obs['spectro'], obs['photo']
            mod_spectro, mod_photo = mod['spectro'], mod['photo']

            ck_spectro = mod_spectro.get('ck', 1) if norm == 'yes' else 1
            ck_photo = mod_photo.get('ck', 1) if norm == 'yes' else 1


            # Get plot style for each observation
            color, edgecolor, marker, size = self._get_plot_style(indobs)

            # --- Spectroscopic data ---
            if len(obs_spectro['wav']) > 0:
                ins = obs_spectro.get('ins', ['unknown'])[0]
                obs_wav = np.array(obs_spectro['wav'])
                obs_flx = (np.array(obs_spectro['flx'])) / ck_spectro / (10**factor)
                mod_flx = np.array(mod_spectro['flx']) / ck_spectro / (10**factor)
                err = np.array(obs_spectro.get('err', [])) / ck_spectro / (10**factor) if uncert == 'yes' else None

                label = self._get_label(ins, label_ins, used_labels, 'Spectroscopic data')
                self._plot_data_point(ax, axr, obs_wav, obs_flx, mod_flx, std_global, color, edgecolor, marker, size, label, err)

                # Residuals histogram
                axr2.hist((obs_flx - mod_flx) / std_global, bins=100, orientation='horizontal', color='black', alpha=0.8, density=True)

            # --- Photometric data ---
            if len(obs_photo['wav']) > 0:
                ins = obs_photo.get('ins', ['unknown'])[0]
                obs_wav = np.array(obs_photo['wav'])[0]
                obs_flx = np.array(obs_photo['flx'])[0] / ck_photo / (10**factor)
                mod_flx = np.array(mod_photo['flx'])[0] / ck_photo / (10**factor)
                err = np.array(obs_photo.get('err', [])) / ck_photo / (10**factor) if uncert == 'yes' else None

                try:
                    filt = np.load(u.find_filter_file(ins))
                    x = filt['x_filt']
                    y = filt['y_filt']
                    files_loaded = True
                except Exception as e:
                    self._logger.warning(f"Could not load filter {ins}: {e}")
                    files_loaded = False

                label = self._get_label(ins, label_ins, used_labels, 'Photometric data')
                self._plot_data_point(ax, axr, obs_wav, obs_flx, mod_flx, std_global, color, edgecolor, marker, size, label, yerr = err, xerr = np.array([abs(x[0]-obs_wav), abs(x[-1]-obs_wav)])[:,np.newaxis], plot_model='scatter')

                # Transmission filters
                # Create the filter subplot once, outside the loop over instruments
                if trans == 'yes' and filter_ax == 'no' and files_loaded == True:
                    filter_ax = 'yes'
                    axfilt = plt.subplot2grid((9, 11), (0, 0), rowspan=2, colspan=10, sharex=ax)
                    axfilt.set_ylabel('Transmission')
                    axfilt.tick_params(bottom=False, labelbottom=False)
                    axfilt.set_ylim(0, 1.2)
                    if logx == 'yes':
                        axfilt.set_xscale('log')

                # Then, in the same block (or inside the photo loop), loop over all instruments and plot their filters
                if trans == 'yes' and files_loaded == True:
                    axfilt.plot(x, y, alpha=0.6, c=color)

        ax.legend(frameon=False)

        ax.set_ylabel(rf'Flux ($10^{{{factor}}}$ W m$^{-2}$ $\mu$m$^{-1}$)')
        axr.set_xlabel(r'Wavelength ($\mu$m)')
        axr.set_ylabel(r'Residuals ($\sigma$)')
        axr2.axis('off')
        ax.tick_params(bottom=False, labelbottom=False)

        return fig, ax, axr, axr2


