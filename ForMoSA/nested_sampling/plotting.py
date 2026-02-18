import numpy as np
import corner
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ForMoSA.utils.misc as utils
from ForMoSA.error import ForMoSAError


from matplotlib.patches import Circle
import matplotlib.patheffects as path_effects

from matplotlib import font_manager
font_path = '/Users/rajpoot/Library/Fonts/JuliaMono-Regular.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

# set font size to 18
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()


class NestedSamplingPlotting(object):
    '''
    Class of visualisation of the results of the nested sampling.

    Parameters
    ----------
    logger               (Logger): Logger used
    plotting_config_dict   (dict): Dictionary of plotting configurations {'color': colors, 'edgecolor': edgecolors, 'marker': markers, 'size': sizes}
    burn_in                 (int): Burn-in to apply to the chains

    Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
    '''

    def __init__(self, logger, plotting_config_dict: dict, burn_in: int = 0, ns_results: dict = dict(), list_params: list = []):

        self._logger = logger
        self._color_out = 'magenta'
        self._plotting_config = plotting_config_dict
        self._burn_in = burn_in
        self._ns_results = ns_results
        self._list_params = list_params


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

    @property
    def burn_in(self):                 # Burn-in to apply to the chains
        return self._burn_in

    @burn_in.setter                    # Burn-in setter
    def burn_in(self, burn_in):
        self._burn_in = burn_in
        return burn_in

    @property
    def ns_results(self):              # Results of Nested Sampling
        if len(self._ns_results) == 0:
            msg = 'Please run the Nested Sampling algorithm or load results of a Nested Sampling run'
            self._logger.critical(msg)
            raise ForMoSAError(msg)
        return self._ns_results

    @property
    def samples(self):                 # Samples
        return self.ns_results['samples']

    @property
    def weights(self):                 # Weights
        return self.ns_results['weights']

    @property
    def list_params(self):             # List of parameters
        return self._list_params

    @property
    def dict_params_idx(self):          # Index of parameters
        return {param: k for k, param in enumerate(self.list_params)}


    ##################################################
    # Methods
    ##################################################


    @staticmethod
    def _plot_data_point(ax: matplotlib.axes.Axes, axr: matplotlib.axes.Axes, obs_wav: np.ndarray, obs_flx: np.ndarray, mod_flx: np.ndarray, std_global: float, color: str, edgecolor: str, marker: str, size, label: str=None, yerr: np.ndarray=None, xerr: np.ndarray=None, plot_model: str='plot', plot_nativ_model: bool = False) -> None:
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
        if not(plot_nativ_model):
            if plot_model == 'plot':
                ax.plot(obs_wav, mod_flx, c='black')
            elif plot_model == 'scatter':
                ax.scatter(obs_wav, mod_flx, marker='o', c='black', s=50)

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


    def plot_corner(self, param_names: list = [], figsize: tuple=(15, 15), **corner_kwargs) -> matplotlib.figure.Figure:
        '''
        Method to corner plot the results samples

        Parameters
        ----------
        param_names   (list): Names of the parameters
        figsize      (tuple): Size of the figure to plot
        **corner_kwargs     : Remaining keeword args (see https://corner.readthedocs.io/)

        Returns:
            - fig (matplotlib.figure.Figure): Matplotlib Figure object

        Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        self._logger.info('ForMoSA - Corner plot')
        if len(param_names) == 0:
            param_names = list(self.dict_params_idx.keys())

        corner_kwargs['labels'] = param_names
        samples, weights = self.samples[self.burn_in:], self.weights[self.burn_in:]
        corner_kwargs['weights'] = weights
        corner_kwargs['range'] = [0.99 for i in range(len(param_names))]
        corner_kwargs['fill_contours'] = True
        corner_kwargs['plot_contours'] = True

        idx = []
        for param in param_names:
            idx.append(self.dict_params_idx[param])

        samples = samples[:, idx]

        fig = plt.figure(figsize=figsize)
        fig.clf()
        fig = corner.corner(samples, **corner_kwargs)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.97)

        return fig


    def plot_chains(self, param_names: list = [], figsize:tuple=(12, 15), show_weights: bool=True) -> tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
        '''
        Method to plot the chains of the samples results.

        Parameters
        ----------
        param_names         (list): List of parameter names
        figsize            (tuple): Size of the figure to plot
        show_weights        (bool): Whether to overplot the weights

        Returns:
            - fig    (matplotlib.figure.Figure): Matplotlib Figure object
            - ax   (matplotlib.axes._axes.Axes): Matplotlib Axes object

        Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        self._logger.info(' Plotting posterior chains for each parameter.')
        if len(param_names) == 0:
            param_names = list(self.dict_params_idx.keys())

        samples, weights = self.samples, self.weights

        idx = []
        for param in param_names:
            idx.append(self.dict_params_idx[param])

        samples = samples[:, idx]
        param_best_values = np.average(samples[self.burn_in:], axis=0, weights=weights[self.burn_in:])

        n_params = samples.shape[1]
        n_rows = (n_params + 1) // 2
        fig, axs = plt.subplots(n_rows, 2, figsize=figsize)
        axs = axs.flatten()

        for param_idx in range(n_params):
            ax = axs[param_idx]
            param_name = param_names[param_idx]
            ax.plot(samples[:, param_idx], color=self.color_out, alpha=0.8)
            ax.set_ylabel(param_name)
            ax.axvline(self.burn_in, linestyle='--', color='red')
            ax.text(x = 0.8, y = 0.8, s='burn in', color='red', transform=ax.transAxes, fontsize=14)

            if show_weights:
                ax_w = ax.twinx()
                ax_w.plot(weights, color='black', alpha=0.4)
                ax_w.set_yticks([])
                ax_w.text(x=0.8, y=0.70, s='weights', color='grey', transform=ax_w.transAxes, fontsize=14)

            if param_name != 'log(L/L$\\mathrm{_{\\odot}}$)':
                ax.axhline(param_best_values[param_idx], color='k', linestyle='--')

        for idx in range(n_params, len(axs)):
            fig.delaxes(axs[idx])

        fig.subplots_adjust(left=0.1, right=0.98, bottom=0.09, top=0.97)

        return fig, axs[:n_params]


    # def plot_radar(self, param_names: list = [], quantiles=[16, 50, 84], alpha_fill=0.2) -> tuple[plt.Figure, plt.Axes]:
    #     '''
    #     Method to radar plot the samples with normalized scaling based on prior-like ranges, and raw value annotations.

    #     Parameters
    #     ----------
    #     param_names   (list): List of parameter names
    #     quantiles    (tuple): Mean +- sigma to report the posterior values
    #     alpha_fill   (float): Filling factor for the uncertainty

    #     Returns
    #     -------
    #     fig    (matplotlib.figure.Figure): Matplotlib Figure object
    #     ax   (matplotlib.axes._axes.Axes): Matplotlib Axes object
    #     '''

    #     self._logger.info(' Radar plot of the chains.')
    #     if len(param_names) == 0:
    #         param_names = list(self.dict_params_idx.keys())

    #     samples, weights = self.samples[self.burn_in:], self.weights[self.burn_in:]

    #     idx = []
    #     for param in param_names:
    #         idx.append(self.dict_params_idx[param])

    #     samples = samples[:, idx]

    #     # Compute quantiles for each parameter
    #     q_low, q_med, q_high = [], [], []
    #     for i in range(samples.shape[1]):
    #         q = utils.get_weighted_percentile(quantiles, samples[:, i], weights=weights)
    #         q_low.append(q[0])
    #         q_med.append(q[1])
    #         q_high.append(q[2])

    #     q_low = np.array(q_low)
    #     q_med = np.array(q_med)
    #     q_high = np.array(q_high)

    #     # Use min/max of samples to simulate prior bounds
    #     prior_mins = np.min(samples, axis=0)
    #     prior_maxs = np.max(samples, axis=0)

    #     # Normalize based on "prior-like" range
    #     q_low_norm, q_med_norm, q_high_norm = [], [], []
    #     for i in range(len(q_low)):
    #         min_val = prior_mins[i]
    #         max_val = prior_maxs[i]
    #         range_val = max_val - min_val if max_val != min_val else 1.0
    #         q_low_norm.append((q_low[i] - min_val) / range_val)
    #         q_med_norm.append((q_med[i] - min_val) / range_val)
    #         q_high_norm.append((q_high[i] - min_val) / range_val)

    #     # Close the circle
    #     q_low_norm.append(q_low_norm[0])
    #     q_med_norm.append(q_med_norm[0])
    #     q_high_norm.append(q_high_norm[0])
    #     q_med = np.append(q_med, q_med[0])
    #     q_low = np.append(q_low, q_low[0])
    #     q_high = np.append(q_high, q_high[0])
    #     prior_mins = np.append(prior_mins, prior_mins[0])
    #     prior_maxs = np.append(prior_maxs, prior_maxs[0])

    #     # Angles for the radar plot
    #     angles = np.linspace(0, 2 * np.pi, len(param_names), endpoint=False).tolist()
    #     angles.append(angles[0])

    #     fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    #     ax.fill_between(angles, q_low_norm, q_high_norm, color=self.color_out, alpha=alpha_fill)
    #     ax.plot(angles, q_med_norm, color=self.color_out, linewidth=2)
    #     ax.scatter(angles[:-1], q_med_norm[:-1], color='black', s=20, zorder=3)

    #     ax.set_xticks(angles[:-1])
    #     ax.set_xticklabels(param_names, fontsize=12)
    #     ax.set_yticklabels([])
    #     # ax.set_title('Radar plot', size=14, pad=20)
    #     ax.grid(True)

    #     # Display ticks
    #     for i, angle in enumerate(angles[:-1]):
    #         min_val = prior_mins[i]
    #         max_val = prior_maxs[i]
    #         ticks = np.linspace(min_val, max_val, num=5)
    #         range_val = max_val - min_val if max_val != min_val else 1.0
    #         for i in range(len(ticks)-2):
    #             radius = (ticks[i+1] - min_val) / range_val
    #             ax.text(angle, radius, f'{ticks[i+1]:.2f}', ha='center', va='center', fontsize=8, color='black')

    #     return fig, ax

    def plot_radar(self, param_names: list = [], quantiles=[16, 50, 84], alpha_fill=0.15) -> tuple[plt.Figure, plt.Axes]:
        '''
        Improved method to radar plot the samples with enhanced visual design.
        
        Parameters
        ----------
        param_names   (list): List of parameter names
        quantiles    (tuple): Mean +- sigma to report the posterior values
        alpha_fill   (float): Filling factor for the uncertainty
        
        Returns
        -------
        fig    (matplotlib.figure.Figure): Matplotlib Figure object
        ax   (matplotlib.axes._axes.Axes): Matplotlib Axes object
        '''
        
        self._logger.info(' Radar plot of the chains.')
        if len(param_names) == 0:
            param_names = list(self.dict_params_idx.keys())
        
        samples, weights = self.samples[self.burn_in:], self.weights[self.burn_in:]
        
        idx = []
        for param in param_names:
            idx.append(self.dict_params_idx[param])
        
        samples = samples[:, idx]
        
        # Compute quantiles for each parameter
        q_low, q_med, q_high = [], [], []
        for i in range(samples.shape[1]):
            q = utils.get_weighted_percentile(quantiles, samples[:, i], weights=weights)
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
        
        # Create figure with custom styling
        fig = plt.figure(figsize=(6, 6), dpi=300)
        ax = fig.add_subplot(111, polar=True)
        
        # Set background color for modern look
        fig.patch.set_facecolor('#FAFBFC')
        ax.set_facecolor('#FFFFFF')
        
        # Improved color scheme - using a sophisticated blue-purple gradient
        main_color = '#4A5FD9'  # Deep blue
        fill_color = '#6B7FE8'  # Lighter blue
        uncertainty_color = '#A8B3F5'  # Very light blue
        
        # Plot uncertainty band with gradient effect
        ax.fill_between(angles, q_low_norm, q_high_norm, 
                        color=uncertainty_color, alpha=0.35, linewidth=0, zorder=2)
        
        # Add a second, more opaque layer for the inner region
        # mid_low = [(q_low_norm[i] + q_med_norm[i]) / 2 for i in range(len(q_low_norm))]
        # mid_high = [(q_high_norm[i] + q_med_norm[i]) / 2 for i in range(len(q_high_norm))]
        # ax.fill_between(angles, mid_low, mid_high, 
        #                 color=fill_color, alpha=0.25, linewidth=0, zorder=2)
        
        # Plot main line with enhanced styling
        ax.plot(angles, q_med_norm, color=main_color, linewidth=2.5, zorder=3, 
                solid_capstyle='round')
        
        # Add larger, styled markers at each point
        for i in range(len(angles[:-1])):
            # Outer white ring for contrast
            ax.scatter(angles[i], q_med_norm[i], color='white', s=120, zorder=4, 
                    edgecolors='none')
            # Main point
            ax.scatter(angles[i], q_med_norm[i], color=main_color, s=80, zorder=5, 
                    edgecolors='white', linewidths=2)
        
        # Customize gridlines for cleaner look
        ax.grid(True, color='gray', linewidth=1.2, alpha=0.5, linestyle='--', zorder=1)
        
        # Style the radial gridlines
        ax.spines['polar'].set_color("#808183")
        ax.spines['polar'].set_linewidth(1.5)
        
        # Set parameter labels with improved styling - positioned further out
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(param_names, fontsize=12, fontweight='600', 
                        color='#24292E')
        
        # Remove default radial labels
        ax.set_yticklabels([])
        ax.set_ylim(0, 1)

        # Display ticks
        # for i, angle in enumerate(angles[:-1]):
        #     min_val = prior_mins[i]
        #     max_val = prior_maxs[i]
        #     ticks = np.linspace(min_val, max_val, num=5)
        #     range_val = max_val - min_val if max_val != min_val else 1.0
        #     for i in range(len(ticks)-2):
        #         radius = (ticks[i+1] - min_val) / range_val
        #         ax.text(angle, radius, f'{ticks[i+1]:.2f}',
        #                 ha='center', va='center', fontsize=8, 
        #                 color='#586069', family='sans-serif', zorder=10)

        
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
             
            # Get the median value for annotation
            value = q_med[i]
            
            # Format the value nicely
            if abs(value) >= 1000:
                value_str = f'{value:.0f}'
            elif abs(value) >= 10:
                value_str = f'{value:.1f}'
            else:
                value_str = f'{value:.2f}'
            
            # Create text with shadow effect for better readability
            text = ax.text(angle+0.15, label_radius, value_str, 
                        ha='center', va='center', 
                        fontsize=12, fontweight='600',
                        color='#24292E',
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
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
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

        color = self.color[indobs % len(self.color)]
        edgecolor = self.edgecolor[indobs % len(self.edgecolor)]
        marker = self.marker[indobs % len(self.marker)]
        size = self.size[indobs % len(self.size)]

        if color == 'NA':
            color = f'C{indobs}'
        if edgecolor == 'NA':
            edgecolor = color
        if size == 'NA':
            size = default_size
        else:
            size = float(size)
        if marker == 'NA':
            marker = default_marker

        return color, edgecolor, marker, size


    def plot_fit(self, modif_data: dict, best_model: dict, figsize=(10, 7), uncert: str='yes', trans: str='yes', logx: str='no', logy: str='no', norm: str='no', label_ins: str='no', plot_high_contrast: bool = False, plot_nativ_model: bool = False, nativ_model: dict = {}, label_params: bool = False) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes, matplotlib.axes.Axes]:
        '''
        Method to plot the best fit compared with the data, including residuals and filter transmissions.

        Parameters:
        modif_data           (dict): Modified data {indobs: {'spectro': dict, 'photo': dict}}
        best_model           (dict): Best model {indobs: {'spectro': dict, 'photo': dict}}
        figsize             (tuple): Figure size.
        uncert                (str): Plot uncertainties if 'yes'.
        trans                 (str): Plot transmission filters if 'yes'.
        logx                  (str): Use logarithmic x-axis if 'yes'.
        logy                  (str): Use logarithmic y-axis if 'yes'.
        norm                  (str): Normalize spectra if 'yes'.
        label_ins             (str): Show instrument labels if 'yes'.
        plot_high_contract   (bool): Whether to plot high contrast data
        plot_nativ_model     (bool): Whether to plot nativ model
        nativ_model          (dict): Nativ model
        label_params         (bool): Whether to label best parameters for the model

        Returns:
        tuple: (fig, ax, axr, axr2) where:
            - fig is the figure object,
            - ax is the main axes for spectra,
            - axr is the axes for residuals,
            - axr2 is the axes for residuals histogram.

        Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        self._logger.info('ForMoSA - Best fit and residuals plot')

        if plot_nativ_model and not(nativ_model):
            msg = 'If you want to plot the nativ model, please provide the dictionary of nativ model'
            self._logger_error(msg)
            raise ForMoSAError(msg)

        filter_ax = 'no'

        if not modif_data or not best_model:
            msg = 'Results are empty. Please run the sampling or compute the best model first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)

        samples, weights = self.samples[self.burn_in:], self.weights[self.burn_in:]
        param_best_values = np.average(samples, axis=0, weights=weights)

        fig = plt.figure(figsize=figsize)
        fig.clf()
        gs = gridspec.GridSpec(9, 11)
        ax = fig.add_subplot(gs[2:7, 0:10])
        axr = fig.add_subplot(gs[7:9, 0:10], sharex=ax)
        axr2 = fig.add_subplot(gs[7:9, 10:11], sharey=axr)

        # First pass: collect all residuals globally
        # This step is done to rescale the data globally such that there is only one significant digit in the flux
        global_residuals = []
        global_flux = []
        global_wavelength = []
        for obs, mod in zip(modif_data.values(), best_model.values()):
            obs_spectro, obs_photo = obs['spectro'], obs['photo']
            mod_spectro, mod_photo = mod['spectro'], mod['photo']

            if len(obs_spectro['wav']) > 0:
                obs_flx = np.array(obs_spectro['flx'])
                obs_wav = np.array(obs_spectro['wav'])
                if not(plot_high_contrast) and np.unique(obs_spectro['speckles'][0] != 0) and len(modif_data) > 1:
                    continue
                elif len(modif_data) == 1:
                    plot_high_contrast = True

                mod_flx = np.array(mod_spectro['flx'])
                global_residuals.append(obs_flx - mod_flx)
                global_flux.append(obs_flx)
                global_wavelength.append(obs_wav)

            if len(obs_photo['wav']) > 0:
                obs_flx = np.array(obs_photo['flx'])
                obs_wav = np.array(obs_photo['wav'])
                mod_flx = np.array(mod_photo['flx'])
                global_residuals.append(obs_flx - mod_flx)
                global_flux.append(obs_flx)
                global_wavelength.append(obs_wav)

        global_flux = np.concatenate(global_flux)
        global_wavelength = np.concatenate(global_wavelength)
        isort = np.argsort(global_wavelength)
        global_flux, global_wavelength = global_flux[isort], global_wavelength[isort]

        # Rescale the global data to one significant digit
        global_flux, factor = utils.scale_to_one_significant_digit(global_flux)
        global_residuals = np.concatenate(global_residuals) / (10 ** factor)
        std_global = np.nanstd(global_residuals)
        if std_global == 0 or np.isnan(std_global):
            std_global = 1.0  # Avoid division by zero

        used_labels = set()

        # Second pass: plot data (spectro and photo)
        for indobs, (obs, mod) in enumerate(zip(modif_data.values(), best_model.values())):
            obs_spectro, obs_photo = obs['spectro'], obs['photo']
            mod_spectro, mod_photo = mod['spectro'], mod['photo']

            # --- Spectroscopic data ---
            if len(obs_spectro['wav']) > 0:
                # Get plot style for each spectroscopic observation
                if not(label_ins):
                    default_color = 'magenta'
                else:
                    default_color = 'NA'
                color, edgecolor, marker, size = self._get_plot_style(indobs, default_color=default_color, default_edge='darkmagenta')

                # Get data for each spectroscopic observation
                obs_wav = np.array(obs_spectro['wav'])
                obs_flx = np.array(obs_spectro['flx']) / (10**factor)
                speckles = np.array(obs_spectro['speckles']) / (10**factor)
                system = np.array(obs_spectro['estimated_system']) / (10 ** factor)
                mod_flx = np.array(mod_spectro['flx']) / (10**factor)
                err = np.array(obs_spectro.get('err', None)) / (10**factor) if uncert == 'yes' else None

                if np.unique(speckles)[0] != 0 and not(plot_high_contrast):
                    continue
                elif np.unique(speckles)[0] != 0:
                    mod_flx += speckles
                if np.unique(system)[0] != 0 and not(plot_high_contrast):
                    continue
                elif np.unique(system)[0] != 0:
                    mod_flx += system

                # Get label for each spectroscopic observation
                ins = obs_spectro.get('ins', ['unknown'])[0]
                label = self._get_label(ins, label_ins, used_labels, 'Spectroscopic data')
                self._plot_data_point(ax, axr, obs_wav, obs_flx, mod_flx, std_global, color, edgecolor, marker, size, label, err, plot_nativ_model = plot_nativ_model)

                # Residuals histogram
                axr2.hist((obs_flx - mod_flx) / std_global, bins=100, orientation='horizontal', color='black', alpha=0.8, density=True)

            # --- Photometric data ---
            if len(obs_photo['wav']) > 0:
                # Get plot style for each observation
                if not(label_ins):
                    default_color = 'magenta'
                else:
                    default_color = 'NA'
                color, edgecolor, marker, size = self._get_plot_style(indobs, default_color='blue', default_edge='darkblue', default_marker='D')

                # Get data for each photometric observation
                obs_wav = np.array(obs_photo['wav'])[0]
                obs_flx = np.array(obs_photo['flx'])[0] / (10**factor)
                mod_flx = np.array(mod_photo['flx'])[0] / (10**factor)
                err = np.array(obs_photo.get('err', [])) / (10**factor) if uncert == 'yes' else None

                # Get label for each photometric observation
                ins = obs_photo.get('ins', ['unknown'])[0]
                try:
                    filt = np.load(utils.find_filter_file(ins))
                    x = filt['x_filt']
                    y = filt['y_filt']
                    cut = y > 1e-2   # Sometimes the transmission filters are extremely extended so we apply a cut to make the extension more consistent with the shape of the tramsission filter
                    x, y = x[cut], y[cut]
                    files_loaded = True
                except Exception as e:
                    self._logger.warning(f"Could not load filter {ins}: {e}")
                    files_loaded = False

                label = self._get_label(ins, label_ins, used_labels, 'Photometric data')
                self._plot_data_point(ax, axr, obs_wav, obs_flx, mod_flx, std_global, color, edgecolor, marker, size, label, yerr = err, xerr = np.array([abs(x[0]-obs_wav), abs(x[-1]-obs_wav)])[:,np.newaxis], plot_model='scatter', plot_nativ_model=plot_nativ_model)

                # Transmission filters
                # Create the filter subplot once, outside the loop over instruments
                if trans == 'yes' and filter_ax == 'no' and files_loaded == True:
                    filter_ax = 'yes'
                    axfilt = fig.add_subplot(gs[0:2, 0:10], sharex=ax)
                    axfilt.set_ylabel('Transmission')
                    axfilt.tick_params(bottom=False, labelbottom=False)
                    axfilt.set_ylim(0, 1.2)
                    if logx == 'yes':
                        axfilt.set_xscale('log')

                # Then, in the same block (or inside the photo loop), loop over all instruments and plot their filters
                if trans == 'yes' and files_loaded == True:
                    axfilt.plot(x, y, alpha=0.6, c=color)

            if plot_nativ_model:
                mod_wav, mod_flx = nativ_model['spectro']['wav'], nativ_model['spectro']['flx'] / (10 ** factor)
                ax.plot(mod_wav, mod_flx, c='black')

        ax.legend(frameon=False)

        ax.set_ylabel(rf'Flux ($10^{{{factor}}}$ W m$^{-2}$ $\mu$m$^{-1}$)')
        axr.set_xlabel(r'Wavelength ($\mu$m)')
        axr.set_ylabel(r'Residuals ($\sigma$)')
        axr2.axis('off')
        ax.tick_params(bottom=False, labelbottom=False)

        if label_params:
            text_str = ', '.join([f"{key} = {param_best_values[i]:.2g}" for key, i in self.dict_params_idx.items()])
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='top')

        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.11, top=0.97)

        return fig, ax, axr, axr2, (1 / 10 ** factor)

