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
    NestedSampling   (NestedSampling): Instance of :class:'~NestedSampling'
    color_out                   (str): Colour used for the plots
    
    Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
    '''
    
    def __init__(self, logger, color_out: str = 'purple'):
        
        self.color_out = color_out
        self._logger = logger  # Optionnel mais utile
        
        
    def _plot(self, results: dict, param_names: list, param_best_values: dict, modif_data: dict, best_model: dict) -> None:
        '''
        Method to use all the plotting methods

        Parameters
        ----------
        results       (dict): Dictionary of the results {'samples': samples, 'weights': weights}
        param_names   (list): Names of the parameters
        param_best_value    (dict): Dictionary of best results of nested sampling {param_name: best_value}
        modif_data (dict): Dictionary containing the modified data {indobs: {'spectro': dict, 'photo': dict}}
        best_model (dict): Dictionary containing the best model {indobs: {'spectro': dict, 'photo': dict}}
    
        Authors: Allan Denis
        '''

        self._plot_corner(results, param_names)
        self._plot_chains(results, param_names, param_best_values)
        self._plot_radar(results, param_names)
        self._plot_fit(modif_data, best_model)
        

    def _plot_corner(self, results: dict, param_names: list, levels_sig: list=[0.997, 0.95, 0.68], bins: int=100, quantiles: tuple=(0.16, 0.5, 0.84), figsize: tuple=(15, 15)) -> matplotlib.figure.Figure:
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
            fig (matplotlib.figure.Figure): Matplotlib Figure object

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


    def _plot_chains(self, results: dict, param_names: list, param_best_values: dict, figsize:tuple=(12, 15), show_weights: bool=True) -> tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
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
    

    def _plot_radar(self, results: dict, param_names: list, quantiles=[0.16, 0.5, 0.84], alpha_fill=0.2) -> tuple[plt.Figure, plt.Axes]:
        '''
        Method to radar plot the samples with normalized scaling but raw value annotations
    
        Parameters
        ----------
        results       (dict): Dictionary of results {'samples': samples, 'weights': weights}
        param_names   (list): List of parameter names
        quantiles    (tuple): Mean +- sigma to report the posterior values
        alpha_fill   (float): Filling factor for the uncertainty
    
        Returns:
        fig    (matplotlib.figure.Figure): Matplotlib Figure object
        ax   (matplotlib.axes._axes.Axes): Matplotlib Axes object
        '''
    
        self._logger.info(' Radar plot of the chains.')
        
        if not results:
            msg = ' Results are empty. Please first run the nested sampling algorithm'
            self._logger.error(msg)
            raise ForMoSAError(msg)
    
        samples, weights = results['samples'], results['weights']
        
        if len(param_names) != samples.shape[1]:
            msg = f' Param_names has a len ({len(param_names)}) different than the sampling shape ({samples.shape[1]}).'
            self._logger.error(msg)
            raise ForMoSAError(msg)
    
        q_low, q_med, q_high = [], [], []
    
        for i in range(samples.shape[1]):
            q = u.weighted_quantile(samples[:, i], quantiles, weights=weights)
            q_low.append(q[0])
            q_med.append(q[1])
            q_high.append(q[2])
    
        q_low = np.array(q_low)
        q_med = np.array(q_med)
        q_high = np.array(q_high)
    
        # Normalisation paramètre par paramètre
        q_low_norm = []
        q_med_norm = []
        q_high_norm = []
        for i in range(len(q_low)):
            min_val = q_low[i]
            max_val = q_high[i]
            range_val = max_val - min_val if max_val != min_val else 1.0
            q_low_norm.append(0)
            q_med_norm.append((q_med[i] - min_val) / range_val)
            q_high_norm.append(1)
    
        # Fermer le cercle
        q_low_norm.append(q_low_norm[0])
        q_med_norm.append(q_med_norm[0])
        q_high_norm.append(q_high_norm[0])
        q_med = np.append(q_med, q_med[0])
        q_low = np.append(q_low, q_low[0])
        q_high = np.append(q_high, q_high[0])
    
        angles = np.linspace(0, 2 * np.pi, len(param_names), endpoint=False).tolist()
        angles.append(angles[0])
    
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
        ax.fill_between(angles, q_low_norm, q_high_norm, color=self.color_out, alpha=alpha_fill)
        ax.plot(angles, q_med_norm, color=self.color_out, linewidth=2)
        ax.scatter(angles[:-1], q_med_norm[:-1], color='black', s=20, zorder=3)
    
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(param_names, fontsize=12)
        ax.set_yticklabels([])  # Remove standard tickes as parameters are normalised
        ax.set_title('radar plot', size=14, pad=20)
        ax.grid(True)
    
        # Ticks for each parameter
        for angle, q_min, q_max in zip(angles[:-1], q_low[:-1], q_high[:-1]):
            ticks = np.linspace(q_min, q_max, num=3)  # ajustable
            for i, tick_val in enumerate(ticks):
                radius = (i + 1) / 3  # linear between 0 and 1
                ax.text(angle, radius, f'{tick_val:.2f}',
                        ha='center', va='center', fontsize=8, color='black')
    
        return fig, ax


    def _plot_fit(self, modif_data: dict, best_model: dict, figsize=(15, 11), uncert='no', trans='no', logx='no', logy='no', norm='no') -> tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes, matplotlib.axes._axes.Axes, matplotlib.axes._axes.Axes]:
        '''
        Method to plot the best fit coomparing with the data

        Parameters
        ----------
        modif_data (dict): Dictionary containing the modified data {indobs: {'spectro': dict, 'photo': dict}}
        best_model (dict): Dictionary containing the best model {indobs: {'spectro': dict, 'photo': dict}}
        figsize  (tuple): Size of the figure to plot
        uncert     (str): Whether to overplot the uncertainties
        trans      (str): Whether to overplot the transmission filters
        logx       (str): Whether to use a logarithm scale for the x-axis
        logy       (str): Whether to use a logarithm scale for the y-axis
        norm       (str): Whether to plot the normalized spectra
  
        Returns:
            - fig     (matplotlib.figure.Figure): Figure object
            - ax   (matplotlib.axes._axes.Axes): Axes object, main spectra
            - axr  (matplotlib.axes._axes.Axes): Axes object, residuals
            - axr2 (matplotlib.axes._axes.Axes): Axes object, density histogram

        Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''
        
        self._logger.info(' ForMoSA - Best fit and residuals plot')
    
        if not modif_data or not best_model:
            msg = ' Results are empty. Please first run the nested sampling algorithm'
            self._logger.error(msg)
            raise ForMoSAError(msg)
    
        # Setup plot layout
        fig = plt.figure(figsize=figsize)
        fig.tight_layout()
        ax = plt.subplot2grid((9, 11), (2, 0), rowspan=4, colspan=10)
        axr = plt.subplot2grid((9, 11), (6, 0), rowspan=2, colspan=10, sharex=ax)
        axr2 = plt.subplot2grid((9, 11), (6, 10), rowspan=2, colspan=1)
    
        if trans == 'yes':
            axfilt = plt.subplot2grid((9, 11), (0, 0), rowspan=2, colspan=10, sharex=ax)  # for Filters
            
        plotted_spectro_label = False
        plotted_photo_label = False
        filter_colors = {}
    
        for obs, mod in zip(modif_data.values(), best_model.values()):
            obs_spectro, obs_photo = obs['spectro'], obs['photo']
            mod_spectro, mod_photo = mod['spectro'], mod['photo']
    
            ck_spectro = mod_spectro.get('ck', 1) if norm == 'yes' else 1
            ck_photo = mod_photo.get('ck', 1) if norm == 'yes' else 1
    
            # ===== SPECTROSCOPY =====
            if len(obs_spectro['wav']) > 0:
                obs_wav = np.array(obs_spectro['wav'])
                obs_flx = np.array(obs_spectro['flx']) / ck_spectro
                obs_flx, factor = u.scale_to_one_significant_digit(obs_flx)
                mod_flx = np.array(mod_spectro['flx']) / ck_spectro / (10**factor)
    
                if uncert == 'yes':
                    ax.errorbar(obs_wav, obs_flx, yerr=np.array(obs_spectro['err']) / ck_spectro, c='k', alpha=0.2)
                ax.plot(obs_wav, obs_flx, c='k')
                ax.plot(obs_wav, mod_flx, c=self.color_out, alpha=0.8)
    
                residuals = obs_flx * ck_spectro - mod_flx * ck_spectro
                sigma_res = np.nanstd(residuals)
                axr.plot(obs_wav, residuals / sigma_res, c=self.color_out, alpha=0.8)
                axr.axhline(0, color='k', alpha=0.5, linestyle='--')
                axr2.hist(residuals / sigma_res, bins=100, color=self.color_out, alpha=0.5, density=True, orientation='horizontal')
    
                if not plotted_spectro_label:
                    ax.plot([], [], c='k', label='Spectroscopic data')
                    ax.plot([], [], c=self.color_out, label='Spectroscopic model')
                    axr.plot([], [], c=self.color_out, label='Spectroscopic data-model')
                    axr2.hist([], bins=100, color=self.color_out, alpha=0.2, density=True, orientation='horizontal', label='density')
                    plotted_spectro_label = True
                    axr2.legend(frameon=False, handlelength=0)
    
            # ===== PHOTOMETRY =====
            if len(obs_photo['wav']) > 0:
                obs_wav = np.array(obs_photo['wav'])
                obs_flx = np.array(obs_photo['flx']) / ck_photo 
                obs_flx, factor = u.scale_to_one_significant_digit(obs_flx)
                mod_flx = np.array(mod_photo['flx']) / ck_photo / (10**factor)
    
                # If the user wants to plot the transmission filters:
                if trans == 'yes':
                    for pho in obs_photo['ins']:
                        if pho not in filter_colors:
                            filter_colors[pho] = 'black' 
                
                        filter_path = u.find_filter_file(pho)
                        filt = np.load(filter_path)
                        x = filt['x_filt']
                        y = filt['y_filt']
                
                        # Plot on axfilt (top subplot)
                        axfilt.plot(x, y, color=filter_colors[pho], alpha=0.6)
                
                        # Label at peak transmission
                        peak_idx = np.argmax(y)
                        peak_wav = x[peak_idx]
                        axfilt.text(peak_wav, y[peak_idx] + 0.05, pho, ha='center', fontsize=8, rotation=0, color='gray')

    
                if uncert == 'yes':
                    ax.errorbar(obs_wav, obs_flx, yerr=np.array(obs_photo['err']) / ck_photo, fmt='o', c='k', alpha=0.7)
                ax.plot(obs_wav, obs_flx, 'ko', alpha=0.7)
                ax.plot(obs_wav, mod_flx, 'o', color=self.color_out)
    
                residuals = obs_flx * ck_photo - mod_flx * ck_photo
                sigma_res = np.nanstd(residuals)
                axr.plot(obs_wav, residuals / sigma_res, 'o', c=self.color_out, alpha=0.8)
                axr.axhline(0, color='k', alpha=0.5, linestyle='--')
    
                if not plotted_photo_label:
                    ax.plot([], [], 'ko', label='Photometry data')
                    ax.plot([], [], 'o', c=self.color_out, label='Photometry model')
                    axr.plot([], [], 'o', c=self.color_out, label='Photometry data-model')
                    plotted_photo_label = True
    
        # Axis config
        if logx == 'yes':
            ax.set_xscale('log')
            axr.set_xscale('log')
            axfilt.set_xscale('log')
        if logy == 'yes':
            ax.set_yscale('log')
    
        axr.set_xlabel(r'Wavelength (µm)')
        ax.set_ylabel(rf'Flux ($10^{factor}$ W m-2 µm-1)')
        axr.set_ylabel(r'Residuals ($\sigma$)')
    
        # Filter axis config
        if trans == 'yes':
            axfilt.set_ylabel('Transmission')
            axfilt.tick_params(bottom=False, labelbottom=False)
            axfilt.set_ylim(0, 1.2)
    
        axr2.axis('off')
        ax.tick_params(bottom=False, labelbottom=False)
        ax.legend(frameon=False)
        axr.legend(frameon=False)
        
    
        return fig, ax, axr, axr2