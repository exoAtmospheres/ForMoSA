
from __future__ import print_function, division
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import corner
import xarray as xr
import pickle
import astropy.constants as cst
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

# Import ForMoSA
from ForMoSA.global_file import GlobFile
from ForMoSA.utils_spec import resolution_decreasing, continuum_estimate
from ForMoSA.nested_sampling.nested_modif_spec import modif_spec
from ForMoSA.nested_sampling.nested_modif_spec import doppler_fct
from ForMoSA.nested_sampling.nested_modif_spec import vsini_fct

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class ComplexRadar():
    '''
    Class to create Radar plots with asymmetric error bars.

    Author: Paulina Palma-Bifani
            Adapted from Damian Cummins: https://github.com/DamianCummins/statsbomb-football-event-visualisations/blob/master/Statsbomb%20Womens%20World%20Cup%202019%20visualisation.ipynb

    '''

    def __init__(self, fig, variables, ranges, n_ordinate_levels=6):
        '''
        Initialize class.

        Args:
            fig               (object): matplotlib figure object
            variables           (list): list of parameters to plot
            ranges       (list(tuple)): upper and lower limits for each parameters
            n_ordinate_levels    (int): (default = 6) number of gridlines in the plot
        Returns:
            None
        '''
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.9,0.9], polar=True, label = "axes{}".format(i)) for i in range(len(variables))]

        l, text = axes[0].set_thetagrids(angles, labels=variables)

        [[txt.set_fontweight('bold'),
              txt.set_fontsize(12),
              txt.set_position((0,-0.2))] for txt in text]

        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)

        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) for x in grid]

            gridlabel[0] = "" # clean up origin
            ax.set_rgrids(grid, labels=gridlabel,angle=angles[i])

            ax.set_ylim(*ranges[i])

        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]

    def plot(self, data, *args, **kw):
        '''
        Function to display the plot.

        Args:
            data       (list): best value for each parameter
            *args           : Variable length argument list.
            **kw            : Arbitrary keyword arguments.
        Returns:
            None
        '''
        sdata = self.scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        '''
        Add symmetric error bars to the plot.

        Args:
            data       (list): best value for each parameter
            *args           : Variable length argument list.
            **kw            : Arbitrary keyword arguments.
        Returns:
            None
        '''
        sdata = self.scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill_between(self, list_down, list_up, *args, **kw):
        '''
        Add asymmetric error bars to the plot.

        Args:
            list_down (list): list of lower error bars
            list_up   (list): list of upper error bars
            *args           : Variable length argument list.
            **kw            : Arbitrary keyword arguments.
        Returns:
            None
        '''
        sdata_down = self.scale_data(list_down, self.ranges)
        sdata_up = self.scale_data(list_up, self.ranges)
        self.ax.fill_between(self.angle,np.r_[sdata_down,sdata_down[0]], np.r_[sdata_up,sdata_up[0]], *args, **kw)

    def scale_data(self, data, ranges):
        '''
        Function to check that lower and upper limits are correctly ordered. It scales data[1:] to ranges[0]

        Args:
            data              (list): best value for each parameter
            ranges     (list(tuple)): upper and lower limits for each parameters
            *args           : Variable length argument list.
            **kw            : Arbitrary keyword arguments.
        Returns:
            None
        '''
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            if not np.isnan(d):
                assert (y1 <= d <= y2) or (y2 <= d <= y1)
        x1, x2 = ranges[0]
        d = data[0]
        sdata = [d]
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            if y1 > y2:
                d = _invert(d, (y1, y2))
                y1, y2 = y2, y1
            sdata.append((d-y1) / (y2-y1) * (x2 - x1) + x1)
        return sdata


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class PlottingForMoSA():
    '''
    Class containing all the plotting functionalities of ForMoSA.

    Author: Paulina Palma-Bifani, Simon Petrus, Matthieu Ravet and Allan Denis
    '''

    def __init__(self, config_file_path=None, color_out='blue', global_params=None):
        '''
        Initialize class by inheriting the global parameter class of ForMoSA.

        Args:
            config_file_path                (str): path to the config.ini file currently used
            color_out                       (str): color to use for the model
            global_params    (GlobFile, optional): already-initialized GlobFile object
        Returns:
            None
        '''
        if global_params is not None:
            self.global_params = global_params
        elif config_file_path is not None:
            self.global_params = GlobFile(config_file_path)
        else:
            raise ValueError("Either 'config_file_path' or 'global_params' must be provided.")
        self.color_out = color_out


    def _get_posteriors(self, burn_in=0):
        '''
        Function to get the posteriors, including luminosity derivation and Bayesian evidence logz.

        Args:
            None
        Returns:
            None
        '''
        with open(self.global_params.result_path + '/result_' + self.global_params.ns_algo + '.pic', 'rb') as open_pic:
            result = pickle.load(open_pic)
        self.samples = result['samples'][burn_in:]
        self.weights = result['weights'][burn_in:]

        # To test the quality of the fit
        self.logl=result['logl'][burn_in:]
        ind = np.where(self.logl==max(self.logl))
        self.theta_best = self.samples[ind][0]

        self.sample_logz    = round(result['logz'][0],1)
        self.sample_logzerr = round(result['logz'][1],1)
        self.outputs_string = 'logz = '+ str(self.sample_logz)+' ± '+str(self.sample_logzerr)

        ds = xr.open_dataset(self.global_params.model_path, decode_cf=False, engine='netcdf4')
        attrs = ds.attrs
        extra_parameters = [['r', 'R', r'(R$\mathrm{_{Jup}}$)'],
                            ['d', 'd', '(pc)'],
                            [r'$\alpha$', r'$\alpha$', ''],
                            ['rv', 'RV', r'(km.s$\mathrm{^{-1}}$)'],
                            ['av', 'Av', '(mag)'],
                            ['vsini', 'v.sin(i)', r'(km.s$\mathrm{^{-1}}$)'],
                            ['ld', 'limb darkening', ''],
                            ['bb_t', 'bb_t', '(K)'],
                            ['bb_r', 'bb_r', r'(R$\mathrm{_{Jup}}$)']
                            ]

        tot_list_param_title = []
        theta_index = []
        if self.global_params.par1[0] != 'NA' and self.global_params.par1[0] != 'constant':
            tot_list_param_title.append(attrs['title'][0] + ' ' + attrs['unit'][0])
            theta_index.append('par1')
        if self.global_params.par2[0] != 'NA' and self.global_params.par2[0] != 'constant':
            tot_list_param_title.append(attrs['title'][1] + ' ' + attrs['unit'][1])
            theta_index.append('par2')
        if self.global_params.par3[0] != 'NA' and self.global_params.par3[0] != 'constant':
            tot_list_param_title.append(attrs['title'][2] + ' ' + attrs['unit'][2])
            theta_index.append('par3')
        if self.global_params.par4[0] != 'NA' and self.global_params.par4[0] != 'constant':
            tot_list_param_title.append(attrs['title'][3] + ' ' + attrs['unit'][3])
            theta_index.append('par4')
        if self.global_params.par5[0] != 'NA' and self.global_params.par5[0] != 'constant':
            tot_list_param_title.append(attrs['title'][4] + ' ' + attrs['unit'][4])
            theta_index.append('par5')

        # Extra-grid parameters

        if self.global_params.r[0] != 'NA' and self.global_params.r[0] != 'constant':
            tot_list_param_title.append(extra_parameters[0][1] + ' ' + extra_parameters[0][2])
            theta_index.append('r')
        if self.global_params.d[0] != 'NA' and self.global_params.d[0] != 'constant':
            tot_list_param_title.append(extra_parameters[1][1] + ' ' + extra_parameters[1][2])
            theta_index.append('d')

        # - - - - - - - - - - - - - - - - - - - - -

        # Individual parameters / observation
        main_obs_path = self.global_params.main_observation_path

        if len(self.global_params.alpha) > 3: # If you want separate alpha for each observations
            for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
                if self.global_params.alpha[indobs*3] != 'NA' and self.global_params.alpha[indobs*3] != 'constant': # Check if the idobs is different from constant
                    tot_list_param_title.append(extra_parameters[2][1] + fr'$_{indobs}$' + ' ' + extra_parameters[2][2])
                    theta_index.append(f'alpha_{indobs}')
        else: # If you want 1 common alpha for all observations
            if self.global_params.alpha[0] != 'NA' and self.global_params.alpha[0] != 'constant':
                tot_list_param_title.append(extra_parameters[2][1] + ' ' + extra_parameters[2][2])
                theta_index.append('alpha')
        if len(self.global_params.rv) > 3: # If you want separate rv for each observations
            for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
                if self.global_params.rv[indobs*3] != 'NA' and self.global_params.rv[indobs*3] != 'constant': # Check if the idobs is different from constant
                    tot_list_param_title.append(extra_parameters[3][1] + fr'$_{indobs}$' + ' ' + extra_parameters[3][2])
                    theta_index.append(f'rv_{indobs}')
        else: # If you want 1 common rv for all observations
            if self.global_params.rv[0] != 'NA' and self.global_params.rv[0] != 'constant':
                tot_list_param_title.append(extra_parameters[3][1] + ' ' + extra_parameters[3][2])
                theta_index.append('rv')
        if len(self.global_params.vsini) > 4: # If you want separate vsini for each observations
            for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
                if self.global_params.vsini[indobs*4] != 'NA' and self.global_params.vsini[indobs*4] != 'constant': # Check if the idobs is different from constant
                    tot_list_param_title.append(extra_parameters[5][1] + fr'$_{indobs}$' + ' ' + extra_parameters[5][2])
                    theta_index.append(f'vsini_{indobs}')
        else: # If you want 1 common vsini for all observations
            if self.global_params.vsini[0] != 'NA' and self.global_params.vsini[0] != 'constant':
                tot_list_param_title.append(extra_parameters[5][1] + ' ' + extra_parameters[5][2])
                theta_index.append('vsini')
        if len(self.global_params.ld) > 3: # If you want separate ld for each observations
            for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
                if self.global_params.ld[indobs*3] != 'NA' and self.global_params.ld[indobs*3] != 'constant': # Check if the idobs is different from constant
                    tot_list_param_title.append(extra_parameters[6][1] + fr'$_{indobs}$' + ' ' + extra_parameters[6][2])
                    theta_index.append(f'ld_{indobs}')
        else: # If you want 1 common vsini for all observations
            if self.global_params.ld[0] != 'NA' and self.global_params.ld[0] != 'constant':
                tot_list_param_title.append(extra_parameters[6][1] + ' ' + extra_parameters[6][2])
                theta_index.append('ld')

        # - - - - - - - - - - - - - - - - - - - - -

        if self.global_params.av[0] != 'NA' and self.global_params.av[0] != 'constant':
            tot_list_param_title.append(extra_parameters[4][1] + ' ' + extra_parameters[4][2])
            theta_index.append('av')
        ## cpd bb
        if self.global_params.bb_t[0] != 'NA' and self.global_params.bb_t[0] != 'constant':
            tot_list_param_title.append(extra_parameters[7][1] + ' ' + extra_parameters[7][2])
            theta_index.append('bb_t')
        if self.global_params.bb_r[0] != 'NA' and self.global_params.bb_r[0] != 'constant':
            tot_list_param_title.append(extra_parameters[8][1] + ' ' + extra_parameters[8][2])
            theta_index.append('bb_r')
        self.theta_index = np.asarray(theta_index)

        posterior_to_plot = self.samples
        if self.global_params.r[0] != 'NA' and self.global_params.r[0] != 'constant':
            posterior_to_plot = []
            tot_list_param_title.append(r'log(L/L$\mathrm{_{\odot}}$)')

            for res, results in enumerate(self.samples):
                ind_theta_r = np.where(self.theta_index == 'r')
                r_picked = results[ind_theta_r[0]]

                lum = np.log10(4 * np.pi * (r_picked * cst.R_jup.value) ** 2 * results[0] ** 4 * cst.sigma_sb.value / cst.L_sun.value)
                results = np.concatenate((results, np.asarray(lum)))
                posterior_to_plot.append(results)

        self.posterior_to_plot = np.array(posterior_to_plot)
        self.posteriors_names = tot_list_param_title


    def plot_corner(self, levels_sig=[0.997, 0.95, 0.68], bins=100, quantiles=(0.16, 0.5, 0.84), figsize=(15,15)):
        '''
        Function to display the corner plot

        Args:
            levels_sig    (list): (default = [0.997, 0.95, 0.68]) 1, 2 and 3 sigma contour levels of the corner plot
            bins           (int): (default = 100) number of bins for the posteriors
            quantiles     (list): (default = (0.16, 0.5, 0.84)) mean +- sigma to report the posterior values
            burn_in        (int): (default = 0) number of steps to remove from the plot
        Returns:
            - fig         (object): matplotlib figure object
        '''
        print('ForMoSA - Corner plot')

        fig = plt.figure(figsize=figsize)
        fig = corner.corner(self.posterior_to_plot,
                            weights=self.weights,
                            labels=self.posteriors_names,
                            range=[0.999999 for p in self.posteriors_names],
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


        fig.supxlabel(self.outputs_string, va='top')

        return fig


    def plot_chains(self, figsize=(7,15), show_weights=True):
        '''
        Plot to check the convergence of the posterior chains.
        Multiple (sub-)axis plot.

        Args:
            figsize     (tuple): (default = (7, 15)) size of the plot
            show_weights (bool): True or False if you want to show the weights on the chain
        Returns:
            - fig  (object) : matplotlib figure object
            - ax   (object) : matplotlib axes objects
        '''
        print('ForMoSA - Posteriors chains for each parameter')

        col = int(len(self.posterior_to_plot[0][:])/2)+int(len(self.posterior_to_plot[0][:])%2)
        fig, axs = plt.subplots(col, 2, figsize=figsize)

        n=0
        for i in range(col):
            for j in range(2):
                axs[i, j].plot(self.posterior_to_plot[:,n], color=self.color_out, alpha=0.8)
                axs[i, j].set_ylabel(self.posteriors_names[n])
                if show_weights == True:
                    ax_w = axs[i, j].twinx()
                    ax_w.plot(self.weights, color='black', alpha=0.5)
                    ax_w.text(x=0, y=0.00005, s='weights')
                    if j == 0:
                        ax_w.set_yticks([])
                if self.posteriors_names[n]=='log(L/L$\\mathrm{_{\\odot}}$)':
                    pass
                else:
                    axs[i, j].axhline(self.theta_best[n],color='k',linestyle='--')
                if n == len(self.posteriors_names)-1:
                    break
                else:
                    n+=1

        return fig, axs


    def plot_radar(self, ranges, label='', quantiles=[0.16, 0.5, 0.84]):
        '''
        Radar plot to check the distribution of the parameters.
        Useful to compare different models.

        Args:
            ranges     (list(tuple)): upper and lower limits for each parameters
            label              (str): (default = '') label of the plot
            quantiles         (list): (default = (0.16, 0.5, 0.84)) mean +- sigma to report the posterior values
        Returns:
            - fig  (object) : matplotlib figure object
            - radar.ax   (object) : matplotlib radar class axes object

        '''
        print('ForMoSA - Radar plot')

        list_posteriors = []
        list_uncert_down = []
        list_uncert_up = []
        for l in range(len(self.posterior_to_plot[1,:])):
            q16, q50, q84 = corner.quantile(self.posterior_to_plot[:,l], quantiles)

            list_posteriors.append(q50)
            list_uncert_down.append(q16)
            list_uncert_up.append(q84)

        fig = plt.figure(figsize=(6, 6))
        radar = ComplexRadar(fig, self.posteriors_names, ranges)

        radar.plot(list_posteriors, 'o-', color=self.color_out, label=label)
        radar.fill_between(list_uncert_down,list_uncert_up, color=self.color_out, alpha=0.2)

        radar.ax.legend(loc='center', bbox_to_anchor=(0.5, -0.20),frameon=False, ncol=2)

        return fig, radar.ax


    def _get_spectra(self, theta):
        '''
        Function to get the data and best model asociated.

        Args:
            theta                   (list): best parameter values
        Returns:
            - modif_spec_LL  list(n-array): list containing the spectroscopic wavelength, spectroscopic fluxes of the data,
                                            spectroscopic errors of the data, spectroscopic fluxes of the model,
                                            photometric wavelength, photometric fluxes of the data, photometric errors of the data,
                                            spectroscopic fluxes of the model,
                                            planet transmission, star fluxes, systematics and scaling factors
        '''

        # Create a list for each spectra (obs and mod) for each observation + scaling factors
        modif_spec_LL = []

        for indobs, obs in enumerate(sorted(glob.glob(self.global_params.main_observation_path))):

            # Recovery of the observational dictionnary
            self.global_params.observation_path = obs
            obs_name = os.path.splitext(os.path.basename(self.global_params.observation_path))[0]
            obs_dict = np.load(os.path.join(self.global_params.result_path, f'spectrum_obs_{obs_name}.npz'), allow_pickle=True)


            # Recovery of the spectroscopy and photometry model
            path_grid_spectro = os.path.join(self.global_params.adapt_store_path, f'adapted_grid_spectro_{self.global_params.grid_name}_{obs_name}_nonan.nc')
            ds_spectro = xr.open_dataset(path_grid_spectro, decode_cf=False, engine='netcdf4')
            grid_spectro = ds_spectro['grid']
            path_grid_photo = os.path.join(self.global_params.adapt_store_path, f'adapted_grid_photo_{self.global_params.grid_name}_{obs_name}_nonan.nc')
            ds_photo = xr.open_dataset(path_grid_photo, decode_cf=False, engine='netcdf4')
            grid_photo = ds_photo['grid']

            # Emulator (if necessary)
            if self.global_params.emulator[0] != 'NA':
                # PCA or NMF
                mod_dict = dict(np.load(os.path.join(self.global_params.result_path, f'{self.global_params.emulator[0]}_mod_{obs_name}.npz'), allow_pickle=True))
            else:
                # Standard method
                mod_dict = {'wav_spectro': np.asarray(ds_spectro.coords['wavelength']), 'res_spectro': np.asarray(ds_spectro.attrs['res'])}
            ds_spectro.close()
            ds_photo.close()

            # Interpolating the model resolution
            interp_mod_to_obs = interp1d(mod_dict['wav_spectro'], mod_dict['res_spectro'], fill_value='extrapolate') # Interpolate model resolution onto the data
            mod_dict['res_spectro'] = interp_mod_to_obs(obs_dict['wav_spectro'])

            if self.global_params.par3[0] == 'NA':
                if len(obs_dict['wav_spectro']) != 0:
                    interp_spectro = np.asarray(grid_spectro.interp(par1=theta[0], par2=theta[1],
                                                            method=self.global_params.method, kwargs={"fill_value": "extrapolate"}))
                else:
                    interp_spectro = np.asarray([])
                if len(obs_dict['wav_photo']) != 0:
                    interp_photo = np.asarray(grid_photo.interp(par1=theta[0], par2=theta[1],
                                                            method=self.global_params.method, kwargs={"fill_value": "extrapolate"}))
                else:
                    interp_photo = np.asarray([])
            elif self.global_params.par4[0] == 'NA':
                if len(obs_dict['wav_spectro']) != 0:
                    interp_spectro = np.asarray(grid_spectro.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                            method=self.global_params.method, kwargs={"fill_value": "extrapolate"}))
                else:
                    interp_spectro = np.asarray([])
                if len(obs_dict['wav_photo']) != 0:
                    interp_photo = np.asarray(grid_photo.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                            method=self.global_params.method, kwargs={"fill_value": "extrapolate"}))
                else:
                    interp_photo = np.asarray([])
            elif self.global_params.par5[0] == 'NA':
                if len(obs_dict['wav_spectro']) != 0:
                    interp_spectro = np.asarray(grid_spectro.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            method=self.global_params.method, kwargs={"fill_value": "extrapolate"}))
                else:
                    interp_spectro = np.asarray([])
                if len(obs_dict['wav_photo']) != 0:
                    interp_photo = np.asarray(grid_photo.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            method=self.global_params.method, kwargs={"fill_value": "extrapolate"}))
                else:
                    interp_photo = np.asarray([])
            else:
                if len(obs_dict['wav_spectro']) != 0:
                    interp_spectro = np.asarray(grid_spectro.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            par5=theta[4],
                                                            method=self.global_params.method, kwargs={"fill_value": "extrapolate"}))
                else:
                    interp_spectro = np.asarray([])
                if len(obs_dict['wav_photo']) != 0:
                    interp_photo = np.asarray(grid_photo.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            par5=theta[4],
                                                            method=self.global_params.method, kwargs={"fill_value": "extrapolate"}))
                else:
                    interp_photo = np.asarray([])

            # Recreate the flux array
            if self.global_params.emulator[0] != 'NA':
                if self.global_params.emulator[0] == 'PCA':
                    if len(mod_dict['vectors_spectro']) != 0:
                        flx_mod_spectro = (mod_dict['flx_mean_spectro']+mod_dict['flx_std_spectro'] * (interp_spectro[1:] @ mod_dict['vectors_spectro'])) * interp_spectro[0][np.newaxis]
                    else:
                        flx_mod_spectro = np.asarray([])
                    if len(mod_dict['vectors_photo']) != 0:
                        flx_mod_photo = (mod_dict['flx_mean_photo']+mod_dict['flx_std_photo'] * (interp_photo[1:] @ mod_dict['vectors_photo'])) * interp_photo[0][np.newaxis]
                    else:
                        flx_mod_photo = np.asarray([])
                elif self.global_params.emulator[0] == 'NMF':
                    if len(mod_dict['vectors_spectro']) != 0:
                        flx_mod_spectro = interp_spectro[:] @ mod_dict['vectors_spectro']
                    else:
                        flx_mod_spectro = np.asarray([])
                    if len(mod_dict['vectors_photo']) != 0:
                        flx_mod_photo = interp_photo[:] @ mod_dict['vectors_photo']
                    else:
                        flx_mod_photo = np.asarray([])
            else:
                flx_mod_spectro = interp_spectro
                flx_mod_photo = interp_photo

            # Modification of the synthetic spectrum with the extra-grid parameters
            modif_spec_LL.append(modif_spec(self.global_params, theta, self.theta_index,
                                      obs_dict, 
                                      flx_mod_spectro, flx_mod_photo, 
                                      mod_dict['wav_spectro'], mod_dict['res_spectro'],
                                      indobs=indobs))

        return modif_spec_LL


    def _get_full_spectra(self, theta, grid_used='original', wav_bounds=[], res=1000, re_interp=False, int_method="linear", indobs=0):
        '''
        Extract a model spectrum from a grid at a given theta, resolution and wavelength extent.

        Args:
            theta                       (list): best parameter values
            grid_used                    (str): (default = 'original') Path to the grid from where to extract the spectrum. If 'original', the current grid will be used.
            wav_bounds                  (list): (default = []) Desired wavelength range. If [] max and min values of the model wavelength range will be use to create the final wavelength range.
            res                          (int): (default = 1000) Spectral resolution (at Nyquist).
            re_interp                (boolean): (default = False). Option to reinterpolate or not the grid.
            int_method                   (str): (default = "linear") Interpolation method for the grid (if reinterpolated).
        Returns:
            - wav_final                (array): Wavelength array of the full model
            - flx_final                (array): Flux array of the full model
            - ck                       (float): Scaling factor of the full model
        '''

        obs_dict, _, _, _, _, ck = self._get_spectra(theta)[indobs]
        # WARNING : In case of multiple spectra, it is possible to work with different scaling factors. Here we only take the scaling factor of the first spectrum
        #in the MOSAIC (used for the plot_fit)

        # Recover the original grid
        if grid_used == 'original':
            path_grid = self.global_params.model_path
        else:
            path_grid = grid_used

        # Recover the original grid
        ds = xr.open_dataset(path_grid, decode_cf=False, engine="netcdf4")

        # Possibility of re-interpolating holes if the grid contains to much of them (WARNING: Very long process)
        if re_interp == True:
            print('-> The possible holes in the grid are (re)interpolated: ')
            for key_ind, key in enumerate(ds.attrs['key']):
                print(str(key_ind+1) + '/' + str(len(ds.attrs['key'])))
                ds = ds.interpolate_na(dim=key, method=self.global_params.method, fill_value="extrapolate", limit=None,
                                            max_gap=None)

        wav_mod_nativ = ds["wavelength"].values
        res_mod_nativ = np.asarray(ds.attrs['res'])
        grid = ds['grid']
        # Interpolating grid's resolution
        interp_mod_to_obs = interp1d(wav_mod_nativ, res_mod_nativ, fill_value='extrapolate') # Interpolate model resolution onto the data
        res_mod_obs_nativ = interp_mod_to_obs(obs_dict['wav_spectro'])
        ds.close()

        if self.global_params.par3[0] == 'NA':
            flx_mod_nativ = grid.interp(par1=theta[0], par2=theta[1],method=int_method, kwargs={"fill_value": "extrapolate"})
        elif self.global_params.par4[0] == 'NA':
            flx_mod_nativ = grid.interp(par1=theta[0], par2=theta[1], par3=theta[2],method=int_method, kwargs={"fill_value": "extrapolate"})
        elif self.global_params.par5[0] == 'NA':
            flx_mod_nativ = grid.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],method=int_method, kwargs={"fill_value": "extrapolate"})
        else:
            flx_mod_nativ = grid.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],par5=theta[4],method=int_method, kwargs={"fill_value": "extrapolate"})

        # Convert everything into array
        wav_mod_nativ = np.asarray(wav_mod_nativ, dtype=float)
        flx_mod_nativ = np.asarray(flx_mod_nativ, dtype=float)

        # Apply theta modifications
        modif_spec_LL = modif_spec(self.global_params, theta, self.theta_index,
                                      obs_dict, 
                                      flx_mod_nativ, [], 
                                      wav_mod_nativ, res_mod_obs_nativ,
                                      indobs=indobs)
        
        # Get back the modified model flux to compute the logL
        flx_mod_modif = modif_spec_LL[3]

        # Check that the imputed resolution is lower than the grid's, otherwise will use the grid resolution
        if res > np.min(res_mod_nativ):
            res = np.min(res_mod_nativ)
            print(f"WARNING: The requested resolution is too high for this grid (Rgrid = {np.round(np.min(res_mod_nativ), 2)})")
            print('replacing it...')
            print()

        # Decrease the resolution (simple smoothing)
        dwav = np.abs(obs_dict['wav_spectro'] - np.roll(obs_dict['wav_spectro'], 1))
        dwav[0] = dwav[1]
        sigma_lsf = 1. / res / (2. * np.sqrt(2. * np.log(2.)))
        sigma_lsf_gauss_filter = np.mean(sigma_lsf / dwav)
        flx_mod_modif = gaussian_filter(flx_mod_modif,
                                    sigma=sigma_lsf_gauss_filter,
                                    mode='nearest')

        # Resample the final spectrum
        if len(wav_bounds) == 0:
            wav_bounds = [min(wav_mod_nativ), max(wav_mod_nativ)]
        wav_final = [wav_bounds[0]]  # Start with the minimum wavelength
        dwav = [0]
        while wav_final[-1] < wav_bounds[1]:
            dwav_unit = wav_final[-1] / (2 * res)  # Compute spacing (Nyquist sampling)
            dwav.append(dwav_unit)
            wav_final.append(wav_final[-1] + dwav_unit)
        wav_final = np.array(wav_final)[(wav_mod_nativ[0] < wav_final) * (wav_final < wav_mod_nativ[-1])] # Make sure you don't extrapolate

        # Interpolate
        inter_func = interp1d(wav_mod_nativ, flx_mod_modif, fill_value='extrapolate')
        flx_final = inter_func(wav_final)

        return wav_final, flx_final, ck



    def plot_fit(self, figsize=(13, 7), uncert='no', trans='no', logx='no', logy='no', norm='no'):
        '''
        Plot the best fit comparing with the data.

        Args:
            figsize    (tuple): (default = (10, 5)) Size of the plot
            uncert     (str): (default = no) 'yes' or 'no' to plot spectra with associated error bars
            trans      (str): (default = no) 'yes' or 'no' to plot transmision curves for photometry
            logx       (str): (default = no) 'yes' or 'no' to plot the wavelength in log scale
            logy       (str): (default = no) 'yes' or 'no' to plot the flux in log scale
            norm       (str): (default = no) 'yes' or 'no' to plot the normalized spectra
        Returns:
            - fig   (object): matplotlib figure object
            - ax    (object): matplotlib axes objects, main spectra plot
            - axr   (object): matplotlib axes objects, residuals
            - axr2  (object): matplotlib axes objects, right side density histogram
        '''

        print('ForMoSA - Best fit and residuals plot')

        # Figure setup
        fig = plt.figure(figsize=figsize)
        fig.tight_layout()
        size = (7,11)
        ax = plt.subplot2grid(size, (0, 0), rowspan=5, colspan=10)
        axr = plt.subplot2grid(size, (5, 0), rowspan=2, colspan=10, sharex=ax)
        axr2 = plt.subplot2grid(size, (5, 10), rowspan=2, colspan=1)

        # Indices for plot
        iobs_spectro = 0
        iobs_photo = 0

        # Iterate on each obs
        for indobs, obs in enumerate(sorted(glob.glob(self.global_params.main_observation_path))):
            # Get back spectra
            obs_dict, flx_mod_spectro, flx_mod_photo, _, _, ck = self._get_spectra(self.theta_best)[indobs]

            # Scale or not in absolute flux
            if norm != 'yes':
                ck = 1

            # Spectroscopic part
            if len(obs_dict['wav_spectro']) != 0:
                iobs_spectro += 1
                iobs_photo += 1
                if uncert=='yes':
                    ax.errorbar(obs_dict['wav_spectro'], obs_dict['flx_spectro']/ck, yerr=obs_dict['err_spectro']/ck, c='k', alpha=0.2)
                ax.plot(obs_dict['wav_spectro'], obs_dict['flx_spectro']/ck, c='k')
                ax.plot(obs_dict['wav_spectro'], flx_mod_spectro/ck, c=self.color_out, alpha=0.8)
                    
                # Residuals
                residuals = obs_dict['flx_spectro'] - flx_mod_spectro
                sigma_res = np.nanstd(residuals) # Replace np.std by np.nanstd if nans are in the array to ignore them
                axr.plot(obs_dict['wav_spectro'], residuals/sigma_res, c=self.color_out, alpha=0.8)
                axr.axhline(y=0, color='k', alpha=0.5, linestyle='--')
                axr2.hist(residuals/sigma_res, bins=100, color=self.color_out, alpha=0.5, density=True, orientation='horizontal')

                if indobs == iobs_spectro-1:
                    # Add labels out of the loops
                    ax.plot(obs_dict['wav_spectro'], np.empty(len(obs_dict['wav_spectro']))*np.nan, c='k', label='Spectroscopic data')
                    ax.plot(obs_dict['wav_spectro'], np.empty(len(obs_dict['wav_spectro']))*np.nan, c=self.color_out, label='Spectroscopic model')
                    axr.plot(obs_dict['wav_spectro'], np.empty(len(obs_dict['wav_spectro']))*np.nan, c=self.color_out, label='Spectroscopic data-model')
                    axr2.hist(residuals/sigma_res, bins=100 , color=self.color_out, alpha=0.2, density=True, orientation='horizontal', label='density')
                    iobs_spectro = -1
                axr2.legend(frameon=False,handlelength=0)

            # Photometry part
            if len(obs_dict['wav_photo']) != 0:
                iobs_photo += 1
                iobs_spectro += 1
                # If you want to plot the transmission filters
                if trans == 'yes':
                    for pho_ind, pho in enumerate(obs_dict['ins_photo']):
                        path_list = __file__.split("/")[:-2]
                        separator = '/'
                        filter_pho = np.load(separator.join(path_list) + '/phototeque/' + pho + '.npz')
                        ax.fill_between(filter_pho['x_filt'], filter_pho['y_filt']*0.8*min(obs_dict['flx_photo']/ck),color=self.color_out, alpha=0.3)
                        ax.text(np.mean(filter_pho['x_filt']), np.mean(filter_pho['y_filt']*0.4*min(obs_dict['flx_photo']/ck)), pho, horizontalalignment='center', c='gray')

                if uncert=='yes':
                    ax.errorbar(obs_dict['wav_photo'], obs_dict['flx_photo']/ck, yerr=obs_dict['err_photo']/ck, c='k', fmt='o', alpha=0.7)    
                ax.plot(obs_dict['wav_photo'], obs_dict['flx_photo']/ck, 'ko', alpha=0.7)
                ax.plot(obs_dict['wav_photo'], flx_mod_photo/ck, 'o', color=self.color_out)

                # Residuals
                residuals_phot = obs_dict['flx_photo'] - flx_mod_photo
                sigma_res = np.std(residuals_phot)
                axr.plot(obs_dict['wav_photo'], residuals_phot/sigma_res, 'o', c=self.color_out, alpha=0.8)
                axr.axhline(y=0, color='k', alpha=0.5, linestyle='--')

                if indobs == iobs_photo-1:
                    # Add labels out of the loops
                    ax.plot(obs_dict['wav_photo'], np.empty(len(obs_dict['wav_photo']))*np.nan, 'ko', label='Photometry data')
                    ax.plot(obs_dict['wav_photo'], np.empty(len(obs_dict['wav_photo']))*np.nan, 'o', c=self.color_out, label='Photometry model')
                    axr.plot(obs_dict['wav_photo'], np.empty(len(obs_dict['wav_photo']))*np.nan, 'o', c=self.color_out, label='Photometry data-model')

                    iobs_photo = -1

        # Set xlog-scale
        if logx == 'yes':
            ax.set_xscale('log')
            axr.set_xscale('log')

        # Set xlog-scale
        if logy == 'yes':
            ax.set_yscale('log')

        # Labels
        axr.set_xlabel(r'Wavelength (µm)')
        if norm != 'yes':
            ax.set_ylabel(r'Flux (W m-2 µm-1)')
        else:
            ax.set_ylabel(r'Normalised flux (W m-2 µm-1)')
        axr.set_ylabel(r'Residuals ($\sigma$)')

        axr2.axis('off')
        ax.tick_params(labelbottom=False, bottom=False)
        ax.legend(frameon=False)
        axr.legend(frameon=False)

        return fig, ax, axr, axr2


    def plot_HiRes_comp_model(self, figsize=(10, 5), indobs=0):
        '''
        Specific function to plot the best fit comparing with the data for high-resolution spectroscopy.

        Args:
            figsize                   (tuple): (default = (10, 5)) Size of the plot
            indobs                      (int): Index of the current observation loop
        Returns:      
            - fig1, ax1              (object): matplotlib figure object
            - fig, ax                (object): matplotlib axes objects
            - flx_obs_broadened       (array): Flux of the observation with all fitted contributions removed + broadened at vsini
        '''
        print('ForMoSA - Planet model and data')


        # Get back spectra
        obs_dict, flx_mod_spectro, _, _, _, _ = self._get_spectra(self.theta_best)[indobs]

        # Prepare plot
        fig1, ax1 = plt.subplots(1, 1, figsize = figsize)
        fig, ax = plt.subplots(1, 1, figsize = figsize)

        # Spectroscopic part
        if len(obs_dict['wav_spectro']) != 0:

            # Remove contributions to show planetary signal if necessary
            flx_obs_calib = obs_dict['flx_spectro'] - obs_dict['system'] - obs_dict['star_flx']
            flx_mod_calib = flx_mod_spectro- obs_dict['system'] - obs_dict['star_flx']

            # Compute intrinsic resolution of the data because of the v.sini (if defined)
            try:
                if len(self.global_params.vsini) > 4:
                    new_res = 3.0*1e5 / (self.theta_best[self.theta_index == f'vsini_{indobs}'])
                else:
                    new_res = 3.0*1e5 / (self.theta_best[self.theta_index == 'vsini'])
                new_res = new_res * np.ones(len(obs_dict['wav_spectro']))
                flx_obs_broadened = resolution_decreasing(obs_dict['wav_spectro'], flx_obs_calib, obs_dict['res_spectro'], obs_dict['wav_spectro'], new_res)
            except:
                flx_obs_broadened = flx_obs_calib

            ax.plot(obs_dict['wav_spectro'], flx_obs_broadened, c='k')
            ax.plot(obs_dict['wav_spectro'], flx_mod_calib, c='r')

            ax.set_xlabel(r'wavelength ($\mu$m)')
            ax.set_ylabel('Flux (ADU)')

            ax1.plot(obs_dict['wav_spectro'], flx_obs_calib, c='k')
            ax1.plot(obs_dict['wav_spectro'], flx_mod_calib, c = 'r')

            if self.global_params.hc_type[indobs % len(self.global_params.hc_type)] != 'NA':
                legend_data = 'data - star'
            else:
                legend_data = 'data'

            ax.legend([legend_data, 'planet model'])
            ax.tick_params(axis='both')


        ax1.legend([legend_data, "planet model"])
        ax1.set_xlabel('wavelength ($ \mu $m)')
        ax1.tick_params(axis='both')

        return fig1, ax1, fig, ax, flx_obs_broadened


    def compute_ccf_single_rv(self, rv, wav_mod, flx_mod, flx_mod_no_rv, res_mod_obs, wav_obs, flx_obs, res_obs, transm_obs, Sf, indobs):
        '''
        Compute a cross-correlation coefficient for a single rv. This function is used for the parallelised ccf computation
        

        Args:
            rv            (float) : rv value to apply to the model
            wav_mod       (ndarray ) : wavelength grid of the model
            flx_mod       (ndarray) : flux of the model
            flx_mod_no_rv (ndarray) : flx of the model at 0 rv (used for autocorrelation)
            res_mod_obs   (ndarray) : resolution of the model interpolated onto obs_wav
            wav_obs       (ndarray) : wavelength grid of the observation
            flx_obs       (ndarray) : flux of the observation
            res_obs       (ndarray) : resolution of the observation
            transm_obs    (ndarray) : atmospheric and instrumental transmission
            Sf            (float) : L2 norm of the observation
            indobs        (int) : Index of the current observation loop

        Returns:
            - ccf  (float) : cross-correlation coefficient
            - acf  (float) : autocorrelation coefficient
            - logL (float) : logL value
        '''
        
        wav_mod_rv, flx_mod_rv = doppler_fct(wav_mod, flx_mod, rv)
        flx_mod_rv = resolution_decreasing(wav_mod_rv, flx_mod_rv, res_mod_obs, wav_obs, res_obs)
        flx_cont_mod_rv = continuum_estimate(wav_obs, flx_mod_rv, res_obs, self.global_params.wav_cont[indobs % len(self.global_params.wav_cont)], float(self.global_params.res_cont[indobs % len(self.global_params.res_cont)]))
        flx_mod_rv -= flx_cont_mod_rv
        flx_mod_rv *= transm_obs
        
        # Normalize the model to make it comparable to the data in terms of flux
        flx_mod_rv /= np.sqrt(np.nansum(flx_mod_rv**2))
        ccf = np.nansum(flx_mod_rv * flx_obs)    # Cross correlation function
        acf = np.nansum(flx_mod_rv * flx_mod_no_rv)   # Auto correlation function

        Sg = np.nansum(np.square(flx_mod_rv))
        R = np.nansum(flx_obs * flx_mod_rv)
        C2 = R**2 / (Sf * Sg)
        logL = -len(flx_obs) / 2 * np.log(1 - C2)
        
        return ccf, acf, logL


    def plot_ccf(self, rv_grid = [-300,300], rv_step = 0.5, figsize = (10,5), window_normalisation = 100, continuum_res = 500, vsini = [], wav_mod_nativ=[], flx_mod_nativ=[], res_mod_nativ=[], indobs=0, plot=True, map_rv_vsini = False, flx_obs = [], wav_obs = [], res_obs = [], transm_obs = []):
        '''
        Plot the cross-correlation function. It is used for high resolution spectroscopy.

        Args:
            figsize                   (tuple): (default = (10, 5)) Size of the plot
            rv_grid                    (list): (default = [-300,300]) Maximum and minumum values of the radial velocity shift (in km/s)
            rv_step                   (float): (default = 0.5) Radial velocity shift steps (in km/s)
            figsize                   (tuple): (default = (10,5)) Size of the figure to plot
            window_normalisation        (int): (default = 100) Window used to exclude around the peak of the CCF for noise estimation
            vsini                     (float): (default = []) v.sin(i) used to apply to the model (in the case the user wants to apply another v.sin(i) than the v.sin(i) estimated by the NS)     
            wav_mod_nativ             (array): (default = []) Wavelength of the model to cross-correlate with the data in the case the user wants to use the rv_vsini map function or a different model (individual molecule for example)
            flx_mod_nativ             (array): (default = []) Flux of the model to cross-correlate with the data in the case the user wants to use the rv_vsini map function or a different model (individual molecule for example)
            res_mod_nativ             (array): (default = []) Resolution of the model to cross-correlate with the data in the case the user wants to use the rv_vsini map function or a different model (individual molecule for example)
            indobs                      (int): (default = 0) Index of the current observation loop
            plot                       (bool): (default = True) Whether to plot the ccf
            map_rv_vsini               (bool): (default = False) Whether the user wants to use the rv_vsini map function
            flx_obs                   (array): (default = []) Data in the case the user wants to use the rv_vsini map function. This avoids repeating the same operation for each v.sini defined by the v.sini grid and sames some time
            wav_obs                   (array): (default = []) Wavelength in the case the user wants to use the rv_vsini map function. This avoids repeating the same operation for each v.sini defined by the v.sini grid and sames some time
            res_obs                   (array): (default = []) Resolution in the case the user wants to use the rv_vsini map function. This avoids repeating the same operation for each v.sini defined by the v.sini grid and sames some time
            transm_obs                (array): (default = []) Transmission in the case the user wants to use the rv_vsini map function. This avoids repeating the same operation for each v.sini defined by the v.sini grid and sames some time
        Returns:
            - fig1                   (object): matplotlib figure object
            - ax1                    (object): matplotlib axes objects
            - rv_grid                  (list): Radial velocity grid
            - ccf_norm                 (list): Cross-correlation function
            - acf_norm                 (list): Auto-correlation function
        '''
        print('ForMoSA - CCF plot')

        # Gauss function used to estimate the peak of the radial velocity
        def gauss(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))

        rv_grid = np.arange(rv_grid[0], rv_grid[1], rv_step)
        
        # This condition is used to retrieve the obseervations and the nativ model. 
        # In the case the user uses the rv_vsini_map function, the observations and nativ models are already defined as inputs of the function
        # This saves some time by avoiding the repetition of this set of operations at each v.sini of the v.sini grid in the rv_vsini_map function
        if not(map_rv_vsini):
            # In this case, we extract the obs
            obs_dict, _, _, _, _, _ = self._get_spectra(self.theta_best)[indobs]
            wav_obs, flx_obs, star_flx_obs, system_obs, res_obs, transm_obs = obs_dict['wav_spectro'], obs_dict['flx_spectro'], obs_dict['star_flx'], obs_dict['system'], obs_dict['res_spectro'], obs_dict['transm']

            # Retrieve data to cross correlate the model with
            flx_obs = flx_obs - star_flx_obs - system_obs # If star_flx and/or system are not define, this won't change anything
                
            # Normalize the data
            flx_obs /= np.sqrt(np.sum(flx_obs**2))
            
            # Second step, we retrieve the native model at rv and v.sini = 0
            theta_best = np.copy(self.theta_best)
            try:
                if len(self.global_params.rv) > 3:
                    theta_best[self.theta_index == f'rv_{indobs}'] = 0
                else:
                    theta_best[self.theta_index == 'rv'] = 0
            except:
                pass
            try:
                if len(self.global_params.vsini) > 4:
                    theta_best[self.theta_index == f'vsini_{indobs}'] = 0
                else:
                    theta_best[self.theta_index == 'vsini'] = 0
            except:
                pass
            # Recover the grid
            ds = xr.open_dataset(self.global_params.model_path, decode_cf=False, engine="netcdf4")
            wav_mod_nativ = ds["wavelength"].values
            res_mod_nativ = np.asarray(ds.attrs['res'])
            _, _, _, flx_mod_nativ, _, _ = self._get_spectra(self.theta_best)[indobs]
            
            # This condition arrises if the user does not use the rv_vsini_map function AND does not want to apply a specific vsini to the template
            # in that case, we use the best v.sini infered by the nested sampling
            if vsini == []:   
                vsini = self.theta_best[self.theta_index == 'vsini']

        # Second.5, interpolate the resolution of the model
        interp_mod_to_obs = interp1d(wav_mod_nativ, res_mod_nativ, fill_value='extrapolate')
        res_mod_obs = interp_mod_to_obs(wav_obs)
    
        # Third sted, we apply rotational broadening]
        flx_mod_vsini, res_mod_vsini = vsini_fct(wav_mod_nativ, flx_mod_nativ, res_mod_obs, 0.6, vsini, self.global_params.vsini[indobs*4 + 3 % len(self.global_params.vsini)])  # We consider the limb darkening to be fixed at 0.6

        # Finally, we generate the model at rv = 0 (for auto correlation)
        self.global_params.continuum_sub[indobs] = continuum_res
        flx_mod_vsini_no_rv = resolution_decreasing(wav_mod_nativ, flx_mod_vsini, res_mod_vsini, wav_obs, res_obs)
        flx_cont_mod_vsini_no_rv = continuum_estimate(wav_obs, flx_mod_vsini_no_rv, res_obs, self.global_params.wav_cont[indobs % len(self.global_params.wav_cont)], float(self.global_params.res_cont[indobs % len(self.global_params.res_cont)]))
        flx_mod_vsini_no_rv -= flx_cont_mod_vsini_no_rv
        
        # Multiply by telluric and instrumental transmission (if any)
        if len(transm_obs > 0):
            flx_mod_vsini_no_rv *= transm_obs

        flx_mod_vsini_no_rv /= np.sqrt(np.sum(flx_mod_vsini_no_rv**2))
    
        ccf = np.zeros(len(rv_grid))
        acf = np.zeros(len(rv_grid))
        logL = np.zeros(len(rv_grid))
        
        Sf = np.nansum(np.square(flx_obs))
  
        # compute CCF with pool of workers
        with ThreadPool(processes=mp.cpu_count()) as pool:
            pbar = tqdm(total=len(rv_grid), leave=False)

            def update(*a):
                pbar.update()
                
            # Loop in rv
            tasks = []
            for rv in tqdm(rv_grid):
                tasks.append(pool.apply_async(compute_ccf_single_rv, args=(self, rv, wav_mod_nativ, flx_mod_vsini, flx_mod_vsini_no_rv, res_mod_vsini, wav_obs, flx_obs, res_obs, transm_obs, Sf, indobs), callback=update))
                
            pool.close()
            pool.join()
            
            # extract results
            ccf = np.zeros(rv_grid.size)
            acf = np.zeros(rv_grid.size)
            logL = np.zeros(rv_grid.size)
            for irv, task in enumerate(tasks):
                res = task.get()
                ccf[irv] = res[0]
                acf[irv] = res[1]
                logL[irv] = res[2]
                
            
        # Rescaling cross-correlation function to estimate a SNR
        acf_norm = acf - np.median(acf[(np.abs(rv_grid) > window_normalisation)])
        ccf_norm = ccf - np.median(ccf[(np.abs(rv_grid-rv_grid[np.argmax(ccf)]) > window_normalisation)])
        ccf_noise = np.std(ccf_norm[(np.abs(rv_grid-rv_grid[np.argmax(ccf)]) > window_normalisation)])
        ccf_norm = ccf_norm / ccf_noise
        #sigma_ccf = sigma_ccf / ccf_noise
        
        if (plot) and (not(map_rv_vsini)):  
            # Rescaling autocorrelation function to make it comparable with cross-correlation function
            acf_norm = acf_norm / np.max(acf_norm) * np.max(ccf_norm)
            ind_curve_fit = np.abs(rv_grid - rv_grid[np.argmax(ccf_norm)]) < 15
            rv = rv_grid[np.argmax(ccf_norm)] 
            p0 = [ccf_norm[np.argmax(ccf_norm)], rv, self.theta_best[self.theta_index=='vsini'][0]]
            popt, pcov = curve_fit(gauss, rv_grid[ind_curve_fit], ccf_norm[ind_curve_fit], p0=p0)
             
            acf = gauss(rv_grid, popt[0], popt[1], popt[2])
           
            fig1, ax1 = plt.subplots(1,1, figsize=figsize)
            ax1.plot(rv_grid, ccf_norm, label = 'ccf')
            ax1.plot(rv_grid + popt[1], acf_norm)
            ax1.axvline(x = popt[1], linestyle = '--', c='C3')
            ax1.set_xlabel('RV (km/s)')
            ax1.set_ylabel('S/N')
            ax1.legend(['ccf', 'acf'])
            
            print(f'SNR = {np.nanmax(ccf_norm):.1f}, RV = {popt[1]:.1f} km/s')
            return fig1, ax1, rv_grid, ccf_norm, acf_norm, ccf_noise, logL
        
        elif (not(plot)) and (not(map_rv_vsini)):
            # Rescaling autocorrelation function to make it comparable with cross-correlation function
            acf_norm = acf_norm / np.max(acf_norm) * np.max(ccf_norm)
            ind_curve_fit = np.abs(rv_grid - rv_grid[np.argmax(ccf_norm)]) < 15
            rv = rv_grid[np.argmax(ccf_norm)] 
            p0 = [ccf_norm[np.argmax(ccf_norm)], rv, self.theta_best[self.theta_index=='vsini'][0]]
            popt, pcov = curve_fit(gauss, rv_grid[ind_curve_fit], ccf_norm[ind_curve_fit], p0=p0)
             
            acf = gauss(rv_grid, popt[0], popt[1], popt[2])
            
            print(f'SNR = {np.nanmax(ccf_norm):.1f}, RV = {popt[1]:.1f} km/s')
            return rv_grid, ccf_norm, acf_norm, ccf_noise, logL
            
        else:
            return rv_grid, logL


    def plot_map_rv_vsini(self, rv_grid = [-100,100], rv_step = 1.0, vsini_grid=[1,100], vsini_step = 1.0, wav_mod_nativ=[], flx_mod_nativ=[], res_mod_nativ=[], indobs=0, continuum_res=500):
        '''
        Plot a RV v.sini map. It is used for high resolution spectroscopy

        Args:
            rv_grid                 (list): (default = [-100,100]) Maximum and minumum values of the radial velocity shift (in km/s)
            rv_step                 (float): (default = 0.5) Radial velocity shift steps (in km/s)
            vsini_grid              (list): (default = [1,100]) Maximum and minimum values of the v.sini broadening (in km/s)
            vsini_step              (float): (default = 1.0) v.sini broadening steps (in km/s)
            wav_mod_nativ           (array): (default = []) Wavelength of the model to cross-correlate with the data in the case the user wants to use a different model (individual molecule for example)
            flx_mod_nativ           (array): (default = []) Flux of the model to cross-correlate with the data in the case the user wants to use a different model (individual molecule for example)
            res_mod_nativ           (array): (default = []) Resolution of the model to cross-correlate with the data in the case the user wants to use a different model (individual molecule for example)
        Returns:
            - ccf_map               (ndarray): 2D cross correlation map of rv and v.sini
            - fig                   (object) : matplotlib figure object
            - ax                    (object) : matplotlib axes objects
        '''
        
        print('ForMoSA - RV-vsini mapping plot')
        
        vsini_grid = np.arange(vsini_grid[0], vsini_grid[1], vsini_step)
        logL_map = np.empty((len(vsini_grid), int((rv_grid[1] - rv_grid[0]) / rv_step)))
        
        # First step, we retrieve the star and systematics contaminations associated to the best model (if any)
        obs_dict, _, _, _, _, _ = self._get_spectra(self.theta_best)[indobs]
        wav_obs, flx_obs, star_flx_obs, system_obs, res_obs, transm_obs = obs_dict['wav_spectro'], obs_dict['flx_spectro'], obs_dict['star_flx'], obs_dict['system'], obs_dict['res_spectro'], obs_dict['transm']

        # Retrieve data to cross correlate the model with
        flx_obs = flx_obs - star_flx_obs - system_obs
            
        # Normalize the data
        flx_obs /= np.sqrt(np.sum(flx_obs**2))
        
        # Second step, we retrieve the native model at rv and v.sini = 0
        if (len(flx_mod_nativ) == 0) and (len(res_mod_nativ) == 0):
            theta_best = np.copy(self.theta_best)
            try:
                if len(self.global_params.rv) > 3:
                    theta_best[self.theta_index == f'rv_{indobs}'] = 0
                else:
                    theta_best[self.theta_index == 'rv'] = 0
            except:
                pass
            try:
                if len(self.global_params.vsini) > 4:
                    theta_best[self.theta_index == f'vsini_{indobs}'] = 0
                else:
                    theta_best[self.theta_index == 'vsini'] = 0
            except:
                pass
            # Recover the grid
            ds = xr.open_dataset(self.global_params.model_path, decode_cf=False, engine="netcdf4")
            wav_mod_nativ = ds["wavelength"].values
            res_mod_nativ = np.asarray(ds.attrs['res'])
            _, _, _, flx_mod_nativ, _, _ = self._get_spectra(self.theta_best)[indobs]
        else:
            pass
        
        
        for i, vsini_i in enumerate(tqdm(vsini_grid)):
            grid, logL = self.plot_ccf(rv_grid=rv_grid, rv_step=rv_step, vsini=vsini_i, plot=False, wav_mod_nativ=wav_mod_nativ, flx_mod_nativ=flx_mod_nativ, res_mod_nativ=res_mod_nativ, map_rv_vsini=True, flx_obs=flx_obs, wav_obs=wav_obs, res_obs = res_obs, transm_obs = transm_obs)
            logL_map[i] = logL
 
        rv_grid = grid
        max_indices = np.unravel_index(np.argmax(logL_map), logL_map.shape)
        rv_peak, vsini_peak = rv_grid[max_indices[1]], vsini_grid[max_indices[0]]
        
        fig  = plt.figure('rv-vsin(i) map', figsize=(8,5))
        ax = fig.add_subplot()
        
        im = ax.imshow(logL_map, cmap=plt.cm.RdBu_r, extent=[rv_grid[0], rv_grid[-1],vsini_grid[0],vsini_grid[-1]])
        
        ax.set_xlabel('RV (km/s)')
        ax.set_ylabel('$v\,\sin i$ [km/s]')
        
        ax.axhline(y=vsini_peak, linestyle='--', c='C3')
        ax.axvline(x=rv_peak, linestyle='--', c='C3')
        
        ax.set_title(f'RV = {rv_peak:.1f} km/s, $v\,\sin i$ = {vsini_peak:.1f} km/s')
        
        cbar = fig.colorbar(im)
        cbar.set_label("logL", fontsize=22, labelpad=10)
        
        return logL_map, fig, ax
            


    def plot_PT(self, path_temp_profile, figsize=(6,5), model = 'ExoREM', emission_contribution = False):
        '''
        Function to plot the Pressure-Temperature profiles.
        Adpated from Nathan Zimniak.

        Args:
            path_temp_profile    (str): Path to the temperature profile grid
            figsize            (tuple): (default = (6, 5)) Size of the plot
            model                (str): (default = 'ExoREM') Name of the model grid
        Returns:
            - fig  (object) : matplotlib figure object
            - ax   (object) : matplotlib axes objects
        '''
        print('ForMoSA - Pressure-Temperature profile')

        samples = self.posterior_to_plot
        weights = self.weights

        # put nans where data is not realistic
        out=[]
        for i in range(0, len(samples)):
            if samples[i][0] < 400 or samples[i][0] > 2000:
                out.append(i)
            elif samples[i][1] < 3.00 or samples[i][1] > 5.00:
                out.append(i)
            elif 10**samples[i][2] < 0.32 or 10**samples[i][2] > 10.00:
                out.append(i)
            elif samples[i][3] < 0.10 or samples[i][3] > 0.80:
                out.append(i)
        for i in out:
            samples[i] = np.nan
        samples = samples[~np.isnan(samples).any(axis=1)]
        #Crée une liste pour chaque paramètre
        Teffs, loggs, MHs, COs = [], [], [], []
        if model == 'ATMO':
            gammas = []
        for i in range(0, len(samples)):
            Teffs.append(samples[i][0])
            loggs.append(samples[i][1])
            if model == 'ExoREM':
                MHs.append(10**(samples[i][2]))
                COs.append(samples[i][3])
            if model == 'ATMO':
                MHs.append(samples[i][2])
                COs.append(samples[i][4])
                gammas.append(samples[i][3])

        #Charge la grille de profils de température
        temperature_grid_xa = xr.open_dataarray(path_temp_profile, decode_cf=False, engine='netcdf4')
        temperature_grid_xa = temperature_grid_xa.where(~np.isnan(temperature_grid_xa))
        #Crée les profils de température associés aux points de la grille
        P = temperature_grid_xa.coords['P'].values
        P *= 1e-5
        temperature_profiles = np.full((len(samples), len(P)), np.nan)
        for i in range(0, len(samples)):
            if model == 'ExoREM':
                temperature_profiles[i][:] = np.asarray(temperature_grid_xa.interp(Teff=Teffs[i], logg=loggs[i], MH=MHs[i], CO=COs[i], kwargs={'fill_value':'extrapolate'}))#, kwargs={'fill_value':'extrapolate'})
            elif model == 'ATMO':
                temperature_profiles[i][:] = temperature_grid_xa.interp(Teff=Teffs[i], logg=loggs[i], MH=MHs[i], CO=COs[i], gamma=gammas[i])#, kwargs={'fill_value':'extrapolate'})
        if model == 'ATMO':
            #Calcule le 2eme facteur de robustesse (pour ATMO)
            nbNans = [0]*len(P)
            for i in range(0, len(temperature_profiles[0,:])):
                for j in range(0, len(temperature_profiles[:,0])):
                    if str(temperature_profiles[j,i]) == "nan":
                        nbNans[i] = nbNans[i]+1
            FdR2 = (len(samples)-np.array(nbNans))/len(samples)
            FdR1 = temperature_grid_xa.attrs['Facteur de robustesse 1']
            FdR = FdR1*FdR2
            #Extrapole les températures
            for i in range(0, len(samples)):
                newT = xr.DataArray(list(temperature_profiles[i][:]), [('pressure', list(np.array(P)))])
                newT = newT.interpolate_na(dim = 'pressure', method='linear', fill_value='extrapolate')
                temperature_profiles[i][:] = list(newT)
        #Calcule le profil le plus probable
        Tfit = []
        for i in range(0, len(P)):
            Tfit.append(np.nanpercentile(temperature_profiles[:,i], 50))
        #Calcule les percentiles 68 et 96 du profil le plus probable
        Tinf68, Tsup68, Tinf95, Tsup95 = [], [], [], []
        for i in range(0, len(P)):
            indices = np.argsort(temperature_profiles[:,i])
            sorted_temperatures_profiles = temperature_profiles[indices,i]
            sorted_weights = weights[indices]
            cumweights = np.cumsum(sorted_weights)
            Tinf68.append(np.interp(16/100, cumweights, sorted_temperatures_profiles))
            Tsup68.append(np.interp(54/100, cumweights, sorted_temperatures_profiles))
            Tinf95.append(np.interp(2/100, cumweights, sorted_temperatures_profiles))
            Tsup95.append(np.interp(98/100, cumweights, sorted_temperatures_profiles))
        #Plot le profil le plus probable et les percentiles associés
        
        if emission_contribution == True:
            
            for indobs, obs in enumerate(sorted(glob.glob(self.global_params.main_observation_path))):
                
                spectra, ck = self._get_spectra(self.theta_best)
                wav_obs_spectro, res_obs_spectro, wav_mod_nativ, flx_mod_nativ, res_mod_obs = spectra[indobs][0]*1e-6, spectra[indobs][11], spectra[indobs][13]*1e-6, spectra[indobs][14]*1e6, spectra[indobs][15]
                flx_mod_spectro = resolution_decreasing(self.global_params, wav_obs_spectro, [], res_obs_spectro, wav_mod_nativ, flx_mod_nativ, res_mod_obs, 'mod', indobs)
                
                h = cst.h.value      # Planck constant (J·s)
                c = cst.c.value      # Speed light (m/s)
                k_B = cst.k_B.value  # Boltzmann constant (J/K)
    
                term1 = (2 * h * (c)**2) / (wav_obs_spectro**5 * flx_mod_spectro)
                brightness_temperature = (h * c) / (wav_obs_spectro * k_B) * 1 / np.log(term1 + 1)
            
                pressure = interp1d(Tfit, P)
                pressure_level = pressure(brightness_temperature)
                # for b_T in brightness_temperature:
                #     idx = np.argmin(np.abs(Tfit - b_T))
                #     pressure_level.append(P[idx])
        
        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        ax.fill_betweenx(P, Tinf95, Tsup95, color=self.color_out, alpha=0.1, label=r'2 $\sigma$')
        ax.fill_betweenx(P, Tinf68, Tsup68, color=self.color_out, alpha=0.2, label=r'1 $\sigma$')
        ax.plot(Tfit, P, c=self.color_out, label='Best fit')
        
        ax.plot()
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.set_ylim([max(P), min(P)])
        
        x_fill = [ax.get_xticks()[0], ax.get_xticks()[-1]]
        ax.fill_between(x_fill, min(pressure_level), max(pressure_level), facecolor='lightblue', alpha=0.7, label='main contribution')
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Pressure (bars)')
        ax.legend(frameon=False)
        
        return fig, ax
    
    
    def plot_vmr(self, path_vmr, molecule, figsize=(6,5), model = 'ExoREM', fig=None, ax=None):
        '''
        Function to plot the vmr profiles of a molecule.
        Adpated from Nathan Zimniak.

        Args:
            path_vmr    (str): Path to the temperature profile grid
            molecule    (str): name of the molecule
            figsize     (tuple): (default = (6, 5)) Size of the plot
            model       (str): (default = 'ExoREM') Name of the model grid
            fig         (object): (default = None) matplotlib figure object   
            ax          (object): (default = None) matplotlib figure object
        Returns:
            - fig  (object) : matplotlib figure object
            - ax   (object) : matplotlib axes objects
        '''
        print('ForMoSA - Volume Mixing Ratio profile -', molecule)

        samples = self.posterior_to_plot

        # put nans where data is not realistic
        out=[]
        for i in range(0, len(samples)):
            if samples[i][0] < 400 or samples[i][0] > 2000:
                out.append(i)
            elif samples[i][1] < 3.00 or samples[i][1] > 5.00:
                out.append(i)
            elif 10**samples[i][2] < 0.32 or 10**samples[i][2] > 10.00:
                out.append(i)
            elif samples[i][3] < 0.10 or samples[i][3] > 0.80:
                out.append(i)
        for i in out:
            samples[i] = np.nan
        samples = samples[~np.isnan(samples).any(axis=1)]
        #Crée une liste pour chaque paramètre
        Teffs, loggs, MHs, COs = [], [], [], []
        if model == 'ATMO':
            gammas = []
        for i in range(0, len(samples)):
            Teffs.append(samples[i][0])
            loggs.append(samples[i][1])
            if model == 'ExoREM':
                MHs.append(10**(samples[i][2]))
                COs.append(samples[i][3])
            if model == 'ATMO':
                MHs.append(samples[i][2])
                COs.append(samples[i][4])
                gammas.append(samples[i][3])

        #Charge la grille de profils de température
        vmr_grid_xa = xr.open_dataarray(path_vmr)
        #Crée les profils de température associés aux points de la grille
        P = vmr_grid_xa.coords['P']
        vmr_profiles = np.full((len(samples), len(P)), np.nan)
        for i in range(0, len(samples)):
            if model == 'ExoREM':
                vmr_profiles[i][:] = vmr_grid_xa.interp(Teff=Teffs[i], logg=loggs[i], MH=MHs[i], CO=COs[i])#, kwargs={'fill_value':'extrapolate'})
            elif model == 'ATMO':
                vmr_profiles[i][:] = vmr_grid_xa.interp(Teff=Teffs[i], logg=loggs[i], MH=MHs[i], CO=COs[i], gamma=gammas[i])#, kwargs={'fill_value':'extrapolate'})
        if model == 'ATMO':
            #Calcule le 2eme facteur de robustesse (pour ATMO)
            nbNans = [0]*len(P)
            for i in range(0, len(vmr_profiles[0,:])):
                for j in range(0, len(vmr_profiles[:,0])):
                    if str(vmr_profiles[j,i]) == "nan":
                        nbNans[i] = nbNans[i]+1
            FdR2 = (len(samples)-np.array(nbNans))/len(samples)
            FdR1 = vmr_grid_xa.attrs['Facteur de robustesse 1']
            FdR = FdR1*FdR2
            #Extrapole les températures
            for i in range(0, len(samples)):
                newT = xr.DataArray(list(vmr_profiles[i][:]), [('pressure', list(np.array(P)))])
                newT = newT.interpolate_na(dim = 'pressure', method='linear', fill_value='extrapolate')
                vmr_profiles[i][:] = list(newT)
        #Calcule le profil le plus probable
        vmrfit = []
        for i in range(0, len(P)):
            vmrfit.append(np.nanpercentile(vmr_profiles[:,i], 50))
        #Calcule les percentiles 68 et 96 du profil le plus probable
        vmrinf68, vmrsup68, vmrinf95, vmrsup95 = [], [], [], []
        for i in range(0, len(P)):
            vmrinf68.append(np.nanpercentile(vmr_profiles[:,i], 16))
            vmrsup68.append(np.nanpercentile(vmr_profiles[:,i], 84))
            vmrinf95.append(np.nanpercentile(vmr_profiles[:,i], 2))
            vmrsup95.append(np.nanpercentile(vmr_profiles[:,i], 98))
        #Plot le profil le plus probable et les percentiles associés
        
        if fig == None:
            fig = plt.figure(figsize=figsize)
            ax = plt.axes()
            
        ax.plot(vmrfit, P, label=molecule)
        ax.set_yscale('log'), ax.set_xscale('log')
        ax.invert_yaxis()
        ax.set_ylim([max(P), min(P)])
        ax.set_xlabel('Volume mixing ratio')
        ax.set_ylabel('Pressure (Pa)')
        ax.legend(frameon=False)

        return fig, ax


    def plot_Clouds(self, cloud_prop, path_cloud_profile, figsize=(6,5)):
        '''
        Function to plot cloud profiles.
        Adapted from Nathan Zimniak

        Args:
            cloud_prop (str) : Choose the cloud species. The options are
                                ['eddy_diffusion_coefficient',
                                'vmr_CH4',
                                'vmr_CO',
                                'vmr_CO2',
                                'vmr_FeH',
                                'vmr_H2O',
                                'vmr_H2S',
                                'vmr_HCN',
                                'vmr_K',
                                'vmr_Na',
                                'vmr_NH3',
                                'vmr_PH3',
                                'vmr_TiO',
                                'vmr_VO',
                                'cloud_opacity_Fe',
                                'cloud_opacity_Mg2SiO4',
                                'cloud_particle_radius_Fe',
                                'cloud_particle_radius_Mg2SiO4',
                                'cloud_vmr_Fe',
                                'cloud_vmr_Mg2SiO4']
        Returns:
            - fig  (object) : matplotlib figure object
            - ax   (object) : matplotlib axes objects
        '''
        print('ForMoSA - Cloud profile')

        samples = self.posterior_to_plot
        weights = self.weights

        #Supprime les points hors de la grille
        out=[]
        for i in range(0, len(samples)):
            if samples[i][0] < 400 or samples[i][0] > 2000:
                out.append(i)
            elif samples[i][1] < 3.00 or samples[i][1] > 5.00:
                out.append(i)
            elif 10**samples[i][2] < 0.32 or 10**samples[i][2] > 10.00:
                out.append(i)
            elif samples[i][3] < 0.10 or samples[i][3] > 0.80:
                out.append(i)
        for i in out:
            samples[i] = np.nan
        samples = samples[~np.isnan(samples).any(axis=1)]
        #Crée une liste pour chaque paramètre
        Teffs, loggs, MHs, COs = [], [], [], []
        for i in range(0, len(samples)):
            Teffs.append(samples[i][0])
            loggs.append(samples[i][1])
            MHs.append(10**(samples[i][2]))
            COs.append(samples[i][3])
        #Charge la grille de profils d'une propriété d'un nuage
        cloud_prop_grid_xa = xr.open_dataarray(path_cloud_profile)
        #Crée les profils d'une propriété d'un nuage associés aux points de la grille
        P = cloud_prop_grid_xa.coords['P'] * 1e-5
        cloud_prop_profiles = np.full((len(samples), len(P)), np.nan)
        for i in range(0, len(samples)):
            cloud_prop_profiles[i][:] = cloud_prop_grid_xa.interp(Teff=Teffs[i], logg=loggs[i], MH=MHs[i], CO=COs[i])#, kwargs={'fill_value':'extrapolate'})
        # Calcule le profil le plus probable
        propfit = []
        for i in range(0, len(P)):
            propfit.append(np.nanpercentile(cloud_prop_profiles[:, i], 50))
        # Calcule les percentiles 68 et 96 du profil le plus probable
        propinf68, propsup68, propinf95, propsup95 = [], [], [], []
        for i in range(0, len(P)):
            indices = np.argsort(cloud_prop_profiles[:,i])
            sorted_cloud_prop_profiles = cloud_prop_profiles[indices,i]
            sorted_weights = weights[indices]
            cumweights = np.cumsum(sorted_weights)
            propinf68.append(np.interp(16/100, cumweights, sorted_cloud_prop_profiles))
            propsup68.append(np.interp(54/100, cumweights, sorted_cloud_prop_profiles))
            propinf95.append(np.interp(2/100, cumweights, sorted_cloud_prop_profiles))
            propsup95.append(np.interp(98/100, cumweights, sorted_cloud_prop_profiles))

        # Plot le profil le plus probable et les percentiles associés
        fig = plt.figure(figsize=figsize)
        ax = plt.axes()

        ax.fill_betweenx(P, propinf95, propsup95, color=self.color_out, alpha=0.1, label=r'2 $\sigma$')
        ax.fill_betweenx(P, propinf68, propsup68, color=self.color_out, alpha=0.2, label=r'1 $\sigma$')
        ax.plot(propfit, P, color=self.color_out, label='Best fit')

        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.set_ylim([max(P), min(P)])
        ax.minorticks_on()
        if cloud_prop == 'T':
            ax.set_xlabel('Temperature (K)')
        elif cloud_prop == 'eddy_diffusion_coefficient':
            ax.set_xlabel('Eddy diffusion coefficient ($m^2.s^{-1}$)')
        elif cloud_prop == 'vmr_CH4':
            ax.set_xlabel('$CH_4$ volume mixing ratio')
        elif cloud_prop == 'vmr_CO':
            ax.set_xlabel('CO volume mixing ratio')
        elif cloud_prop == 'vmr_CO2':
            ax.set_xlabel('$CO_2$ volume mixing ratio')
        elif cloud_prop == 'vmr_FeH':
            ax.set_xlabel('FeH volume mixing ratio')
        elif cloud_prop == 'vmr_H2O':
            ax.set_xlabel('$H_2O$ volume mixing ratio')
        elif cloud_prop == 'vmr_H2S':
            ax.set_xlabel('$H_2S$ volume mixing ratio')
        elif cloud_prop == 'vmr_HCN':
            ax.set_xlabel('HCN volume mixing ratio')
        elif cloud_prop == 'vmr_K':
            ax.set_xlabel('K volume mixing ratio')
        elif cloud_prop == 'vmr_Na':
            ax.set_xlabel('Na volume mixing ratio')
        elif cloud_prop == 'vmr_NH3':
            ax.set_xlabel('$NH_3$ volume mixing ratio')
        elif cloud_prop == 'vmr_PH3':
            ax.set_xlabel('$PH_3$ volume mixing ratio')
        elif cloud_prop == 'vmr_TiO':
            ax.set_xlabel('TiO volume mixing ratio')
        elif cloud_prop == 'vmr_VO':
            ax.set_xlabel('VO volume mixing ratio')
        elif cloud_prop == 'cloud_opacity_Fe':
            ax.set_xlabel('Fe cloud opacity')
        elif cloud_prop == 'cloud_opacity_Mg2SiO4':
            ax.set_xlabel('$Mg_2SiO_4$ cloud opacity')
        elif cloud_prop == 'cloud_particle_radius_Fe':
            ax.set_xlabel('Fe cloud particle radius (m)')
        elif cloud_prop == 'cloud_particle_radius_Mg2SiO4':
            ax.set_xlabel('$Mg_2SiO_4$ cloud particle radius (m)')
        elif cloud_prop == 'cloud_vmr_Fe':
            ax.set_xlabel('Fe cloud volume mixing ratio')
        elif cloud_prop == 'cloud_vmr_Mg2SiO4':
            ax.set_xlabel('$Mg_2SiO_4$ cloud volume mixing ratio')
        ax.set_ylabel('Pressure (bars)')

        ax.legend(frameon=False)

        return fig, ax



