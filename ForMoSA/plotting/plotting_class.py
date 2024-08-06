
from __future__ import print_function, division
import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d
import corner
import xarray as xr
import pickle
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('../'))

# Import ForMoSA
from main_utilities import GlobFile
from nested_sampling.nested_modif_spec import modif_spec
from nested_sampling.nested_modif_spec import doppler_fct
from nested_sampling.nested_modif_spec import lsq_fct
from nested_sampling.nested_modif_spec import vsini_fct_accurate
from adapt.extraction_functions import resolution_decreasing, adapt_model, decoupe
from adapt.extraction_functions import adapt_observation_range




def bin_data(wave, data, bin_size):
    '''
    Function to bin data given a bin size

    Args:
        wave         (array): wavelength of the data
        data         (array): data
        bin_size       (int): size of the bin to apply

    Returns:
        - wave_binned  (array): binned wavelength
        - data_binned  (array): binned data

    Author: Allan Denis
    '''
    # First quick check that len of data is a multpiple of bin_size
    while(len(data)%bin_size != 0):
        wave, data = wave[:-1], data[:-1]
        
    bins = np.arange(0, len(wave), bin_size)
    wave_binned = np.add.reduceat(wave, bins) / bin_size
    data_binned = np.add.reduceat(data, bins) 
    
    return wave_binned, data_binned
    
    

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

    def __init__(self, config_file_path, color_out):
        '''
        Initialize class by inheriting the global parameter class of ForMoSA.

        Args:
            config_file_path   (str): path to the config.ini file currently used
            color_out          (str): color to use for the model
        Returns:
            None
        '''
        
        self.global_params = GlobFile(config_file_path)  
        self.color_out     = color_out


    def _get_posteriors(self):
        '''
        Function to get the posteriors, including luminosity derivation and Bayesian evidence logz.

        Args:
            None
        Returns:
            None
        '''
        with open(self.global_params.result_path + '/result_' + self.global_params.ns_algo + '.pic', 'rb') as open_pic:
            result = pickle.load(open_pic)
        self.samples = result['samples']
        self.weights = result['weights']

        # To test the quality of the fit
        self.logl=result['logl']
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
                            ['bb_T', 'bb_T', '(K)'],
                            ['bb_R', 'bb_R', r'(R$\mathrm{_{Jup}}$)']
                            ]

        tot_list_param_title = []
        theta_index = []
        if self.global_params.par1 != 'NA':
            tot_list_param_title.append(attrs['title'][0] + ' ' + attrs['unit'][0])
            theta_index.append('par1')
        if self.global_params.par2 != 'NA':
            tot_list_param_title.append(attrs['title'][1] + ' ' + attrs['unit'][1])
            theta_index.append('par2')
        if self.global_params.par3 != 'NA':
            tot_list_param_title.append(attrs['title'][2] + ' ' + attrs['unit'][2])
            theta_index.append('par3')
        if self.global_params.par4 != 'NA':
            tot_list_param_title.append(attrs['title'][3] + ' ' + attrs['unit'][3])
            theta_index.append('par4')
        if self.global_params.par5 != 'NA':
            tot_list_param_title.append(attrs['title'][4] + ' ' + attrs['unit'][4])
            theta_index.append('par5')

        # Extra-grid parameters

        if self.global_params.r != 'NA' and self.global_params.r[0] != 'constant':
            tot_list_param_title.append(extra_parameters[0][1] + ' ' + extra_parameters[0][2])
            theta_index.append('r')
        if self.global_params.d != 'NA' and self.global_params.d[0] != 'constant':
            tot_list_param_title.append(extra_parameters[1][1] + ' ' + extra_parameters[1][2])
            theta_index.append('d')

        # - - - - - - - - - - - - - - - - - - - - -
                
        # Individual parameters / observation

        if len(self.global_params.alpha) > 3: # If you want separate alpha for each observations
            main_obs_path = self.global_params.main_observation_path
            for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
                if self.global_params.alpha[indobs*3] != 'NA' and self.global_params.alpha[indobs*3] != 'constant': # Check if the idobs is different from constant
                    tot_list_param_title.append(extra_parameters[2][1] + fr'$_{indobs}$' + ' ' + extra_parameters[2][2])
                    theta_index.append(f'alpha_{indobs}')
        else: # If you want 1 common alpha for all observations
            if self.global_params.alpha != 'NA' and self.global_params.alpha != 'constant':
                tot_list_param_title.append(extra_parameters[2][1] + ' ' + extra_parameters[2][2])
                theta_index.append('alpha')
        if len(self.global_params.rv) > 3: # If you want separate rv for each observations
            main_obs_path = self.global_params.main_observation_path
            for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
                if self.global_params.rv[indobs*3] != 'NA' and self.global_params.rv[indobs*3] != 'constant': # Check if the idobs is different from constant
                    tot_list_param_title.append(extra_parameters[3][1] + fr'$_{indobs}$' + ' ' + extra_parameters[3][2])
                    theta_index.append(f'rv_{indobs}')
        else: # If you want 1 common rv for all observations
            if self.global_params.rv != 'NA' and self.global_params.rv != 'constant':
                tot_list_param_title.append(extra_parameters[3][1] + ' ' + extra_parameters[3][2])
                theta_index.append('rv')
        if len(self.global_params.vsini) > 4: # If you want separate vsini for each observations
            main_obs_path = self.global_params.main_observation_path
            for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
                if self.global_params.vsini[indobs*4] != 'NA' and self.global_params.vsini[indobs*4] != 'constant': # Check if the idobs is different from constant
                    tot_list_param_title.append(extra_parameters[5][1] + fr'$_{indobs}$' + ' ' + extra_parameters[5][2])
                    theta_index.append(f'vsini_{indobs}')
        else: # If you want 1 common vsini for all observations
            if self.global_params.vsini != 'NA' and self.global_params.vsini != 'constant':
                tot_list_param_title.append(extra_parameters[5][1] + ' ' + extra_parameters[5][2])
                theta_index.append('vsini')
        if len(self.global_params.ld) > 3: # If you want separate ld for each observations
            main_obs_path = self.global_params.main_observation_path
            for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
                if self.global_params.ld[indobs*3] != 'NA' and self.global_params.ld[indobs*3] != 'constant': # Check if the idobs is different from constant
                    tot_list_param_title.append(extra_parameters[6][1] + fr'$_{indobs}$' + ' ' + extra_parameters[6][2])
                    theta_index.append(f'ld_{indobs}')
        else: # If you want 1 common vsini for all observations
            if self.global_params.ld != 'NA' and self.global_params.ld != 'constant':
                tot_list_param_title.append(extra_parameters[6][1] + ' ' + extra_parameters[6][2])
                theta_index.append('ld')

        # - - - - - - - - - - - - - - - - - - - - -

        if self.global_params.av != 'NA' and self.global_params.av[0] != 'constant':
            tot_list_param_title.append(extra_parameters[4][1] + ' ' + extra_parameters[4][2])
            theta_index.append('av')
        ## cpd bb
        if self.global_params.bb_T != 'NA' and self.global_params.bb_T[0] != 'constant':
            tot_list_param_title.append(extra_parameters[7][1] + ' ' + extra_parameters[7][2])
            theta_index.append('bb_T')
        if self.global_params.bb_R != 'NA' and self.global_params.bb_R[0] != 'constant':
            tot_list_param_title.append(extra_parameters[8][1] + ' ' + extra_parameters[8][2])
            theta_index.append('bb_R')
        self.theta_index = np.asarray(theta_index)

        posterior_to_plot = self.samples
        if self.global_params.r != 'NA' and self.global_params.r[0] != 'constant':
            posterior_to_plot = []
            tot_list_param_title.append(r'log(L/L$\mathrm{_{\odot}}$)')
            
            for res, results in enumerate(self.samples): 
                ind_theta_r = np.where(self.theta_index == 'r')
                r_picked = results[ind_theta_r[0]]
                
                lum = np.log10(4 * np.pi * (r_picked * 69911000.) ** 2 * results[0] ** 4 * 5.670e-8 / 3.83e26)
                #print(lum)
                results = np.concatenate((results, np.asarray(lum)))
                #print(results)
                posterior_to_plot.append(results)

        self.posterior_to_plot = np.array(posterior_to_plot)
        self.posteriors_names = tot_list_param_title


    def plot_corner(self, levels_sig=[0.997, 0.95, 0.68], bins=100, quantiles=(0.16, 0.5, 0.84), burn_in=0):
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

        self._get_posteriors()

        fig = corner.corner(self.posterior_to_plot[burn_in:],
                            weights=self.weights[burn_in:],
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
                            label_kwargs=dict(fontsize=14))
    

        fig.supxlabel(self.outputs_string, va='top')

        return fig


    def plot_chains(self, figsize=(7,15)):
        '''
        Plot to check the convergence of the posterior chains.
        Multiple (sub-)axis plot.

        Args:
            figsize     (tuple): (default = (7, 15)) size of the plot
        Returns:
            - fig  (object) : matplotlib figure object
            - ax   (object) : matplotlib axes objects
        '''
        print('ForMoSA - Posteriors chains for each parameter')

        self._get_posteriors()
        
        col = int(len(self.posterior_to_plot[0][:])/2)+int(len(self.posterior_to_plot[0][:])%2)
        fig, axs = plt.subplots(col, 2, figsize=figsize)

        n=0
        for i in range(col):
            for j in range(2):
                axs[i, j].plot(self.posterior_to_plot[:,n], color=self.color_out, alpha=0.8)
                axs[i, j].set_ylabel(self.posteriors_names[n])
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

        self._get_posteriors()

        list_posteriors = []
        list_uncert_down = []
        list_uncert_up = []
        for l in range(len(self.posterior_to_plot[1,:])):
            q16, q50, q84 = corner.quantile(self.posterior_to_plot[:,l], quantiles)
            
            list_posteriors.append(self.theta_best)
            list_uncert_down.append(q16)
            list_uncert_up.append(q84)

        fig = plt.figure(figsize=(6, 6))
        radar = ComplexRadar(fig, self.posteriors_names, ranges)

        radar.plot(list_posteriors, 'o-', color=self.color_out, label=label)
        radar.fill_between(list_uncert_down,list_uncert_up, color=self.color_out, alpha=0.2)

        radar.ax.legend(loc='center', bbox_to_anchor=(0.5, -0.20),frameon=False, ncol=2)

        return fig, radar.ax


    def _get_spectra(self,theta):
        '''
        Function to get the data and best model asociated.

        Args:
            theta                   (list): best parameter values
        Returns:
            - modif_spec_chi2  list(n-array): list containing the spectroscopic wavelength, spectroscopic fluxes of the data, 
                                            spectroscopic errors of the data, spectroscopic fluxes of the model, 
                                            photometric wavelength, photometric fluxes of the data, photometric errors of the data, 
                                            spectroscopic fluxes of the model,
                                            planet transmission, star fluxes, systematics
            - ck                list(floats): list scaling factor(s)
        '''
        # Get the posteriors
        self._get_posteriors()

        # Create a list for each spectra (obs and mod) for each observation + scaling factors
        modif_spec_MOSAIC = []
        CK = []

        for indobs, obs in enumerate(sorted(glob.glob(self.global_params.main_observation_path))):

            self.global_params.observation_path = obs
            obs_name = os.path.splitext(os.path.basename(self.global_params.observation_path))[0]

            spectrum_obs = np.load(os.path.join(self.global_params.result_path, f'spectrum_obs_{obs_name}.npz'), allow_pickle=True)
            wav_obs_spectro = np.asarray(spectrum_obs['obs_spectro_merge'][0], dtype=float)
            flx_obs_spectro = np.asarray(spectrum_obs['obs_spectro_merge'][1], dtype=float)
            err_obs_spectro = np.asarray(spectrum_obs['obs_spectro_merge'][2], dtype=float)
            transm_obs = np.asarray(spectrum_obs['obs_opt_merge'][1], dtype=float)
            star_flx_obs = np.asarray(spectrum_obs['obs_opt_merge'][2], dtype=float)
            system_obs = np.asarray(spectrum_obs['obs_opt_merge'][3], dtype=float)
            if 'obs_photo' in spectrum_obs.keys():
                wav_obs_photo = np.asarray(spectrum_obs['obs_photo'][0], dtype=float)
                flx_obs_photo = np.asarray(spectrum_obs['obs_photo'][1], dtype=float)
                err_obs_photo = np.asarray(spectrum_obs['obs_photo'][2], dtype=float)
            else:
                wav_obs_photo = np.asarray([], dtype=float)
                flx_obs_photo = np.asarray([], dtype=float)
                err_obs_photo = np.asarray([], dtype=float)

            # Recovery of the spectroscopy and photometry model
            path_grid_spectro = os.path.join(self.global_params.adapt_store_path, f'adapted_grid_spectro_{self.global_params.grid_name}_{obs_name}_nonan.nc')
            path_grid_photo = os.path.join(self.global_params.adapt_store_path, f'adapted_grid_photo_{self.global_params.grid_name}_{obs_name}_nonan.nc')
            ds = xr.open_dataset(path_grid_spectro, decode_cf=False, engine='netcdf4')
            grid_spectro = ds['grid']
            ds.close()
            ds = xr.open_dataset(path_grid_photo, decode_cf=False, engine='netcdf4')
            grid_photo = ds['grid']
            ds.close()

            if self.global_params.par3 == 'NA':
                if len(grid_spectro['wavelength']) != 0:
                    flx_mod_spectro = np.asarray(grid_spectro.interp(par1=theta[0], par2=theta[1],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_spectro = np.asarray([])
                if len(grid_photo['wavelength']) != 0:
                    flx_mod_photo = np.asarray(grid_photo.interp(par1=theta[0], par2=theta[1],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_photo = np.asarray([])
            elif self.global_params.par4 == 'NA':
                if len(grid_spectro['wavelength']) != 0:
                    flx_mod_spectro = np.asarray(grid_spectro.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_spectro = np.asarray([])
                if len(grid_photo['wavelength']) != 0:
                    flx_mod_photo = np.asarray(grid_photo.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_photo = np.asarray([])
            elif self.global_params.par5 == 'NA':
                if len(grid_spectro['wavelength']) != 0:
                    flx_mod_spectro = np.asarray(grid_spectro.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_spectro = np.asarray([])
                if len(grid_photo['wavelength']) != 0:
                    flx_mod_photo = np.asarray(grid_photo.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_photo = np.asarray([])
            else:
                if len(grid_spectro['wavelength']) != 0:
                    flx_mod_spectro = np.asarray(grid_spectro.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            par5=theta[4],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_spectro = np.asarray([])
                if len(grid_photo['wavelength']) != 0:
                    flx_mod_photo = np.asarray(grid_photo.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            par5=theta[4],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_photo = np.asarray([])

            # Modification of the synthetic spectrum with the extra-grid parameters
            modif_spec_chi2 = modif_spec(self.global_params, theta, self.theta_index,
                                        wav_obs_spectro, flx_obs_spectro, err_obs_spectro, flx_mod_spectro,
                                        wav_obs_photo, flx_obs_photo, err_obs_photo, flx_mod_photo,
                                        transm_obs, star_flx_obs, system_obs, indobs=indobs)
            ck = modif_spec_chi2[8]

            modif_spec_MOSAIC.append(modif_spec_chi2)
            CK.append(ck)

        modif_spec_chi2 = modif_spec_MOSAIC
        ck = CK
        
        return modif_spec_chi2, ck
    

    def get_FULL_spectra(self, theta, grid_used = 'original', wavelengths=[], N_points=1000, re_interp=False, int_method="linear"):
        '''
        Extract a model spectrum from another grid.

        Args:
            theta:                       (list): best parameter values
            grid_used:                    (str): (default = 'original') Path to the grid from where to extract the spectrum. If 'original', the current grid will be used.
            wavelengths:                 (list): (default = []) Desired wavelength range. If [] max and min values of wav_for_adapt range will be use to create the wavelength range.
            N_points:                     (int): (default = 1000) Number of points.
            re_interp:                (boolean): (default = False). Option to reinterpolate or not the grid.
            int_method:                   (str): (default = "linear") Interpolation method for the grid (if reinterpolated).
        Returns:   
            - wav_final                   (array): Wavelength array of the full model
            - flx_final                   (array): Flux array of the full model
            - ck                          (float): Scaling factor of the full model
        '''
        self._get_posteriors()

        if len(wavelengths)==0:
            # Define the wavelength grid for the full spectra as resolution and wavelength range function
            for indobs, obs in enumerate(sorted(glob.glob(self.global_params.main_observation_path))):
                my_string_ind = self.global_params.wav_for_adapt[indobs].split('/')[0].split(',')
                wav_ind = [float(x) for x in my_string_ind]
                if indobs == 0:
                    wav = wav_ind
                else:
                    wav = np.concatenate((wav, wav_ind))
            wav = np.sort(wav)
            wavelengths = np.linspace(wav[0],wav[-1],N_points)
        else:
            wavelengths = np.linspace(wavelengths[0],wavelengths[-1],N_points)

        # Recover the original grid
        if grid_used == 'original':
            path_grid = self.global_params.model_path
        else: 
            path_grid = grid_used

        ds = xr.open_dataset(path_grid, decode_cf=False, engine="netcdf4")

        # Possibility of re-interpolating holes if the grid contains to much of them (WARNING: Very long process)
        if re_interp == True:
            print('-> The possible holes in the grid are (re)interpolated: ')
            for key_ind, key in enumerate(ds.attrs['key']):
                print(str(key_ind+1) + '/' + str(len(ds.attrs['key'])))
                ds = ds.interpolate_na(dim=key, method="linear", fill_value="extrapolate", limit=None,
                                            max_gap=None)
                
        wav_mod_nativ = ds["wavelength"].values
        grid = ds['grid']
        ds.close()
        
        if self.global_params.par3 == 'NA':
            flx_mod_nativ = grid.interp(par1=theta[0], par2=theta[1],method=int_method, kwargs={"fill_value": "extrapolate"})
        elif self.global_params.par4 == 'NA':
            flx_mod_nativ = grid.interp(par1=theta[0], par2=theta[1], par3=theta[2],method=int_method, kwargs={"fill_value": "extrapolate"})
        elif self.global_params.par5 == 'NA':
            flx_mod_nativ = grid.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],method=int_method, kwargs={"fill_value": "extrapolate"}) 
        else:
            flx_mod_nativ = grid.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],par5=theta[4],method=int_method, kwargs={"fill_value": "extrapolate"})
        
        # Interpolate to desire wavelength range
        interp_mod_to_obs = interp1d(wav_mod_nativ, flx_mod_nativ, fill_value='extrapolate')
        flx_mod_final = interp_mod_to_obs(wavelengths)
        
        spectra = self._get_spectra(theta)
        # WARNING : In case of multiple spectra, it is possible to work with different scaling factors. Here we only take the scaling factor of the first spectrum
        #in the MOSAIC (used for the plot_fit)
        ck = float(spectra[-1][0])

        wavelengths = np.asarray(wavelengths, dtype=float)
        flx_mod_final = np.asarray(flx_mod_final, dtype=float)
        flx_mod_final_calib = np.asarray(flx_mod_final*ck, dtype=float)
        #print(flx_mod_final[100],ck)
        err_mod_final_calib = flx_mod_final_calib*0.1

        wav_final, _, _, flx_final, _, _, _, _, _, _, _, _, _ = modif_spec(self.global_params, theta, self.theta_index,
                                                                                    wavelengths, flx_mod_final_calib, err_mod_final_calib, flx_mod_final_calib/ck,
                                                                                    [], [], [], [], [], [])
    
        return wav_final, flx_final, ck


    
    def plot_fit(self, figsize=(10, 5), uncert='no', trans='no', logx='no', logy='no', norm='no'):
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
            - fig    (object) : matplotlib figure object
            - ax     (object) : matplotlib axes objects, main spectra plot
            - axr    (object) : matplotlib axes objects, residuals
            - axr2   (object) : matplotlib axes objects, right side density histogram
        '''
        print('ForMoSA - Best fit and residuals plot')

        fig = plt.figure(figsize=figsize)
        fig.tight_layout()
        size = (7,11)
        ax = plt.subplot2grid(size, (0, 0),rowspan=5 ,colspan=10)
        axr= plt.subplot2grid(size, (5, 0),rowspan=2 ,colspan=10)
        axr2= plt.subplot2grid(size, (5, 10),rowspan=2 ,colspan=1)

       
        spectra, ck = self._get_spectra(self.theta_best)
        iobs_spectro = 0
        iobs_photo = 0
        

        # Scale or not in absolute flux
        if norm != 'yes': 
            if len(spectra[0][0]) != 0:
                ck = np.full(len(spectra[0][0]), 1)
            else:
                ck = np.full(len(spectra[0][4]), 1)


        for indobs, obs in enumerate(sorted(glob.glob(self.global_params.main_observation_path))):
            
            if self.global_params.use_lsqr[indobs] == 'True':
                # If we used the lsq function, it means that our data is contaminated by the starlight difraction
                # so the model is the sum of the planet model + the estimated stellar contribution
                spectra = list(spectra) # Transform spectra to a list so that we can modify its values
                spectra[indobs] = list(spectra[indobs])
                model, planet_contribution, stellar_contribution, star_flx = spectra[indobs][3], spectra[indobs][9], spectra[indobs][10], spectra[indobs][11]
                spectra[indobs][3] = planet_contribution * model + np.dot(stellar_contribution, star_flx[0].T)
                systematics = spectra[indobs][12]
                if len(systematics) > 0:
                    spectra[indobs][3] += systematics

            if len(spectra[indobs][0]) != 0:
                iobs_spectro += 1
                iobs_photo += 1
                if uncert=='yes':
                    ax.errorbar(spectra[indobs][0], spectra[indobs][1]/ck[indobs], yerr=spectra[indobs][2]/ck[indobs], c='k', alpha=0.2)
                ax.plot(spectra[indobs][0], spectra[indobs][1]/ck[indobs], c='k')
                ax.plot(spectra[indobs][0], spectra[indobs][3]/ck[indobs], c=self.color_out, alpha=0.8)


                residuals = spectra[indobs][3] - spectra[indobs][1]
                sigma_res = np.nanstd(residuals) # Replace np.std by np.nanstd if nans are in the array to ignore them
                axr.plot(spectra[indobs][0], residuals/sigma_res, c=self.color_out, alpha=0.8)
                axr.axhline(y=0, color='k', alpha=0.5, linestyle='--')
                axr2.hist(residuals/sigma_res, bins=100 ,color=self.color_out, alpha=0.5, density=True, orientation='horizontal')
                axr2.legend(frameon=False,handlelength=0)

                if indobs == iobs_spectro-1:
                    # Add labels out of the loops
                    ax.plot(spectra[0][0], np.empty(len(spectra[0][0]))*np.nan, c='k', label='Spectroscopic data')
                    ax.plot(spectra[0][0], np.empty(len(spectra[0][0]))*np.nan, c=self.color_out, label='Spectroscopic model')
                    axr.plot(spectra[0][0], np.empty(len(spectra[0][0]))*np.nan, c=self.color_out, label='Spectroscopic model-data')
                    axr2.hist(residuals/sigma_res, bins=100 ,color=self.color_out, alpha=0.5, density=True, orientation='horizontal', label='density')
                    
                    iobs_spectro = -1
                    

            if len(spectra[indobs][4]) != 0:
                iobs_photo += 1
                iobs_spectro += 1
                # If you want to plot the transmission filters
                if trans == 'yes':
                    self.global_params.observation_path = obs
                    obs_name = os.path.splitext(os.path.basename(self.global_params.observation_path))[0]
                    spectrum_obs = np.load(os.path.join(self.global_params.result_path, f'spectrum_obs_{obs_name}.npz'), allow_pickle=True)
                    obs_photo_ins = spectrum_obs['obs_photo_ins']
                    for pho_ind, pho in enumerate(obs_photo_ins):
                        path_list = __file__.split("/")[:-2]
                        separator = '/'
                        filter_pho = np.load(separator.join(path_list) + '/phototeque/' + pho + '.npz')
                        ax.fill_between(filter_pho['x_filt'], filter_pho['y_filt']*0.8*min(spectra[indobs][5]/ck[indobs]),color=self.color_out, alpha=0.3)
                        ax.text(np.mean(filter_pho['x_filt']), np.mean(filter_pho['y_filt']*0.4*min(spectra[indobs][5]/ck[indobs])), pho, horizontalalignment='center', c='gray')
                ax.plot(spectra[indobs][4], spectra[indobs][5] / ck[indobs], 'ko', alpha=0.7)
                ax.plot(spectra[indobs][4], spectra[indobs][7] / ck[indobs], 'o', color=self.color_out)
                

                residuals_phot = spectra[indobs][7]-spectra[indobs][5]
                sigma_res = np.std(residuals_phot)
                axr.plot(spectra[indobs][4], residuals_phot/sigma_res, 'o', c=self.color_out, alpha=0.8)
                axr.axhline(y=0, color='k', alpha=0.5, linestyle='--')

                if indobs == iobs_photo-1:
                    # Add labels out of the loops
                    ax.plot(spectra[0][4], np.empty(len(spectra[0][4]))*np.nan, 'ko', label='Photometry data')
                    ax.plot(spectra[0][4], np.empty(len(spectra[0][4]))*np.nan, 'o', c=self.color_out, label='Photometry model')
                    axr.plot(spectra[0][4], np.empty(len(spectra[0][4]))*np.nan, 'o', c=self.color_out, label='Photometry model-data')
                    
                    iobs_photo = -1
                    
        # Set xlog-scale
        if logx == 'yes':
            ax.set_xscale('log')
            axr.set_xscale('log')
        # Set xlog-scale
        if logy == 'yes':
            ax.set_yscale('log')
        # Remove the xticks from the first ax
        ax.set_xticks([])
        # Labels
        axr.set_xlabel(r'Wavelength (µm)')
        if norm != 'yes': 
            ax.set_ylabel(r'Flux (W m-2 µm-1)')
        else:
            ax.set_ylabel(r'Normalised flux (W m-2 µm-1)')
        axr.set_ylabel(r'Residuals ($\sigma$)')
        
        axr2.axis('off')
        ax.legend(frameon=False)
        axr.legend(frameon=False)
            
        # define the data as global
        self.spectra = spectra

        return fig, ax, axr, axr2
    
    
    def plot_fit_HiRes(self, figsize=(10, 5), uncert='no', trans='no', logx='no', logy='no', norm='no'):
        '''
        Same as plot_fit but with the stellar and planetary models for high-resolution spectroscopy. Does not include residuals in a sub-axis.

        Args:
            figsize    (tuple): (default = (10, 5)) Size of the plot
            uncert     (str): (default = no) 'yes' or 'no' to plot spectra with associated error bars
            trans      (str): (default = no) 'yes' or 'no' to plot transmision curves for photometry
            logx       (str): (default = no) 'yes' or 'no' to plot the wavelength in log scale
            logy       (str): (default = no) 'yes' or 'no' to plot the flux in log scale
            norm       (str): (default = no) 'yes' or 'no' to plot the normalized spectra
        Returns:
            - fig  (object) : matplotlib figure object
            - ax   (object) : matplotlib axes objects
        '''
        print('ForMoSA - Best fit and residuals plot')

        fig1, ax1 = plt.subplots(1,1, figsize=figsize)
       
        spectra, ck = self._get_spectra(self.theta_best)
        iobs_spectro = 0
        

        # Scale or not in absolute flux
        if norm != 'yes': 
            ck = np.full(len(spectra[0][0]), 1)

        for indobs, obs in enumerate(sorted(glob.glob(self.global_params.main_observation_path))):
            
            if self.global_params.use_lsqr[indobs] == 'True':
                # If we used the lsq function, it means that our data is contaminated by the starlight difraction
                # so the model is the sum of the planet model + the estimated stellar contribution
                spectra = list(spectra) # Transform spectra to a list so that we can modify its values
                spectra[indobs] = list(spectra[indobs])
                model, planet_contribution, stellar_contribution, star_flx, systematics = spectra[indobs][3], spectra[indobs][9], spectra[indobs][10], spectra[indobs][11], spectra[indobs][12]
                transm = spectra[indobs][13]
                spectra[indobs][3] = planet_contribution * model + np.dot(stellar_contribution, star_flx[0].T)
                if len(systematics) > 0:
                    spectra[indobs][3] += systematics

            if len(spectra[indobs][0]) != 0:
                iobs_spectro += 1
                if uncert=='yes':
                    ax1.errorbar(spectra[indobs][0], spectra[indobs][1]/ck[indobs], yerr=spectra[indobs][2]/ck[indobs], c='k', alpha=0.2)
                ax1.plot(spectra[indobs][0], spectra[indobs][1] - spectra[indobs][3], 'o', alpha = 0.2, color='g')
                ax1.plot(spectra[indobs][0], spectra[indobs][1]/ck[indobs], c='k')
                ax1.plot(spectra[indobs][0], spectra[indobs][3]/ck[indobs], c='r')
                ax1.plot(spectra[indobs][0], np.dot(stellar_contribution, star_flx[0].T), c='b')
                ax1.plot(spectra[indobs][0], planet_contribution * model, c='purple')

 
                if indobs == iobs_spectro - 1:
                    # Add labels out of the loops
                    ax1.plot(spectra[0][0], np.empty(len(spectra[0][0]))*np.nan, c='k', label='residuals')
                    ax1.plot(spectra[0][0], np.empty(len(spectra[0][0]))*np.nan, c='k', label='data')
                    ax1.plot(spectra[0][0], np.empty(len(spectra[0][0]))*np.nan, c=self.color_out, label='full model')
                    ax1.plot(spectra[0][0], np.empty(len(spectra[0][0]))*np.nan, c='k', label='stellar model')
                    ax1.plot(spectra[0][0], np.empty(len(spectra[0][0]))*np.nan, c='k', label='planetary model')
                    
                    iobs_spectro = -1

            if len(spectra[indobs][4]) != 0:
                # If you want to plot the transmission filters
                if trans == 'yes':
                    self.global_params.observation_path = obs
                    obs_name = os.path.splitext(os.path.basename(self.global_params.observation_path))[0]
                    spectrum_obs = np.load(os.path.join(self.global_params.result_path, f'spectrum_obs_{obs_name}.npz'), allow_pickle=True)
                    obs_photo_ins = spectrum_obs['obs_photo_ins']
                    for pho_ind, pho in enumerate(obs_photo_ins):
                        path_list = __file__.split("/")[:-2]
                        separator = '/'
                        filter_pho = np.load(separator.join(path_list) + '/phototeque/' + pho + '.npz')
                        ax1.fill_between(filter_pho['x_filt'], filter_pho['y_filt']*0.8*min(spectra[indobs][5]/ck[indobs]),color=self.color_out, alpha=0.3)
                        ax1.text(np.mean(filter_pho['x_filt']), np.mean(filter_pho['y_filt']*0.4*min(spectra[indobs][5]/ck[indobs])), pho, horizontalalignment='center', c='gray')
                ax1.plot(spectra[indobs][4], spectra[indobs][5]/ck[indobs], 'ko', alpha=0.7)
                ax1.plot(spectra[indobs][4], spectra[indobs][7]/ck[indobs], 'o', color=self.color_out)
                

                residuals_phot = spectra[indobs][7]-spectra[indobs][5]
                sigma_res = np.std(residuals_phot)


                if indobs == 0:
                    # Add labels out of the loops
                    ax1.plot(spectra[0][4], np.empty(len(spectra[0][4]))*np.nan, 'ko', label='Photometry data')
                    ax1.plot(spectra[0][4], np.empty(len(spectra[0][4]))*np.nan, 'o', c=self.color_out, label='Photometry model')

        # Set xlog-scale
        if logx == 'yes':
            ax1.set_xscale('log')
        # Set xlog-scale
        if logy == 'yes':
            ax1.set_yscale('log')
        # Remove the xticks from the first ax
        ax1.set_xticks([])
        # Labels
        if norm != 'yes': 
            ax1.set_ylabel(r'Flux (ADU)')
        else:
            ax1.set_ylabel(r'Normalised flux (W m-2 µm-1)')
            
        ax1.set_xlabel(r'wavelength ($ \mu $m)')
        
        fig1.legend()
        plt.figure(fig1)
        plt.savefig(self.global_params.result_path + 'full_data.pdf')

        # define the data as global
        self.spectra = spectra

        return fig1, ax1
    
    
    def plot_HiRes_comp_model(self, figsize=(10, 5), norm='no', data_resolution = 0):
        '''
        Specific function to plot the best fit comparing with the data for high-resolution spectroscopy.

        Args:
            figsize             (tuple): (default = (10, 5)) Size of the plot
            norm                  (str): (default = no) 'yes' or 'no' to plot the normalized spectra
            data_resolution       (int): (default = 0) Custom resolution to broadened data
        Returns:
            - fig1  (object) : matplotlib figure object
            - ax1   (object) : matplotlib axes objects
        '''
        print('ForMoSA - Planet model and data')
       
        spectra, ck = self._get_spectra(self.theta_best)
        fig1, ax1 = plt.subplots(1, 1, figsize = figsize)
        fig, ax = plt.subplots(1, 1, figsize = figsize)

        # Scale or not in absolute flux
        if norm != 'yes': 
            ck = np.full(len(spectra[0][0]), 1)
            
        pdf = PdfPages(self.global_params.result_path + 'PLanet_model_and_data_resolution_degraded.pdf')
        plt.ioff()
        

        for indobs, obs in enumerate(sorted(glob.glob(self.global_params.main_observation_path))):
            
            
            if self.global_params.use_lsqr[indobs] == 'True':
                # If we used the lsq function, it means that our data is contaminated by the starlight difraction
                # so the model is the sum of the planet model + the estimated stellar contribution
                spectra = list(spectra) # Transform spectra to a list so that we can modify its values
                spectra[indobs] = list(spectra[indobs])
                model, planet_contribution, stellar_contribution, star_flx, systematics, transm = spectra[indobs][3], spectra[indobs][9], spectra[indobs][10], spectra[indobs][11], spectra[indobs][12], spectra[indobs][13]

            if len(spectra[indobs][0]) != 0:

                if (len(systematics) > 0) and (len(star_flx) > 0):
                    data = spectra[indobs][1] - np.dot(stellar_contribution, star_flx[0].T) - systematics
                elif (len(star_flx) > 0):     # if len(systematics) = 0 but len(star_flx) > 0
                    data = spectra[indobs][1] - np.dot(stellar_contribution, star_flx[0].T)
                elif (len(systematics) > 0):  # if len(star_flx) = 0 but len(systematics) > 0
                    data = spectra[indobs][1] - systematics
                else:                         # if len(star_flx) = 0 and len(systematics) = 0
                    data = spectra[indobs][1]

                wave = spectra[indobs][0]
                planet_model = planet_contribution * model 
                
                # Compute intrinsic resolution of the data because of the v.sini
                resolution = 3.0*1e5 / (self.theta_best[self.theta_index == 'vsini'])
                resolution = resolution * np.ones(len(wave))
                
                if data_resolution > 0:
                    self.global_params.custom_reso[indobs] = 'NA'
                    resolution_data = data_resolution * np.ones(len(wave))
                    data_broadened = vsini_fct_accurate(wave, data, 0.6, self.theta_best[self.theta_index == 'vsini'])
                    data_broadened = resolution_decreasing(self.global_params, wave, [], resolution, wave, data, resolution_data, 'mod')
                    planet_model_broadened = resolution_decreasing(self.global_params, wave, [], resolution, wave, planet_model, resolution_data, 'mod')
       
                
                fig = plt.figure('comp_model', figsize=figsize)
                fig.clf()
                ax = fig.add_subplot(111)
                
                ax.plot(wave, data_broadened, c='k')
                ax.plot(wave, planet_model_broadened, c='r')
                
                ax.set_xlabel(r'wavelength ($\mu$m)')
                ax.set_ylabel('Flux (ADU)')
                
                ax1.plot(wave, data, c='k')
                ax1.plot(wave, planet_model, c = 'r')
                
                if self.global_params.use_lsqr[indobs] == 'True':
                    legend_data = 'data - star'
                else:
                    legend_data = 'data'
                    
                ax.legend([legend_data, 'planet model'])
                
                pdf.savefig()
                
        pdf.close()
        
        ax1.legend([legend_data, "planet model"], fontsize = 18)
        ax1.set_xlabel(r'wavelength ($ \mu $m)', fontsize=18)
        ax1.set_ylabel('Flux (ADU)', fontsize=18)
        plt.figure(fig1)
        plt.savefig(self.global_params.result_path + 'Planet_model_and_data.pdf')
                            
        return fig1, ax1
    
    
    def plot_ccf(self, figsize = (10,5), rv_grid = [-300,300], rv_step = 0.5, window_normalisation = 100, model_wavelength = [], model_spectra = [], model_resolution = [], model_name = 'Full', rv_cor=0):
        '''
        Plot the cross-correlation function. It is used for high resolution spectroscopy.

        Args:
            figsize                   (tuple): (default = (10, 5)) Size of the plot
            rv_grid                    (list): (default = [-300,300]) Maximum and minumum values of the radial velocity shift (in km/s)
            rv_step                   (float): (default = 0.5) Radial velocity shift steps (in km/s)
            window_normalisation        (int): (default = 100) ?
            model_wavelength           (list): (default = []) ?
            model_spectra =            (list): (default = []) ?
            model_resolution           (list): (default = []) ?
            model_name                  (str): (default = 'Full') ?
            rv_cor                      (int): (default = 0) ?
        Returns:
            - fig1                    (object) : matplotlib figure object
            - ax1                     (object) : matplotlib axes objects
            - rv_grid                    (list): Radial velocity grid
            - ccf                        (list): Cross-correlation function
            - acf                        (list): Auto-correlation function  

        Author: Allan Denis
        '''
        print('ForMoSA - CCF plot')
        
        
        fig1, ax1 = plt.subplots(1,1, figsize=figsize)
        
        rv_grid = np.arange(rv_grid[0], rv_grid[1], rv_step)
        ccf = np.array([])
        acf = np.array([])
        
        spectra, ck = self._get_spectra(self.theta_best)
        
        model_array = np.array([])
        wave_array = np.array([])
        err_array = np.array([])
        transm_array = np.array([])
        data_array = np.array([])
        
        for indobs, obs in enumerate(sorted(glob.glob(self.global_params.main_observation_path))):
            
            if self.global_params.use_lsqr[indobs] == 'True':
                # If we used the lsq function, it means that our data is contaminated by the starlight difraction
                # so the model is the sum of the planet model + the estimated stellar contribution
                spectra = list(spectra) # Transform spectra to a list so that we can modify its values
                spectra[indobs] = list(spectra[indobs])
                wavelength, err, model, stellar_contribution, star_flx, systematics = spectra[indobs][0], spectra[indobs][2], spectra[indobs][3], spectra[indobs][10], spectra[indobs][11], spectra[indobs][12]
                transm = spectra[indobs][13]
     

            if len(spectra[indobs][0]) != 0:

                if (len(systematics) > 0) and (len(star_flx) > 0):
                    data = spectra[indobs][1] - np.dot(stellar_contribution, star_flx[0].T) - systematics
                elif (len(star_flx) > 0):     # if len(systematics) = 0 but len(star_flx) > 0
                    data = spectra[indobs][1] - np.dot(stellar_contribution, star_flx[0].T)
                elif (len(systematics) > 0):  # if len(star_flx) = 0 but len(systematics) > 0
                    data = spectra[indobs][1] - systematics
                else:                         # if len(star_flx) = 0 and len(systematics) = 0
                    data = spectra[indobs][1]

                _, _, _, model = doppler_fct(wavelength, data, err, model, -self.theta_best[self.theta_index == 'rv'])
                    
                if len(model_spectra) == 0:
                    model_array = np.append(model_array, model)
                else:
                    obs_spectro, _, _, _, _ = adapt_observation_range(self.global_params, indobs=indobs)
                    res_obs = obs_spectro[0][3]
                    res_obs_interp = interp1d(obs_spectro[0][0], res_obs, fill_value = 'extrapolate')
                    res_obs = res_obs_interp(wavelength)
                    
                    ind = np.where((model_wavelength >= wavelength[0]) & (model_wavelength <= wavelength[-1]))
                    model_wavelength_adapt, model_resolution_adapt, model_spectra_adapt = model_wavelength[ind], model_resolution[ind], model_spectra[ind]
                    model_resolution_interp = interp1d(model_wavelength_adapt, model_resolution_adapt, fill_value = 'extrapolate')
                    model_resolution_adapt = model_resolution_interp(wavelength)
                    model_adapted = resolution_decreasing(self.global_params, wavelength, [], res_obs, model_wavelength_adapt, model_spectra_adapt, model_resolution_adapt,
                                                        'mod', indobs=indobs)
                    
                    
                    if len(transm) > 0:
                        _, _, model_adapted, _, _, _ = lsq_fct(spectra[indobs][1], err, star_flx, transm, model_adapted, systematics)
                        
                    model_adapted = vsini_fct_accurate(wavelength, model_adapted, 0.6, self.theta_best[self.theta_index=='vsini'])
                    
                    model_array = np.append(model_array, model_adapted)
                
                wave_array = np.append(wave_array, wavelength)
                data_array = np.append(data_array, data)
                transm_array = np.append(transm_array, transm)
                err_array = np.append(err_array, err)

        max_ccf = 0
        for rv in tqdm(rv_grid):
            _, _, _, model_doppler = doppler_fct(wave_array, data_array, err_array, model_array, rv+rv_cor)
            ccf = np.append(ccf, np.nansum(model_doppler * data_array))
            acf = np.append(acf, np.nansum(model_doppler * model_array))
            
            if np.nansum(model_doppler * data_array) > max_ccf:
                max_ccf = np.nansum(model_doppler * data_array)
                model_max_ccf = model_doppler
            
        # Rescaling cross-correlation function to estimate a SNR
        acf_norm = acf - np.median(acf[(np.abs(rv_grid) > window_normalisation)])
        ccf_norm = ccf - np.median(ccf[(np.abs(rv_grid-rv_grid[np.argmax(ccf)]) > window_normalisation)])
        ccf_noise = np.std(ccf_norm[(np.abs(rv_grid-rv_grid[np.argmax(ccf)]) > window_normalisation)])
        ccf_norm = ccf_norm / ccf_noise
        
        # Rescaling autocorrelation function to make comparable with cross-correlation function
        acf_norm = acf_norm / np.max(acf_norm) * np.max(ccf_norm)
        
        ax1.plot(rv_grid, ccf_norm, label = 'ccf')
        ax1.plot(rv_grid + rv_grid[np.argmax(ccf_norm)] + rv_cor, acf_norm)
        ax1.axvline(x = rv_grid[np.argmax(ccf_norm)], linestyle = '--', c='C3')
        ax1.set_xlabel('RV (km/s)')
        ax1.set_ylabel('S/N')
        ax1.legend(['ccf', 'acf'])
        print(f'SNR = {np.nanmax(ccf_norm):.1f}, RV = {rv_grid[np.argmax(ccf_norm)]:.1f} km/s')
        #ax1.set_title(f'SNR = {np.nanmax(ccf_norm):.1f}, RV = {rv_grid[np.argmax(ccf_norm)]:.1f} km/s')
        plt.figure(fig1)
        plt.savefig(self.global_params.result_path + 'ccf_' + model_name + '.pdf')

        return fig1, ax1, rv_grid, ccf, acf
    


    
    def plot_PT(self, path_temp_profile, figsize=(6,5), model = 'ExoREM'):
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

        with open(self.global_params.result_path + '/result_' + self.global_params.ns_algo + '.pic', 'rb') as f1:
            result = pickle.load(f1)
            # samples = result.samples
            samples = result['samples']

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
        temperature_grid_xa = xr.open_dataarray(path_temp_profile)
        #Crée les profils de température associés aux points de la grille
        P = temperature_grid_xa.coords['P']
        temperature_profiles = np.full((len(samples), len(P)), np.nan)
        for i in range(0, len(samples)):
            if model == 'ExoREM':
                temperature_profiles[i][:] = temperature_grid_xa.interp(Teff=Teffs[i], logg=loggs[i], MH=MHs[i], CO=COs[i])#, kwargs={'fill_value':'extrapolate'})
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
            Tfit.append(np.percentile(temperature_profiles[:,i], 50))
        #Calcule les percentiles 68 et 96 du profil le plus probable
        Tinf68, Tsup68, Tinf95, Tsup95 = [], [], [], []
        for i in range(0, len(P)):
            Tinf68.append(np.percentile(temperature_profiles[:,i], 16))
            Tsup68.append(np.percentile(temperature_profiles[:,i], 84))
            Tinf95.append(np.percentile(temperature_profiles[:,i], 2))
            Tsup95.append(np.percentile(temperature_profiles[:,i], 98))
        #Plot le profil le plus probable et les percentiles associés

        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        ax.fill_betweenx(P, Tinf95, Tsup95, color=self.color_out, alpha=0.1, label=r'2 $\sigma$')
        ax.fill_betweenx(P, Tinf68, Tsup68, color=self.color_out, alpha=0.2, label=r'1 $\sigma$')
        ax.plot(Tfit, P, c=self.color_out, label='Best fit')
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.set_ylim([max(P), min(P)])
        ax.minorticks_on()
        ax.set_xlabel('Temperature (K)')
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
        
        with open(self.global_params.result_path + '/result_' + self.global_params.ns_algo + '.pic', 'rb') as f1:
            result = pickle.load(f1)
            # samples = result.samples
            samples = result['samples']

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
        cloud_props_grids_xa = xr.open_dataset(path_cloud_profile)
        cloud_prop_grid_xa = cloud_props_grids_xa['P_' + cloud_prop]
        #Crée les profils d'une propriété d'un nuage associés aux points de la grille
        P = cloud_prop_grid_xa.coords['P']
        cloud_prop_profiles = np.full((len(samples), len(P)), np.nan)
        for i in range(0, len(samples)):
            cloud_prop_profiles[i][:] = cloud_prop_grid_xa.interp(Teff=Teffs[i], logg=loggs[i], MH=MHs[i], CO=COs[i])#, kwargs={'fill_value':'extrapolate'})
        # Calcule le profil le plus probable
        propfit = []
        for i in range(0, len(P)):
            propfit.append(np.percentile(cloud_prop_profiles[:, i], 50))
        # Calcule les percentiles 68 et 96 du profil le plus probable
        propinf68, propsup68, propinf95, propsup95 = [], [], [], []
        for i in range(0, len(P)):
            propinf68.append(np.percentile(cloud_prop_profiles[:, i], 16))
            propsup68.append(np.percentile(cloud_prop_profiles[:, i], 84))
            propinf95.append(np.percentile(cloud_prop_profiles[:, i], 2))
            propsup95.append(np.percentile(cloud_prop_profiles[:, i], 98))

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
        ax.set_ylabel('Pressure (Pa)')

        ax.legend(frameon=False)

        return fig, ax
    


