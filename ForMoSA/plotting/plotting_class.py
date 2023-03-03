
from __future__ import print_function, division
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
import corner
import xarray as xr
import pickle

# Import ForMoSA
from main_utilities import GlobFile
from nested_sampling.nested_modif_spec import modif_spec


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class ComplexRadar():
    '''
    Original from Damian Cummins: https://github.com/DamianCummins/statsbomb-football-event-visualisations/blob/master/Statsbomb%20Womens%20World%20Cup%202019%20visualisation.ipynb
    
    Adapted by: P. Palma-Bifani
    '''

    def __init__(self, fig, variables, ranges, n_ordinate_levels=6):
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
        sdata = self.scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = self.scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    
    def fill_between(self, list_down, list_up, *args, **kw):
        sdata_down = self.scale_data(list_down, self.ranges)
        sdata_up = self.scale_data(list_up, self.ranges)
        self.ax.fill_between(self.angle,np.r_[sdata_down,sdata_down[0]], np.r_[sdata_up,sdata_up[0]], *args, **kw)

    def scale_data(self, data, ranges):
        """scales data[1:] to ranges[0]"""
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
    Here all the plotting functionalities of ForMoSA to see the 

    Author: Paulina Palma-Bifani 
    '''

    def __init__(self, config_file_path, color_out):
        '''
        Plotting class initializer
        '''
        
        self.global_params = GlobFile(config_file_path)  
        self.color_out     = color_out



    def _get_posteriors(self):
        '''
        Function to get the posteriors, including luminosity derivation and corvengence parameters logz

        (Adapted from Simon Petrus plotting functions)
        '''
        with open(self.global_params.result_path + '/result_' + self.global_params.ns_algo + '.pic', 'rb') as open_pic:
            result = pickle.load(open_pic)
            samples = result.samples
            self.weights = result.weights

            # To test the quality of the fit
            self.sample_logz    = round(result['logz'],1)
            self.sample_logzerr = round(result['logzerr'],1)
            self.sample_h       = round(result['h'],1)
            self.outputs_string = 'logz = '+ str(self.sample_logz)+' ± '+str(self.sample_logzerr)+ ' ; h = '+str(self.sample_h)
        
        ds = xr.open_dataset(self.global_params.model_path, decode_cf=False, engine='netcdf4')
        attrs = ds.attrs
        extra_parameters = [['r', 'R', r'(R$\mathrm{_{Jup}}$)'],
                            ['d', 'd', '(pc)'],
                            ['rv', 'RV', r'(km.s$\mathrm{^{-1}}$)'],
                            ['av', 'Av', '(mag)'],
                            ['vsini', 'v.sin(i)', r'(km.s$\mathrm{^{-1}}$)'],
                            ['ld', 'limb darkening', '']
                            ]

        tot_list_param_title = []
        theta_index = []
        if self.global_params.par1 != 'NA':
            tot_list_param_title.append(attrs['title'][0] + ' ' + attrs['title'][0])
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

        if self.global_params.r != 'NA' and self.global_params.r[0] != 'constant':
            tot_list_param_title.append(extra_parameters[0][1] + ' ' + extra_parameters[0][2])
            theta_index.append('r')
        if self.global_params.d != 'NA' and self.global_params.d[0] != 'constant':
            tot_list_param_title.append(extra_parameters[1][1] + ' ' + extra_parameters[1][2])
            theta_index.append('d')
        if self.global_params.rv != 'NA' and self.global_params.rv[0] != 'constant':
            tot_list_param_title.append(extra_parameters[2][1] + ' ' + extra_parameters[2][2])
            theta_index.append('rv')
        if self.global_params.av != 'NA' and self.global_params.av[0] != 'constant':
            tot_list_param_title.append(extra_parameters[3][1] + ' ' + extra_parameters[3][2])
            theta_index.append('av')
        if self.global_params.vsini != 'NA' and self.global_params.vsini[0] != 'constant':
            tot_list_param_title.append(extra_parameters[4][1] + ' ' + extra_parameters[4][2])
            theta_index.append('vsini')
        if self.global_params.ld != 'NA' and self.global_params.ld[0] != 'constant':
            tot_list_param_title.append(extra_parameters[5][1] + ' ' + extra_parameters[5][2])
            theta_index.append('ld')
        theta_index = np.asarray(theta_index)

        posterior_to_plot = []
        for res, result in enumerate(samples):
            if self.global_params.r != 'NA':
                if self.global_params.r[0] == "constant":
                    r_picked = float(self.global_params.r[1])
                else:
                    ind_theta_r = np.where(theta_index == 'r')
                    r_picked = result[ind_theta_r[0]]
                lum = np.log10(4 * np.pi * (r_picked * 69911000.) ** 2 * result[0] ** 4 * 5.670e-8 / 3.83e26)
                result = np.concatenate((result, np.asarray([lum])))
            posterior_to_plot.append(result)
        if self.global_params.r != 'NA':
            tot_list_param_title.append(r'log(L/L$\mathrm{_{\odot}}$)')

        self.posterior_to_plot = np.array(posterior_to_plot)
        self.posteriors_names = tot_list_param_title




    def plot_corner(self, levels_sig=[0.997, 0.95, 0.68], bins=100, quantiles=(0.16, 0.5, 0.84)):
        '''
        See the corner plots
        '''
        print('ForMoSA - Corner plot')

        self._get_posteriors()

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
                        title_kwargs=dict(fontsize=10),
                        contour_kwargs=dict(colors=self.color_out, linewidths=0.7),
                        pcolor_kwargs=dict(color='red'),
                        label_kwargs=dict(fontsize=10))

        fig.supxlabel(self.outputs_string, va='top')

        return fig


    def plot_chains(self,figsize=(7,15)):
        '''
        To check the convergence of the chains

        '''
        print('ForMoSA - Posteriors chains for each parameter')

        self._get_posteriors()
        
        col = int(len(self.posterior_to_plot[0][:])/2)+int(len(self.posterior_to_plot[0][:])%2)
        fig, axs = plt.subplots(col, 2, figsize=figsize)

        n=0
        for i in range(col):
            for j in range(2):
                axs[i, j].plot(self.posterior_to_plot[:,i], color=self.color_out, alpha=0.8)
                axs[i, j].set_ylabel(self.posteriors_names[i])
                n+=1
        
        return fig, axs


    def plot_radar(self,ranges,label='',quantiles=[0.16, 0.5, 0.84],chiffres=[0,2,2,2]):
        '''
        To check overall the distribution of the parameters 

        Inputs:
        ranges
        '''

        self._get_posteriors()

        fig1 = plt.figure(figsize=(6, 6))
        radar = ComplexRadar(fig1, self.posteriors_names, ranges)

        list_posteriors=[]
        list_uncert_down=[]
        list_uncert_up=[]
        for l in range(len(self.posterior_to_plot[1,:])):
            q16, q50, q84 = corner.quantile(self.posterior_to_plot[:,l], quantiles)
            latex_value_1 = round(q50, chiffres[l])

            list_posteriors.append(latex_value_1)
            list_uncert_down.append(q16)
            list_uncert_up.append(q84)

        radar.plot(list_posteriors, 'o-', color=self.color_out, label=label)
        radar.fill_between(list_uncert_down,list_uncert_up, color=self.color_out, alpha=0.2)

        radar.ax.legend(loc='center', bbox_to_anchor=(0.5, -0.20),frameon=False, ncol=2)
        
        return fig1, radar.ax


    def _get_spectra(self,):
        '''
        To get the data and best model asociated 
        Use numba: https://numba.pydata.org/

        (Adapted from Simon Petrus plotting functions)
        '''
        #def get_spec(theta, theta_index, global_params, for_plot='no'):
        # Recovery of the spectroscopy and photometry data
        spectrum_obs = np.load(global_params.result_path + '/spectrum_obs.npz', allow_pickle=True)
        wav_obs_merge = spectrum_obs['obs_merge'][0]
        flx_obs_merge = spectrum_obs['obs_merge'][1]
        err_obs_merge = spectrum_obs['obs_merge'][2]
        if 'obs_pho' in spectrum_obs.keys():
            wav_obs_phot = np.asarray(spectrum_obs['obs_pho'][0])
            flx_obs_phot = np.asarray(spectrum_obs['obs_pho'][1])
            err_obs_phot = np.asarray(spectrum_obs['obs_pho'][2])
        else:
            wav_obs_phot = np.asarray([])
            flx_obs_phot = np.asarray([])
            err_obs_phot = np.asarray([])

        # Recovery of the spectroscopy and photometry model
        path_grid_m = global_params.adapt_store_path + '/adapted_grid_merge_' + global_params.grid_name + '_nonan.nc'
        path_grid_p = global_params.adapt_store_path + 'adapted_grid_phot_' + global_params.grid_name + '_nonan.nc'
        ds = xr.open_dataset(path_grid_m, decode_cf=False, engine='netcdf4')
        grid_merge = ds['grid']
        ds.close()
        ds = xr.open_dataset(path_grid_p, decode_cf=False, engine='netcdf4')
        grid_phot = ds['grid']
        ds.close()

        if global_params.par3 == 'NA':
            if len(grid_merge['wavelength']) != 0:
                flx_mod_merge = grid_merge.interp(par1=theta[0], par2=theta[1],
                                                          method="linear", kwargs={"fill_value": "extrapolate"})
            else:
                flx_mod_merge = []
            if len(grid_phot['wavelength']) != 0:
                flx_mod_phot = grid_phot.interp(par1=theta[0], par2=theta[1],
                                                        method="linear", kwargs={"fill_value": "extrapolate"})
            else:
                flx_mod_phot = []
        elif global_params.par4 == 'NA':
            if len(grid_merge['wavelength']) != 0:
                flx_mod_merge = grid_merge.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                          method="linear", kwargs={"fill_value": "extrapolate"})
            else:
                flx_mod_merge = []
            if len(grid_phot['wavelength']) != 0:
                flx_mod_phot = grid_phot.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                        method="linear", kwargs={"fill_value": "extrapolate"})
            else:
                flx_mod_phot = []
        elif global_params.par5 == 'NA':
            if len(grid_merge['wavelength']) != 0:
                flx_mod_merge = grid_merge.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                          method="linear", kwargs={"fill_value": "extrapolate"})
            else:
                flx_mod_merge = []
            if len(grid_phot['wavelength']) != 0:
                flx_mod_phot = grid_phot.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                        method="linear", kwargs={"fill_value": "extrapolate"})
            else:
                flx_mod_phot = []
        else:
            if len(grid_merge['wavelength']) != 0:
                flx_mod_merge = grid_merge.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                          par5=theta[4],
                                                          method="linear", kwargs={"fill_value": "extrapolate"})
            else:
                flx_mod_merge = []
            if len(grid_phot['wavelength']) != 0:
                flx_mod_phot = grid_phot.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                        par5=theta[4],
                                                        method="linear", kwargs={"fill_value": "extrapolate"})
            else:
                flx_mod_phot = []

        # Modification of the synthetic spectrum with the extra-grid parameters
        modif_spec_chi2 = modif_spec(global_params, theta, theta_index,
                                     wav_obs_merge, flx_obs_merge, err_obs_merge, flx_mod_merge,
                                     wav_obs_phot, flx_obs_phot, err_obs_phot, flx_mod_phot)

        return modif_spec_chi2


    
    def plot_fit(self,):
        '''
        Plot the best fit comparing with the data.

        Args:
            global_params: Class containing each parameter used in ForMoSA
            save: If ='yes' save the figure in a pdf format
        Returns:

        Author: Simon Petrus
        '''
        figure_fit = plt.figure(figsize=(10, 5))

        inter_ext_g = 0.1
        inter_ext_d = 0.1
        inter_ext_h = 0.05
        inter_ext_b = 0.15

        b_fits = inter_ext_b
        g_fits = inter_ext_g
        l_fits = 1-inter_ext_g-inter_ext_d
        h_fits = 1-inter_ext_b-inter_ext_h
        fits = figure_fit.add_axes([g_fits, b_fits, l_fits, h_fits])
        fits.set_xlabel(r'Wavelength (um)', labelpad=5)
        fits.set_ylabel(r'Flux (W.m-2.um-1)', labelpad=5)

        ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine='netcdf4')

        with open(global_params.result_path + '/result_' + global_params.ns_algo + '.pic', 'rb') as ns_result:
            result = pickle.load(ns_result)
        samples = result.samples
        logl = result.logl
        ind = np.where(logl == max(logl))
        theta_best = samples[ind][0]

        if global_params.par1 != 'NA':
            theta_index = ['par1']
        if global_params.par2 != 'NA':
            theta_index.append('par2')
        if global_params.par3 != 'NA':
            theta_index.append('par3')
        if global_params.par4 != 'NA':
            theta_index.append('par4')
        if global_params.par5 != 'NA':
            theta_index.append('par5')
        n_free_parameters = len(ds.attrs['key'])
        if global_params.r != 'NA' and global_params.r[0] != 'constant':
            n_free_parameters += 1
            theta_index.append('r')
        if global_params.d != 'NA' and global_params.d[0] != 'constant':
            n_free_parameters += 1
            theta_index.append('d')
        if global_params.rv != 'NA' and global_params.rv[0] != 'constant':
            n_free_parameters += 1
            theta_index.append('rv')
        if global_params.av != 'NA' and global_params.av[0] != 'constant':
            n_free_parameters += 1
            theta_index.append('av')
        if global_params.vsini != 'NA' and global_params.vsini[0] != 'constant':
            n_free_parameters += 1
            theta_index.append('vsini')
        if global_params.ld != 'NA' and global_params.ld[0] != 'constant':
            n_free_parameters += 1
            theta_index.append('ld')
        theta_index = np.asarray(theta_index)
        spectra = get_spec(theta_best, theta_index, global_params, for_plot='yes')

        if global_params.model_name == 'SONORA':
            col_pair = 'peru'
        if global_params.model_name == 'ATMO':
            col_pair = 'darkgreen'
        if global_params.model_name == 'BTSETTL':
            col_pair = 'firebrick'
        if global_params.model_name == 'EXOREM':
            col_pair = 'mediumblue'
        if global_params.model_name == 'DRIFTPHOENIX':
            col_pair = 'darkviolet'
        fits.plot(spectra[0], spectra[1], c='k')
        fits.scatter(spectra[4], spectra[5], c='k')
        fits.plot(spectra[0], spectra[3], c=col_pair)
        fits.scatter(spectra[4], spectra[7], c=col_pair)

        for ns_u_ind, ns_u in enumerate(global_params.wav_fit.split('/')):
            min_ns_u = float(ns_u.split(',')[0])
            max_ns_u = float(ns_u.split(',')[1])
            fits.fill_between([min_ns_u, max_ns_u],
                              [min(min(spectra[1]), min(spectra[3]))],
                              [max(max(spectra[1]), max(spectra[3]))],
                              color='y',
                              alpha=0.2
                              )

        # plt.show()
        #if save != 'no':
        #   figure_fit.savefig(global_params.result_path + '/result_' + global_params.ns_algo + '_fit.pdf',
        #                       bbox_inches='tight', dpi=300)


    def plot_PT(self,):
        '''
        Plot the Pressure-Temperature profiles 
        '''
        print('ForMoSA - PT profile')
       

    


    

        


































