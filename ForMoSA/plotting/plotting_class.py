
from __future__ import print_function, division
import sys, os, yaml
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
        self.theta_index = np.asarray(theta_index)

        posterior_to_plot = []
        for res, results in enumerate(samples):
            if self.global_params.r != 'NA':
                if self.global_params.r[0] == "constant":
                    r_picked = float(self.global_params.r[1])
                else:
                    ind_theta_r = np.where(self.theta_index == 'r')
                    r_picked = results[ind_theta_r[0]]
                lum = np.log10(4 * np.pi * (r_picked * 69911000.) ** 2 * results[0] ** 4 * 5.670e-8 / 3.83e26)
                results = np.concatenate((results, np.asarray(lum)))
            posterior_to_plot.append(results)
        if self.global_params.r != 'NA':
            tot_list_param_title.append(r'log(L/L$\mathrm{_{\odot}}$)')

        self.posterior_to_plot = np.array(posterior_to_plot)
        self.posteriors_names = tot_list_param_title




    def plot_corner(self, levels_sig=[0.997, 0.95, 0.68], bins=100, quantiles=(0.16, 0.5, 0.84), burn_in=0):
        '''
        See the corner plots
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
                axs[i, j].plot(self.posterior_to_plot[:,n], color=self.color_out, alpha=0.8)
                axs[i, j].set_ylabel(self.posteriors_names[n])
                if n == len(self.posteriors_names)-1:
                    break
                else:
                    n+=1
        
        return fig, axs


    def plot_radar(self,ranges,label='',quantiles=[0.16, 0.5, 0.84],chiffres=[0,2,2,2]):
        '''
        To check overall the distribution of the parameters 

        Inputs:
        ranges
        '''
        print('ForMoSA - Radar plot')

        self._get_posteriors()

        list_posteriors = []
        list_uncert_down = []
        list_uncert_up = []
        for l in range(len(self.posterior_to_plot[1,:])):
            q16, q50, q84 = corner.quantile(self.posterior_to_plot[:,l], quantiles)
            
            list_posteriors.append(q50)
            list_uncert_down.append(q16)
            list_uncert_up.append(q84)

        fig1 = plt.figure(figsize=(6, 6))
        radar = ComplexRadar(fig1, self.posteriors_names, ranges)

        radar.plot(list_posteriors, 'o-', color=self.color_out, label=label)
        radar.fill_between(list_uncert_down,list_uncert_up, color=self.color_out, alpha=0.2)

        radar.ax.legend(loc='center', bbox_to_anchor=(0.5, -0.20),frameon=False, ncol=2)
        
        return fig1, radar.ax


    def _get_spectra(self,theta):
        '''
        To get the data and best model asociated 
        Use numba: https://numba.pydata.org/

        (Adapted from Simon Petrus)
        '''
        #def get_spec(theta, theta_index, global_params, for_plot='no'):
        # Recovery of the spectroscopy and photometry data
        spectrum_obs = np.load(self.global_params.result_path + '/spectrum_obs.npz', allow_pickle=True)
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
        path_grid_m = self.global_params.adapt_store_path + '/adapted_grid_merge_' + self.global_params.grid_name + '_nonan.nc'
        path_grid_p = self.global_params.adapt_store_path + '/adapted_grid_phot_' + self.global_params.grid_name + '_nonan.nc'
        ds = xr.open_dataset(path_grid_m, decode_cf=False, engine='netcdf4')
        grid_merge = ds['grid']
        ds.close()
        ds = xr.open_dataset(path_grid_p, decode_cf=False, engine='netcdf4')
        grid_phot = ds['grid']
        ds.close()

        if self.global_params.par3 == 'NA':
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
        elif self.global_params.par4 == 'NA':
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
        elif self.global_params.par5 == 'NA':
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
        modif_spec_chi2 = modif_spec(self.global_params, theta, self.theta_index,
                                     wav_obs_merge, flx_obs_merge, err_obs_merge, flx_mod_merge,
                                     wav_obs_phot, flx_obs_phot, err_obs_phot, flx_mod_phot)

        return modif_spec_chi2


    
    def plot_fit(self, figsize=(10, 5), uncert='yes'):
        '''
        Plot the best fit comparing with the data.

        Args:
            global_params: Class containing each parameter used in ForMoSA
            save: If ='yes' save the figure in a pdf format
        Returns:

        Author: Paulina Palma-Bifani
        '''
        print('ForMoSA - Best fit and residuals plot')

        fig = plt.figure(figsize=figsize)
        fig.tight_layout()
        size = (7,11)
        ax = plt.subplot2grid(size, (0, 0),rowspan=5 ,colspan=10)
        axr= plt.subplot2grid(size, (5, 0),rowspan=2 ,colspan=10)
        axr2= plt.subplot2grid(size, (5, 10),rowspan=2 ,colspan=1)



        with open(self.global_params.result_path + '/result_' + self.global_params.ns_algo + '.pic', 'rb') as ns_result:
            result = pickle.load(ns_result)
        samples = result.samples
        logl = result.logl
        ind = np.where(logl == max(logl))
        theta_best = samples[ind][0]

        spectra = self._get_spectra(theta_best)

        if uncert=='yes':
            ax.errorbar(spectra[0], spectra[1], yerr=spectra[2], c='k', alpha=0.2)
        ax.plot(spectra[0], spectra[1], c='k', label = 'data')
        ax.plot(spectra[0], spectra[3], c=self.color_out, alpha=0.8, label='model')

        residuals = spectra[3] - spectra[1]
        sigma_res = np.std(residuals)
        axr.plot(spectra[0], residuals/sigma_res, c=self.color_out, alpha=0.8, label='model-data')
        axr.axhline(y=0, color='k', alpha=0.5, linestyle='--')
        axr2.hist(residuals/sigma_res, bins=100 ,color=self.color_out, alpha=0.5, density=True, orientation='horizontal', label='density')
        axr2.axis('off')

        axr.set_xlabel(r'Wavelength (µm)')
        ax.set_ylabel(r'Flux (W m-2 µm-1)')
        axr.set_ylabel(r'Residuals ($\sigma$)')
        
        ax.legend(frameon=False)
        axr.legend(frameon=False)
        axr2.legend(frameon=False,handlelength=0)

        # define the data as global
        self.spectra = spectra
        self.residuals = residuals

        return fig, ax, axr, axr2


    def plot_PT(self,figsize=(10,7)):
        '''
        Plot the Pressure-Temperature profiles 
        Calculates the most probable temperature profile

        Return: 

        Adapted from Nathan Zimniak
        Author: Paulina Palma-Bifani
        '''
        print('ForMoSA - Pressure-Temperature profile')

        model = 'ExoREM'
        path_temp_profile = '/Users/ppalmabifani/Desktop/exoAtm/c0_ForMoSA/ForMoSA/DEMO/outputs/pt_profiles/temperature_grid_xarray_ExoREM.nc'

        with open(self.global_params.result_path + '/result_' + self.global_params.ns_algo + '.pic', 'rb') as f1:
            result = pickle.load(f1)
            samples = result.samples

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
        ax.fill_betweenx(P, Tinf95, Tsup95, color=self.color_out, alpha=0.2, label='2 $\sigma$')
        ax.fill_betweenx(P, Tinf68, Tsup68, color=self.color_out, alpha=0.3, label='1 $\sigma$')
        ax.plot(Tfit, P, c=self.color_out, label='PT profile')
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.set_ylim([max(P), min(P)])
        ax.minorticks_on()
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Pressure (Pa)')
        ax.legend(frameon=False)

        return fig
       

    


    

        


































