import numpy as np
import os
import glob
import nestle
import time
import xarray as xr
import pickle
from scipy.stats import pearsonr
from scipy.interpolate import interp1d, UnivariateSpline

from nested_sampling.nested_modif_spec import modif_spec
from nested_sampling.nested_prior_function import uniform_prior, gaussian_prior
from nested_sampling.nested_logL_functions import logL_chi2_classic, logL_chi2_covariance, logL_CCF_Brogi, logL_CCF_Lockwood, logL_CCF_custom
from nested_sampling.nested_MOSAIC import MOSAIC_logL
from main_utilities import yesno, diag_mat

c = 299792.458 # Speed of light in km/s


def import_obsmod(global_params):
    """
    Function to import spectra (model and data) before the inversion

    Args:
        
    Returns:
        

    Authors: Matthieu Ravet (adapted from Simon Petrus)
    """
    # Check if the MOSAIC mode is activated
    if global_params.observation_format == 'MOSAIC':
        main_obs_path = global_params.main_observation_path

        main_file = []

        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            
            global_params.observation_path = obs
            obs_name = os.path.splitext(os.path.basename(global_params.observation_path))[0]
            spectrum_obs = np.load(os.path.join(global_params.result_path, f'spectrum_obs_{obs_name}.npz'), allow_pickle=True)

            wav_obs_merge = spectrum_obs['obs_merge'][0]
            flx_obs_merge = spectrum_obs['obs_merge'][1]
            err_obs_merge = spectrum_obs['obs_merge'][2]
            inv_cov_obs_merge = spectrum_obs['inv_cov_obs']

            if 'obs_pho' in spectrum_obs.keys():
                wav_obs_phot = np.asarray(spectrum_obs['obs_pho'][0])
                flx_obs_phot = np.asarray(spectrum_obs['obs_pho'][1])
                err_obs_phot = np.asarray(spectrum_obs['obs_pho'][2])
            else:
                wav_obs_phot = np.asarray([])
                flx_obs_phot = np.asarray([])
                err_obs_phot = np.asarray([])

            # Recovery of the spectroscopy and photometry model
            path_grid_m = os.path.join(global_params.adapt_store_path, f'adapted_grid_merge_{global_params.grid_name}_{obs_name}_nonan.nc')
            path_grid_p = os.path.join(global_params.adapt_store_path, f'adapted_grid_phot_{global_params.grid_name}_{obs_name}_nonan.nc')
            ds = xr.open_dataset(path_grid_m, decode_cf=False, engine='netcdf4')
            grid_merge = ds['grid']
            ds.close()
            ds = xr.open_dataset(path_grid_p, decode_cf=False, engine='netcdf4')
            grid_phot = ds['grid']
            ds.close()

            main_file.append([[wav_obs_merge, wav_obs_phot], [flx_obs_merge, flx_obs_phot], [err_obs_merge, err_obs_phot], inv_cov_obs_merge, grid_merge, grid_phot])

    else:
        # Recovery of spectroscopy and photometry data
        spectrum_obs = np.load(global_params.result_path + '/spectrum_obs.npz', allow_pickle=True)

        wav_obs_merge = spectrum_obs['obs_merge'][0]
        flx_obs_merge = spectrum_obs['obs_merge'][1]
        err_obs_merge = spectrum_obs['obs_merge'][2]
        inv_cov_obs_merge = spectrum_obs['inv_cov_obs']

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

        main_file = [wav_obs_merge, wav_obs_phot], [flx_obs_merge, flx_obs_phot], [err_obs_merge, err_obs_phot], inv_cov_obs_merge, grid_merge, grid_phot

    return main_file






def loglike(theta, theta_index, global_params, main_file, for_plot='no'):
    """
    

    Args:
        
    Returns:
        

    Authors: Simon Petrus and Matthieu Ravet
    """

    # Check if we are running with the MOSAIC mode

    if global_params.observation_format == 'MOSAIC':
        FINAL_logL = MOSAIC_logL(theta, theta_index, global_params, main_file)

    else:
        # Recovery of spectroscopy and photometry data
        wav_obs_merge = main_file[0][0]
        wav_obs_phot = main_file[0][1]
        flx_obs_merge = main_file[1][0]
        flx_obs_phot = main_file[1][1]
        err_obs_merge = main_file[2][0]
        err_obs_phot = main_file[2][1]
        inv_cov_obs_merge = main_file[3]

        # Recovery of the spectroscopy and photometry model
        grid_merge = main_file[4]
        grid_phot = main_file[5]

        # Calculation of the likelihood for each sub-spectrum defined by the parameter 'wav_fit'
        for ns_u_ind, ns_u in enumerate(global_params.wav_fit.split('/')):
            
            min_ns_u = float(ns_u.split(',')[0])
            max_ns_u = float(ns_u.split(',')[1])
            ind_grid_merge_sel = np.where((grid_merge['wavelength'] >= min_ns_u) & (grid_merge['wavelength'] <= max_ns_u))
            ind_grid_phot_sel = np.where((grid_phot['wavelength'] >= min_ns_u) & (grid_phot['wavelength'] <= max_ns_u))

            # Cutting of the grid on the wavelength grid defined by the parameter 'wav_fit'
            grid_merge_cut = grid_merge.sel(wavelength=grid_merge['wavelength'][ind_grid_merge_sel])
            grid_phot_cut = grid_phot.sel(wavelength=grid_phot['wavelength'][ind_grid_phot_sel])

            # Interpolation of the grid at the theta parameters set
            if global_params.par3 == 'NA':
                if len(grid_merge_cut['wavelength']) != 0:
                    flx_mod_merge_cut = grid_merge_cut.interp(par1=theta[0], par2=theta[1],
                                                            method="linear", kwargs={"fill_value": "extrapolate"})
                else:
                    flx_mod_merge_cut = []
                if len(grid_phot_cut['wavelength']) != 0:
                    flx_mod_phot_cut = grid_phot_cut.interp(par1=theta[0], par2=theta[1],
                                                            method="linear", kwargs={"fill_value": "extrapolate"})
                else:
                    flx_mod_phot_cut = []
            elif global_params.par4 == 'NA':
                if len(grid_merge_cut['wavelength']) != 0:
                    flx_mod_merge_cut = grid_merge_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                            method="linear", kwargs={"fill_value": "extrapolate"})
                else:
                    flx_mod_merge_cut = []
                if len(grid_phot_cut['wavelength']) != 0:
                    flx_mod_phot_cut = grid_phot_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                            method="linear", kwargs={"fill_value": "extrapolate"})
                else:
                    flx_mod_phot_cut = []
            elif global_params.par5 == 'NA':
                if len(grid_merge_cut['wavelength']) != 0:
                    flx_mod_merge_cut = grid_merge_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            method="linear", kwargs={"fill_value": "extrapolate"})
                else:
                    flx_mod_merge_cut = []
                if len(grid_phot_cut['wavelength']) != 0:
                    flx_mod_phot_cut = grid_phot_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            method="linear", kwargs={"fill_value": "extrapolate"})
                else:
                    flx_mod_phot_cut = []
            else:
                if len(grid_merge_cut['wavelength']) != 0:
                    flx_mod_merge_cut = grid_merge_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            par5=theta[4],
                                                            method="linear", kwargs={"fill_value": "extrapolate"})
                else:
                    flx_mod_merge_cut = []
                if len(grid_phot_cut['wavelength']) != 0:
                    flx_mod_phot_cut = grid_phot_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            par5=theta[4],
                                                            method="linear", kwargs={"fill_value": "extrapolate"})
                else:
                    flx_mod_phot_cut = []


            # Re-merging of the data and interpolated synthetic spectrum to a wavelength grid defined by the parameter 'wav_fit'
            ind_merge = np.where((wav_obs_merge >= min_ns_u) & (wav_obs_merge <= max_ns_u))
            ind_phot = np.where((wav_obs_phot >= min_ns_u) & (wav_obs_phot <= max_ns_u))
            if ns_u_ind == 0:
                wav_obs_merge_ns_u = wav_obs_merge[ind_merge]
                flx_obs_merge_ns_u = flx_obs_merge[ind_merge]
                err_obs_merge_ns_u = err_obs_merge[ind_merge]
                flx_mod_merge_ns_u = flx_mod_merge_cut
                wav_obs_phot_ns_u = wav_obs_phot[ind_phot]
                flx_obs_phot_ns_u = flx_obs_phot[ind_phot]
                err_obs_phot_ns_u = err_obs_phot[ind_phot]
                flx_mod_phot_ns_u = flx_mod_phot_cut
                if inv_cov_obs_merge != []:  # Add covariance in the loop (if necessary)
                    inv_cov_obs_merge_ns_u = inv_cov_obs_merge[np.ix_(ind_merge[0],ind_merge[0])]
                else:
                    inv_cov_obs_merge_ns_u = []
            else:
                wav_obs_merge_ns_u = np.concatenate((wav_obs_merge_ns_u, wav_obs_merge[ind_merge]))
                flx_obs_merge_ns_u = np.concatenate((flx_obs_merge_ns_u, flx_obs_merge[ind_merge]))
                err_obs_merge_ns_u = np.concatenate((err_obs_merge_ns_u, err_obs_merge[ind_merge]))
                flx_mod_merge_ns_u = np.concatenate((flx_mod_merge_ns_u, flx_mod_merge_cut))
                wav_obs_phot_ns_u = np.concatenate((wav_obs_phot_ns_u, wav_obs_phot[ind_phot]))
                flx_obs_phot_ns_u = np.concatenate((flx_obs_phot_ns_u, flx_obs_phot[ind_phot]))
                err_obs_phot_ns_u = np.concatenate((err_obs_phot_ns_u, err_obs_phot[ind_phot]))
                flx_mod_phot_ns_u = np.concatenate((flx_mod_phot_ns_u, flx_mod_phot_cut))
                if inv_cov_obs_merge_ns_u != []: # Merge the covariance matrices (if necessary)
                    inv_cov_obs_merge_ns_u = diag_mat([inv_cov_obs_merge_ns_u, inv_cov_obs_merge[np.ix_(ind_merge[0],ind_merge[0])]]) 

        # Modification of the synthetic spectrum with the extra-grid parameters
        modif_spec_LL = modif_spec(global_params, theta, theta_index,
                                    wav_obs_merge_ns_u,  flx_obs_merge_ns_u,  err_obs_merge_ns_u,  flx_mod_merge_ns_u,
                                    wav_obs_phot_ns_u,  flx_obs_phot_ns_u, err_obs_phot_ns_u,  flx_mod_phot_ns_u)
        
        flx_obs, flx_obs_phot = modif_spec_LL[1], modif_spec_LL[5]
        flx_mod, flx_mod_phot = modif_spec_LL[3], modif_spec_LL[7]
        err, err_phot = modif_spec_LL[2], modif_spec_LL[6]
        inv_cov = inv_cov_obs_merge_ns_u
        ck = modif_spec_LL[8]

        # Computation of the photometry logL
        if err_phot != []:
            logL_phot = logL_chi2_classic(flx_obs_phot-flx_mod_phot, err_phot)
        else:
            logL_phot = 0

        # Computation of the spectroscopy logL
        if global_params.logL_type == 'chi2_classic':
            logL_spec = logL_chi2_classic(flx_obs-flx_mod, err)
        if global_params.logL_type == 'chi2_covariance' and inv_cov != []:
            logL_spec = logL_chi2_covariance(flx_obs-flx_mod, inv_cov)
        if global_params.logL_type == 'CCF_Brogi':
            logL_spec = logL_CCF_Brogi(flx_obs, flx_mod)
        if global_params.logL_type == 'CCF_Lockwood':
            logL_spec = logL_CCF_Lockwood(flx_obs, flx_mod)
        if global_params.logL_type == 'CCF_custom':
            logL_spec = logL_CCF_custom(flx_obs, flx_mod, err)

        FINAL_logL = logL_phot + logL_spec

    if for_plot == 'no':
        return FINAL_logL
    else:
        return modif_spec



def prior_transform(theta, theta_index, lim_param_grid, global_params):
    """
    
    Args:
        
    Returns:
        
    Author: Simon Petrus
    """
    prior = []
    if global_params.par1 != 'NA':
        prior_law = global_params.par1[0]
        if prior_law != 'constant':
            if prior_law == 'uniform':
                prior_par1 = uniform_prior([float(global_params.par1[1]), float(global_params.par1[2])], theta[0])
            if prior_law == 'gaussian':
                prior_par1 = gaussian_prior([float(global_params.par1[1]), float(global_params.par1[2])], theta[0])
            if prior_par1 < lim_param_grid[0][0]:
                prior_par1 = lim_param_grid[0][0]
            elif prior_par1 > lim_param_grid[0][1]:
                prior_par1 = lim_param_grid[0][1]
            prior.append(prior_par1)
    if global_params.par2 != 'NA':
        prior_law = global_params.par2[0]
        if prior_law != 'constant':
            if prior_law == 'uniform':
                prior_par2 = uniform_prior([float(global_params.par2[1]), float(global_params.par2[2])], theta[1])
            if prior_law == 'gaussian':
                prior_par2 = gaussian_prior([float(global_params.par2[1]), float(global_params.par2[2])], theta[1])
            if prior_par2 < lim_param_grid[1][0]:
                prior_par2 = lim_param_grid[1][0]
            elif prior_par2 > lim_param_grid[1][1]:
                prior_par2 = lim_param_grid[1][1]
            prior.append(prior_par2)
    if global_params.par3 != 'NA':
        prior_law = global_params.par3[0]
        if prior_law != 'constant':
            if prior_law == 'uniform':
                prior_par3 = uniform_prior([float(global_params.par3[1]), float(global_params.par3[2])], theta[2])
            if prior_law == 'gaussian':
                prior_par3 = gaussian_prior([float(global_params.par3[1]), float(global_params.par3[2])], theta[2])
            if prior_par3 < lim_param_grid[2][0]:
                prior_par3 = lim_param_grid[2][0]
            elif prior_par3 > lim_param_grid[2][1]:
                prior_par3 = lim_param_grid[2][1]
            prior.append(prior_par3)
    if global_params.par4 != 'NA':
        prior_law = global_params.par4[0]
        if prior_law != 'constant':
            if prior_law == 'uniform':
                prior_par4 = uniform_prior([float(global_params.par4[1]), float(global_params.par4[2])], theta[3])
            if prior_law == 'gaussian':
                prior_par4 = gaussian_prior([float(global_params.par4[1]), float(global_params.par4[2])], theta[3])
            if prior_par4 < lim_param_grid[3][0]:
                prior_par4 = lim_param_grid[3][0]
            elif prior_par4 > lim_param_grid[3][1]:
                prior_par4 = lim_param_grid[3][1]
            prior.append(prior_par4)
    if global_params.par5 != 'NA':
        prior_law = global_params.par5[0]
        if prior_law != 'constant':
            if prior_law == 'uniform':
                prior_par5 = uniform_prior([float(global_params.par5[1]), float(global_params.par5[2])], theta[4])
            if prior_law == 'gaussian':
                prior_par5 = gaussian_prior([float(global_params.par5[1]), float(global_params.par5[2])], theta[4])
            if prior_par5 < lim_param_grid[4][0]:
                prior_par5 = lim_param_grid[4][0]
            elif prior_par5 > lim_param_grid[4][1]:
                prior_par5 = lim_param_grid[4][1]
            prior.append(prior_par5)

    if global_params.r != 'NA':
        prior_law = global_params.r[0]
        if prior_law != 'constant':
            ind_theta_r = np.where(theta_index == 'r')
            if prior_law == 'uniform':
                prior_r = uniform_prior([float(global_params.r[1]), float(global_params.r[2])], theta[ind_theta_r[0][0]])
            if prior_law == 'gaussian':
                prior_r = gaussian_prior([float(global_params.r[1]), float(global_params.r[2])], theta[ind_theta_r[0][0]])
            prior.append(prior_r)
    if global_params.d != 'NA':
        prior_law = global_params.d[0]
        if prior_law != 'constant':
            ind_theta_d = np.where(theta_index == 'd')
            if prior_law == 'uniform':
                prior_d = uniform_prior([float(global_params.d[1]), float(global_params.d[2])], theta[ind_theta_d[0][0]])
            if prior_law == 'gaussian':
                prior_d = gaussian_prior([float(global_params.d[1]), float(global_params.d[2])], theta[ind_theta_d[0][0]])
            prior.append(prior_d)
    if global_params.rv != 'NA':
        prior_law = global_params.rv[0]
        if prior_law != 'constant':
            ind_theta_rv = np.where(theta_index == 'rv')
            if prior_law == 'uniform':
                prior_rv = uniform_prior([float(global_params.rv[1]), float(global_params.rv[2])], theta[ind_theta_rv[0][0]])
            if prior_law == 'gaussian':
                prior_rv = gaussian_prior([float(global_params.rv[1]), float(global_params.rv[2])], theta[ind_theta_rv[0][0]])
            prior.append(prior_rv)
    if global_params.av != 'NA':
        prior_law = global_params.av[0]
        if prior_law != 'constant':
            ind_theta_av = np.where(theta_index == 'av')
            if prior_law == 'uniform':
                prior_av = uniform_prior([float(global_params.av[1]), float(global_params.av[2])], theta[ind_theta_av[0][0]])
            if prior_law == 'gaussian':
                prior_av = gaussian_prior([float(global_params.av[1]), float(global_params.av[2])], theta[ind_theta_av[0][0]])
            prior.append(prior_av)
    if global_params.vsini != 'NA':
        prior_law = global_params.vsini[0]
        if prior_law != 'constant':
            ind_theta_vsini = np.where(theta_index == 'vsini')
            if prior_law == 'uniform':
                prior_vsini = uniform_prior([float(global_params.vsini[1]), float(global_params.vsini[2])], theta[ind_theta_vsini[0][0]])
            if prior_law == 'gaussian':
                prior_vsini = gaussian_prior([float(global_params.vsini[1]), float(global_params.vsini[2])], theta[ind_theta_vsini[0][0]])
            prior.append(prior_vsini)
    if global_params.ld != 'NA':
        prior_law = global_params.ld[0]
        if prior_law != 'constant':
            ind_theta_ld = np.where(theta_index == 'ld')
            if prior_law == 'uniform':
                prior_ld = uniform_prior([float(global_params.ld[1]), float(global_params.ld[2])], theta[ind_theta_ld[0][0]])
            if prior_law == 'gaussian':
                prior_ld = gaussian_prior([float(global_params.ld[1]), float(global_params.ld[2])], theta[ind_theta_ld[0][0]])
            prior.append(prior_ld)
    ## adding the CPD params, bb_T and bb_R
    if global_params.bb_T != 'NA':
        prior_law = global_params.bb_T[0]
        if prior_law != 'constant':
            ind_theta_bb_T = np.where(theta_index == 'bb_T')
            if prior_law == 'uniform':
                prior_bb_T = uniform_prior([float(global_params.bb_T[1]), float(global_params.bb_T[2])], theta[ind_theta_bb_T[0][0]])
            if prior_law == 'gaussian':
                prior_bb_T = gaussian_prior([float(global_params.bb_T[1]), float(global_params.bb_T[2])], theta[ind_theta_bb_T[0][0]])
            prior.append(prior_bb_T)
    if global_params.bb_R != 'NA':
        prior_law = global_params.bb_R[0]
        if prior_law != 'constant':
            ind_theta_bb_R = np.where(theta_index == 'bb_R')
            if prior_law == 'uniform':
                prior_bb_R = uniform_prior([float(global_params.bb_R[1]), float(global_params.bb_R[2])], theta[ind_theta_bb_R[0][0]])
            if prior_law == 'gaussian':
                prior_bb_R = gaussian_prior([float(global_params.bb_R[1]), float(global_params.bb_R[2])], theta[ind_theta_bb_R[0][0]])
            prior.append(prior_bb_R)
    return prior


def launch_nested_sampling(global_params):
    """
    
    Args:
        
    Returns:
        
    Author: Simon Petrus and Matthieu
    """

    if global_params.observation_format == 'MOSAIC':
        # LogL functions check-up
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
        print('-> Likelihood functions check-up')
        print()

        main_obs_path = global_params.main_observation_path
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):      
            global_params.observation_path = obs
            obs_name = os.path.splitext(os.path.basename(global_params.observation_path))[0]
            print(obs_name + ' will be computed with ' + global_params.logL_type[indobs])
            print()
            y_n_par = yesno('Is this what you want ? (y/n)')
            if y_n_par == 'n':
                print('Please input the desired logL function for ' + obs_name + ':')
                global_params.logL_type[indobs] = input()
            print()
            print()
            print()


    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine='netcdf4')

    # Count the number of free parameters and identify the parameter position in theta
    if global_params.par1 != 'NA':
        theta_index = ['par1']
        lim_param_grid = [[min(ds['par1'].values), max(ds['par1'].values)]]
    else:
        theta_index = []
        lim_param_grid = []
    if global_params.par2 != 'NA':
        theta_index.append('par2')
        lim_param_grid.append([min(ds['par2'].values), max(ds['par2'].values)])
    if global_params.par3 != 'NA':
        theta_index.append('par3')
        lim_param_grid.append([min(ds['par3'].values), max(ds['par3'].values)])
    if global_params.par4 != 'NA':
        theta_index.append('par4')
        lim_param_grid.append([min(ds['par4'].values), max(ds['par4'].values)])
    if global_params.par5 != 'NA':
        theta_index.append('par5')
        lim_param_grid.append([min(ds['par5'].values), max(ds['par5'].values)])
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
    ## adding cpd
    if global_params.bb_T != 'NA' and global_params.bb_T[0] != 'constant':
        n_free_parameters += 1
        theta_index.append('bb_T')
    if global_params.bb_R != 'NA' and global_params.bb_R[0] != 'constant':
        n_free_parameters += 1
        theta_index.append('bb_R')
    theta_index = np.asarray(theta_index)
    #print(theta_index, n_free_parameters)

    # Import all the data (only done once)
    main_file = import_obsmod(global_params)

    if global_params.ns_algo == 'nestle':
        tmpstot1 = time.time()
        loglike_gp = lambda theta: loglike(theta, theta_index, global_params, main_file=main_file)
        prior_transform_gp = lambda theta: prior_transform(theta, theta_index, lim_param_grid, global_params)
        result = nestle.sample(loglike_gp, prior_transform_gp, n_free_parameters, 
                               callback=nestle.print_progress,
                               npoints=int(float(global_params.npoint)), 
                               method=global_params.n_method,
                               maxiter=global_params.n_maxiter,
                               maxcall=global_params.n_maxcall,
                               dlogz=global_params.n_dlogz,
                               decline_factor=global_params.n_decline_factor)
        tmpstot2 = time.time()-tmpstot1
        print('\n')
        print('########     The code spent ' + str(tmpstot2) + ' sec to run   ########')
        print(result.summary())
        print('\n\n')

    if global_params.ns_algo == 'ultranest':
        import ultranest, ultranest.stepsampler

        tmpstot1 = time.time()
        
        loglike_gp = lambda theta: loglike(theta, theta_index, global_params)

        prior_transform_gp = lambda theta: prior_transform(theta, theta_index, lim_param_grid, global_params)

        sampler = ultranest.ReactiveNestedSampler(theta_index,loglike=loglike_gp, transform=prior_transform_gp,
                                                  wrapped_params=[False, False, False, False])#,
                                                #log_dir=global_params.result_path, resume=True)
        #result = sampler.run(min_num_live_points=100, max_ncalls=100000)

        # have to choose the number of steps the slice sampler should take
        # after first results, this should be increased and checked for consistency.
        nsteps = 2 * len(theta_index)
        # create step sampler:
        sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=nsteps,
                                                                 generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
                                                                 # adaptive_nsteps=False,
                                                                 # max_nsteps=40
                                                                 )
        sampler.print_results()
        #sampler.plot_corner()

        tmpstot2 = time.time()-tmpstot1
        print('\n')
        print('########     Ultranest   ########')
        print('########     The code spent ' + str(tmpstot2) + ' sec to run   ########')
        print(result.summary())
        print('\n\n')

        # initialize our nested sampler
        #sampler = NestedSampler(loglike, ptform, ndim)
    
    if global_params.ns_algo == 'dynesty':
        from dynesty import NestedSampler

        # initialize our nested sampler
        #sampler = NestedSampler(loglike, ptform, ndim)

    with open(global_params.result_path + '/result_' + global_params.ns_algo + '.pic', 'wb') as f1:
        pickle.dump(result, f1)
    
    print(' ')
    print('-> Voilà, on est prêt')

    return 

# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    from ForMoSA.main_utilities import GlobFile

    # USER configuration path
    print()
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    print('-> Configuration of environment')
    print('Where is your configuration file?')
    config_file_path = input()
    print()

    # CONFIG_FILE reading and defining global parameters
    global_params = GlobFile(config_file_path)  # To access any param.: global_params.parameter_name
    print()
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    print('-> Nested sampling')
    print()
    launch_nested_sampling(global_params)
