import numpy as np
import os
import glob
import nestle
import time
import xarray as xr
import pickle

from nested_sampling.nested_modif_spec import modif_spec
from nested_sampling.nested_prior_function import uniform_prior, gaussian_prior
from nested_sampling.nested_logL_functions import *
from main_utilities import diag_mat


def import_obsmod(global_params):
    """
    Function to import spectra (model and data) before the inversion

    Args:
        global_params  (object): Class containing every input from the .ini file.
        
    Returns:
        - main_file (list(array)): Return a list of lists with the wavelengths, flux, errors, covariance matrix,
                                transmission, star flux, systematics and the grids for both spectroscopic and photometric data. 

    Authors: Simon Petrus, Matthieu Ravet and Allan Denis
    """
    main_obs_path = global_params.main_observation_path

    main_file = []

    for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
        
        global_params.observation_path = obs
        obs_name = os.path.splitext(os.path.basename(global_params.observation_path))[0]
        spectrum_obs = np.load(os.path.join(global_params.result_path, f'spectrum_obs_{obs_name}.npz'), allow_pickle=True)

        wav_obs_spectro = np.asarray(spectrum_obs['obs_spectro_merge'][0], dtype=float)
        flx_obs_spectro = np.asarray(spectrum_obs['obs_spectro_merge'][1], dtype=float)
        err_obs_spectro = np.asarray(spectrum_obs['obs_spectro_merge'][2], dtype=float)
        # Optional arrays
        inv_cov_obs = np.asarray(spectrum_obs['obs_opt_merge'][0], dtype=float)
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
        path_grid_spectro = os.path.join(global_params.adapt_store_path, f'adapted_grid_spectro_{global_params.grid_name}_{obs_name}_nonan.nc')
        path_grid_photo = os.path.join(global_params.adapt_store_path, f'adapted_grid_photo_{global_params.grid_name}_{obs_name}_nonan.nc')
        ds = xr.open_dataset(path_grid_spectro, decode_cf=False, engine='netcdf4')
        grid_spectro = ds['grid']
        ds.close()
        ds = xr.open_dataset(path_grid_photo, decode_cf=False, engine='netcdf4')
        grid_photo = ds['grid']
        ds.close()
        

        main_file.append([[wav_obs_spectro, wav_obs_photo], [flx_obs_spectro, flx_obs_photo], [err_obs_spectro, err_obs_photo], inv_cov_obs, transm_obs, star_flx_obs, system_obs, grid_spectro, grid_photo])

    return main_file


def loglike(theta, theta_index, global_params, main_file, for_plot='no'):
    """
    Function that calculates the logarithm of the likelihood. 
    The evaluation depends on the choice of likelihood.
    (If this function is used on the plotting module, it returns the outputs of the modif_spec function)
    
    Args:
        theta           (list): Parameter values randomly picked by the nested sampling
        theta_index     (list): Index for the parameter values randomly picked
        global_params (object): Class containing every input from the .ini file.
        main_file (list(list)): List containing the wavelengths, flux, errors, covariance, and grid information
        for_plot         (str): Default is 'no'. When this function is called from the plotting functions module, we use 'yes'
        
    Returns:
        - FINAL_logL     (float): Final evaluated loglikelihood for both spectra and photometry. 

    Authors: Simon Petrus, Matthieu Ravet and Allan Denis
    """

    # Recovery of each observation spectroscopy and photometry data
    main_obs_path = global_params.main_observation_path
    FINAL_logL = 0


    for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
        
        # Recovery of spectroscopy and photometry data
        wav_obs_spectro = main_file[indobs][0][0]
        wav_obs_photo = main_file[indobs][0][1]
        flx_obs_spectro = main_file[indobs][1][0]
        flx_obs_photo = main_file[indobs][1][1]
        err_obs_spectro = main_file[indobs][2][0]
        err_obs_photo = main_file[indobs][2][1]
        inv_cov_obs = main_file[indobs][3]
        transm_obs = main_file[indobs][4]
        star_flx_obs = main_file[indobs][5]
        system_obs = main_file[indobs][6]
        
        
        # Recovery of the spectroscopy and photometry model
        grid_spectro = main_file[indobs][7]
        grid_photo = main_file[indobs][8]

        # Calculation of the likelihood for each sub-spectrum defined by the parameter 'wav_fit'
        for ns_u_ind, ns_u in enumerate(global_params.wav_fit[indobs].split('/')):
            
            min_ns_u = float(ns_u.split(',')[0])
            max_ns_u = float(ns_u.split(',')[1])
            ind_grid_spectro_sel = np.where((grid_spectro['wavelength'] >= min_ns_u) & (grid_spectro['wavelength'] <= max_ns_u))
            ind_grid_photo_sel = np.where((grid_photo['wavelength'] >= min_ns_u) & (grid_photo['wavelength'] <= max_ns_u))

            # Cutting of the grid on the wavelength grid defined by the parameter 'wav_fit'
            grid_spectro_cut = grid_spectro.sel(wavelength=grid_spectro['wavelength'][ind_grid_spectro_sel])
            grid_photo_cut = grid_photo.sel(wavelength=grid_photo['wavelength'][ind_grid_photo_sel])

            # Interpolation of the grid at the theta parameters set
            if global_params.par3 == 'NA':
                if len(grid_spectro_cut['wavelength']) != 0:
                    flx_mod_spectro_cut = np.asarray(grid_spectro_cut.interp(par1=theta[0], par2=theta[1],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_spectro_cut = np.asarray([])
                if len(grid_photo_cut['wavelength']) != 0:
                    flx_mod_photo_cut = np.asarray(grid_photo_cut.interp(par1=theta[0], par2=theta[1],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_photo_cut = np.asarray([])
            elif global_params.par4 == 'NA':
                if len(grid_spectro_cut['wavelength']) != 0:
                    flx_mod_spectro_cut = np.asarray(grid_spectro_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_spectro_cut = np.asarray([])
                if len(grid_photo_cut['wavelength']) != 0:
                    flx_mod_photo_cut = np.asarray(grid_photo_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_photo_cut = np.asarray([])
            elif global_params.par5 == 'NA':
                if len(grid_spectro_cut['wavelength']) != 0:
                    flx_mod_spectro_cut = np.asarray(grid_spectro_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_spectro_cut = np.asarray([])
                if len(grid_photo_cut['wavelength']) != 0:
                    flx_mod_photo_cut = np.asarray(grid_photo_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_photo_cut = np.asarray([])
            else:
                if len(grid_spectro_cut['wavelength']) != 0:
                    flx_mod_spectro_cut = np.asarray(grid_spectro_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            par5=theta[4],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_spectro_cut = np.asarray([])
                if len(grid_photo_cut['wavelength']) != 0:
                    flx_mod_photo_cut = np.asarray(grid_photo_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            par5=theta[4],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_photo_cut = np.asarray([])


            # Re-merging of the data and interpolated synthetic spectrum to a wavelength grid defined by the parameter 'wav_fit'
            ind_spectro = np.where((wav_obs_spectro >= min_ns_u) & (wav_obs_spectro <= max_ns_u))
            ind_photo = np.where((wav_obs_photo >= min_ns_u) & (wav_obs_photo <= max_ns_u))
            if ns_u_ind == 0:
                wav_obs_spectro_ns_u = wav_obs_spectro[ind_spectro]
                flx_obs_spectro_ns_u = flx_obs_spectro[ind_spectro]
                err_obs_spectro_ns_u = err_obs_spectro[ind_spectro]
                flx_mod_spectro_ns_u = flx_mod_spectro_cut
                if len(inv_cov_obs) != 0:  # Add covariance in the loop (if necessary)
                    inv_cov_obs_ns_u = inv_cov_obs[np.ix_(ind_spectro[0],ind_spectro[0])]
                else:
                    inv_cov_obs_ns_u = np.asarray([])
                if len(transm_obs) != 0: # Add the transmission (if necessary)
                    transm_obs_ns_u = transm_obs[ind_spectro]
                else:
                    transm_obs_ns_u = np.asarray([])
                if len(star_flx_obs) != 0: # Add star flux (if necessary)
                    star_flx_obs_ns_u = star_flx_obs[0,ind_spectro]
                else:
                    star_flx_obs_ns_u = np.asarray([])
                if len(system_obs) != 0: # Add systematics model (if necessary)
                    system_obs_ns_u = system_obs[0,ind_spectro]
                else:
                    system_obs_ns_u = np.asarray([])
                wav_obs_photo_ns_u = wav_obs_photo[ind_photo]
                flx_obs_photo_ns_u = flx_obs_photo[ind_photo]
                err_obs_photo_ns_u = err_obs_photo[ind_photo]
                flx_mod_photo_ns_u = flx_mod_photo_cut
            else:
                wav_obs_spectro_ns_u = np.concatenate((wav_obs_spectro_ns_u, wav_obs_spectro[ind_spectro]))
                flx_obs_spectro_ns_u = np.concatenate((flx_obs_spectro_ns_u, flx_obs_spectro[ind_spectro]))
                err_obs_spectro_ns_u = np.concatenate((err_obs_spectro_ns_u, err_obs_spectro[ind_spectro]))
                flx_mod_spectro_ns_u = np.concatenate((flx_mod_spectro_ns_u, flx_mod_spectro_cut))
                if len(inv_cov_obs_ns_u) != 0: # Merge the covariance matrices (if necessary)
                    inv_cov_obs_ns_u = diag_mat([inv_cov_obs_ns_u, inv_cov_obs[np.ix_(ind_spectro[0],ind_spectro[0])]])
                if len(transm_obs_ns_u) != 0: # Merge the transmissions (if necessary)
                    transm_obs_ns_u = np.concatenate((transm_obs_ns_u, transm_obs[ind_spectro]))
                if len(star_flx_obs_ns_u) != 0: # Merge star fluxes (if necessary)
                    star_flx_obs_ns_u = np.concatenate((star_flx_obs_ns_u, star_flx_obs[0,ind_grid_spectro_sel]),axis=0)
                if len(system_obs) != 0: # Merge systematics model (if necessary)
                    system_obs_ns_u = np.concatenate((system_obs_ns_u, system_obs[0,ind_grid_spectro_sel]), axis=0)
                wav_obs_photo_ns_u = np.concatenate((wav_obs_photo_ns_u, wav_obs_photo[ind_photo]))
                flx_obs_photo_ns_u = np.concatenate((flx_obs_photo_ns_u, flx_obs_photo[ind_photo]))
                err_obs_photo_ns_u = np.concatenate((err_obs_photo_ns_u, err_obs_photo[ind_photo]))
                flx_mod_photo_ns_u = np.concatenate((flx_mod_photo_ns_u, flx_mod_photo_cut))

        # Modification of the synthetic spectrum with the extra-grid parameters
        modif_spec_LL = modif_spec(global_params, theta, theta_index,
                                    wav_obs_spectro_ns_u,  flx_obs_spectro_ns_u,  err_obs_spectro_ns_u,  flx_mod_spectro_ns_u,
                                    wav_obs_photo_ns_u,  flx_obs_photo_ns_u, err_obs_photo_ns_u,  flx_mod_photo_ns_u, 
                                    transm_obs_ns_u, star_flx_obs_ns_u, system_obs_ns_u, indobs=indobs)
        
        flx_obs_spectro_modif, flx_obs_photo_modif = modif_spec_LL[1], modif_spec_LL[5]
        flx_mod_spectro_modif, flx_mod_photo_modif = modif_spec_LL[3], modif_spec_LL[7]
        err_obs_spectro_modif, err_obs_photo_modif = modif_spec_LL[2], modif_spec_LL[6]
        inv_cov_obs_modif = inv_cov_obs_ns_u
        ck = modif_spec_LL[8]
        planet_contribution, stellar_contribution, star_flx_obs, systematics = modif_spec_LL[9], modif_spec_LL[10], modif_spec_LL[11], modif_spec_LL[12]
        
        if global_params.use_lsqr[indobs] == 'True':
            # If our data is contaminated by starlight difraction, the model is the sum of the estimated stellar contribution + planet model
            flx_mod_spectro_modif = planet_contribution * flx_mod_spectro_modif + np.dot(stellar_contribution, star_flx_obs[0].T)
            if len(systematics) > 0:
                flx_mod_spectro_modif += systematics
                

        # Computation of the photometry logL
        if len(flx_obs_photo_modif) != 0:
            logL_photo = logL_chi2_classic(flx_obs_photo_modif-flx_mod_photo_modif, err_obs_photo_modif)
        else:
            logL_photo = 0

        # Computation of the spectroscopy logL
        if len(flx_obs_spectro_modif) != 0:
            if global_params.logL_type[indobs] == 'chi2_classic':
                logL_spectro = logL_chi2_classic(flx_obs_spectro_modif-flx_mod_spectro_modif, err_obs_spectro_modif)
            elif global_params.logL_type[indobs] == 'chi2_covariance' and len(inv_cov_obs_modif) != 0:
                logL_spectro = logL_chi2_covariance(flx_obs_spectro_modif-flx_mod_spectro_modif, inv_cov_obs_modif)
            elif global_params.logL_type[indobs] == 'CCF_Brogi':
                logL_spectro = logL_CCF_Brogi(flx_obs_spectro_modif, flx_mod_spectro_modif)
            elif global_params.logL_type[indobs] == 'CCF_Zucker':
                logL_spectro = logL_CCF_Zucker(flx_obs_spectro_modif, flx_mod_spectro_modif)
            elif global_params.logL_type[indobs] == 'CCF_custom':
                logL_spectro = logL_CCF_custom(flx_obs_spectro_modif, flx_mod_spectro_modif, err_obs_spectro_modif)
            elif global_params.logL_type[indobs] == 'chi2_extended':
                logL_spectro = logL_chi2_extended(flx_obs_spectro_modif-flx_mod_spectro_modif, err_obs_spectro_modif)
            elif global_params.logL_type[indobs] == 'chi2_extended_covariance' and len(inv_cov_obs_modif) != 0:
                logL_spectro = logL_chi2_extended_covariance(flx_obs_spectro_modif-flx_mod_spectro_modif, inv_cov_obs_modif)
            else:
                print()
                print('WARNING: One or more dataset are not included when performing the inversion.')
                print('Please adapt your likelihood function choice.')
                print()
                exit()
        else:
            logL_spectro = 0

        # Compute the final logL (sum of all likelihood under the hypothesis of independent instruments)
        FINAL_logL = logL_photo + logL_spectro + FINAL_logL
        
    if for_plot == 'no':
        return FINAL_logL
    else:
        return modif_spec


def prior_transform(theta, theta_index, lim_param_grid, global_params):
    """
    Function that define the priors to be used for the inversion. 
    We check that the boundaries are consistent with the grid extension.
    
    Args:
        theta           (list): Parameter values randomly picked by the nested sampling
        theta_index     (list): Index for the parameter values randomly picked
        lim_param_grid  (list): Boundaries for the parameters explored
        global_params (object): Class containing every input from the .ini file.
        
    Returns:
        - prior           (list): List containing all the prior information
        
    Author: Simon Petrus, Matthieu Ravet, Allan Denis
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

    # Extra-grid parameters
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

    # - - - - - - - - - - - - - - - - - - - - -
            
    # Individual parameters / observation
            
    if len(global_params.alpha) > 3: # If you want separate alpha for each observations
        main_obs_path = global_params.main_observation_path
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            if global_params.alpha[indobs*3] != 'NA':
                prior_law = global_params.alpha[indobs*3] # Prior laws should be separeted by 2 values (need to be upgraded)
                if prior_law != 'constant':
                    ind_theta_alpha = np.where(theta_index == f'alpha_{indobs}')
                    if prior_law == 'uniform':
                        prior_alpha = uniform_prior([float(global_params.alpha[indobs*3+1]), float(global_params.alpha[indobs*3+2])], theta[ind_theta_alpha[0][0]]) # Prior values should be by two
                    if prior_law == 'gaussian':
                        prior_alpha = gaussian_prior([float(global_params.alpha[indobs*3+1]), float(global_params.alpha[indobs*3+2])], theta[ind_theta_alpha[0][0]])
                    prior.append(prior_alpha)
    else: # If you want 1 common alpha for all observations
        if global_params.alpha != 'NA':
            prior_law = global_params.alpha[0]
            if prior_law != 'constant':
                ind_theta_alpha = np.where(theta_index == 'alpha')
                if prior_law == 'uniform':
                    prior_alpha = uniform_prior([float(global_params.alpha[1]), float(global_params.alpha[2])], theta[ind_theta_alpha[0][0]])
                if prior_law == 'gaussian':
                    prior_alpha = gaussian_prior([float(global_params.alpha[1]), float(global_params.alpha[2])], theta[ind_theta_alpha[0][0]])
                prior.append(prior_alpha)
    if len(global_params.rv) > 3: # If you want separate rv for each observations
        main_obs_path = global_params.main_observation_path
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            if global_params.rv[indobs*3] != 'NA':
                prior_law = global_params.rv[indobs*3] # Prior laws should be separeted by 2 values (need to be upgraded)
                if prior_law != 'constant':
                    ind_theta_rv = np.where(theta_index == f'rv_{indobs}')
                    if prior_law == 'uniform':
                        prior_rv = uniform_prior([float(global_params.rv[indobs*3+1]), float(global_params.rv[indobs*3+2])], theta[ind_theta_rv[0][0]]) # Prior values should be by two
                    if prior_law == 'gaussian':
                        prior_rv = gaussian_prior([float(global_params.rv[indobs*3+1]), float(global_params.rv[indobs*3+2])], theta[ind_theta_rv[0][0]])
                    prior.append(prior_rv)
    else: # If you want 1 common rv for all observations
        if global_params.rv != 'NA':
            prior_law = global_params.rv[0]
            if prior_law != 'constant':
                ind_theta_rv = np.where(theta_index == 'rv')
                if prior_law == 'uniform':
                    prior_rv = uniform_prior([float(global_params.rv[1]), float(global_params.rv[2])], theta[ind_theta_rv[0][0]])
                if prior_law == 'gaussian':
                    prior_rv = gaussian_prior([float(global_params.rv[1]), float(global_params.rv[2])], theta[ind_theta_rv[0][0]])
                prior.append(prior_rv)
    if len(global_params.vsini) > 4: # If you want separate vsini for each observations
        main_obs_path = global_params.main_observation_path
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            if global_params.vsini[indobs*4] != 'NA':
                prior_law = global_params.vsini[indobs*4] # Prior laws should be separeted by 2 values (need to be upgraded)
                if prior_law != 'constant':
                    ind_theta_vsini = np.where(theta_index == f'vsini_{indobs}')
                    if prior_law == 'uniform':
                        prior_vsini = uniform_prior([float(global_params.vsini[indobs*4+1]), float(global_params.vsini[indobs*4+2])], theta[ind_theta_vsini[0][0]]) # Prior values should be by two
                    if prior_law == 'gaussian':
                        prior_vsini = gaussian_prior([float(global_params.vsini[indobs*4+1]), float(global_params.vsini[indobs*4+2])], theta[ind_theta_vsini[0][0]])
                    prior.append(prior_vsini)
    else: # If you want 1 common vsini for all observations
        if global_params.vsini != 'NA':
            prior_law = global_params.vsini[0]
            if prior_law != 'constant':
                ind_theta_vsini = np.where(theta_index == 'vsini')
                if prior_law == 'uniform':
                    prior_vsini = uniform_prior([float(global_params.vsini[1]), float(global_params.vsini[2])], theta[ind_theta_vsini[0][0]])
                if prior_law == 'gaussian':
                    prior_vsini = gaussian_prior([float(global_params.vsini[1]), float(global_params.vsini[2])], theta[ind_theta_vsini[0][0]])
                prior.append(prior_vsini)
    if len(global_params.ld) > 3: # If you want separate ld for each observations
        main_obs_path = global_params.main_observation_path
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            if global_params.ld[indobs*3] != 'NA':
                prior_law = global_params.ld[indobs*3] # Prior laws should be separeted by 2 values (need to be upgraded)
                if prior_law != 'constant':
                    ind_theta_ld = np.where(theta_index == f'ld_{indobs}')
                    if prior_law == 'uniform':
                        prior_ld = uniform_prior([float(global_params.ld[indobs*3+1]), float(global_params.ld[indobs*3+2])], theta[ind_theta_ld[0][0]]) # Prior values should be by two
                    if prior_law == 'gaussian':
                        prior_ld = gaussian_prior([float(global_params.ld[indobs*3+1]), float(global_params.ld[indobs*3+2])], theta[ind_theta_ld[0][0]])
                    prior.append(prior_ld)
    else: # If you want 1 common ld for all observations
        if global_params.ld != 'NA':
            prior_law = global_params.ld[0]
            if prior_law != 'constant':
                ind_theta_ld = np.where(theta_index == 'ld')
                if prior_law == 'uniform':
                    prior_ld = uniform_prior([float(global_params.ld[1]), float(global_params.ld[2])], theta[ind_theta_ld[0][0]])
                if prior_law == 'gaussian':
                    prior_ld = gaussian_prior([float(global_params.ld[1]), float(global_params.ld[2])], theta[ind_theta_ld[0][0]])
                prior.append(prior_ld)

    # - - - - - - - - - - - - - - - - - - - - -

    if global_params.av != 'NA':
        prior_law = global_params.av[0]
        if prior_law != 'constant':
            ind_theta_av = np.where(theta_index == 'av')
            if prior_law == 'uniform':
                prior_av = uniform_prior([float(global_params.av[1]), float(global_params.av[2])], theta[ind_theta_av[0][0]])
            if prior_law == 'gaussian':
                prior_av = gaussian_prior([float(global_params.av[1]), float(global_params.av[2])], theta[ind_theta_av[0][0]])
            prior.append(prior_av)
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
    Function to launch the nested sampling. 
    We first perform LogL function check-ups. 
    Then the free parameters are counted and the data imported. 
    Finally, depending on the nested sampling methode chosen in the config file, we perform the inversion.
    (Methods succesfully implemented are Nestle and PyMultinest)

    Args:
        global_params (object): Class containing every input from the .ini file.

    Returns:
        None
        
    Author: Simon Petrus and Matthieu Ravet
    """

    # LogL functions check-ups
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    print('-> Likelihood functions check-ups')
    print()

    main_obs_path = global_params.main_observation_path
    for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):      
        global_params.observation_path = obs
        obs_name = os.path.splitext(os.path.basename(global_params.observation_path))[0]

        # Check the choice of likelihood (only for MOSAIC)
        print(obs_name + ' will be computed with ' + global_params.logL_type[indobs])

        if global_params.logL_type[indobs] == 'CCF_Brogi' and global_params.continuum_sub[indobs] == 'NA':
            print('WARNING. You cannot use CCF mappings without substracting the continuum')
            print()
            exit()
        elif global_params.logL_type[indobs] == 'CCF_Zucker' and global_params.continuum_sub[indobs] == 'NA':
            print('WARNING. You cannot use CCF mappings without substracting the continuum')
            print()
            exit()
        elif global_params.logL_type[indobs] == 'CCF_custom' and global_params.continuum_sub[indobs] == 'NA':
            print('WARNING. You cannot use CCF mappings without substracting the continuum')
            print()
            exit()
            
        print()
    print('Done !')
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

    # - - - - - - - - - - - - - - - - - - - - -
            
    # Individual parameters / observation
        
    if len(global_params.alpha) > 3: 
        main_obs_path = global_params.main_observation_path
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            if global_params.alpha[indobs*3] != 'NA' and global_params.alpha[indobs*3] != 'constant': # Check if the idobs is different from constant
                n_free_parameters += 1
                theta_index.append(f'alpha_{indobs}')
    else:
        if global_params.alpha != 'NA' and global_params.alpha[0] != 'constant':
            n_free_parameters += 1
            theta_index.append(f'alpha')
    if len(global_params.rv) > 3: 
        main_obs_path = global_params.main_observation_path
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            if global_params.rv[indobs*3] != 'NA' and global_params.rv[indobs*3] != 'constant': # Check if the idobs is different from constant
                n_free_parameters += 1
                theta_index.append(f'rv_{indobs}')
    else:
        if global_params.rv != 'NA' and global_params.rv[0] != 'constant':
            n_free_parameters += 1
            theta_index.append(f'rv')
    if len(global_params.vsini) > 4: 
        main_obs_path = global_params.main_observation_path
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            if global_params.vsini[indobs*4] != 'NA' and global_params.vsini[indobs*4] != 'constant': # Check if the idobs is different from constant
                n_free_parameters += 1
                theta_index.append(f'vsini_{indobs}')
    else:
        if global_params.vsini != 'NA' and global_params.vsini[0] != 'constant':
            n_free_parameters += 1
            theta_index.append('vsini')
    if len(global_params.ld) > 3: 
        main_obs_path = global_params.main_observation_path
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            if global_params.ld[indobs*3] != 'NA' and global_params.ld[indobs*3] != 'constant': # Check if the idobs is different from constant
                n_free_parameters += 1
                theta_index.append(f'ld_{indobs}')
    else:
        if global_params.ld != 'NA' and global_params.ld[0] != 'constant':
            n_free_parameters += 1
            theta_index.append('ld')

    # - - - - - - - - - - - - - - - - - - - - -
            
    if global_params.av != 'NA' and global_params.av[0] != 'constant':
        n_free_parameters += 1
        theta_index.append('av')
    ## adding cpd
    if global_params.bb_T != 'NA' and global_params.bb_T[0] != 'constant':
        n_free_parameters += 1
        theta_index.append('bb_T')
    if global_params.bb_R != 'NA' and global_params.bb_R[0] != 'constant':
        n_free_parameters += 1
        theta_index.append('bb_R')
    theta_index = np.asarray(theta_index)

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
        # Reformat the result file
        with open(global_params.result_path + '/result_' + global_params.ns_algo + '_RAW.pic', 'wb') as f1:
            pickle.dump(result, f1)
        logz = [result['logz'], result['logzerr']]
        samples = result['samples']
        weights = result['weights']
        logvol = result['logvol']
        logl = result['logl']
        tmpstot2 = time.time()-tmpstot1
        print(' ')
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
        print('-> Nestle  ')
        print(' ')
        print('The code spent ' + str(tmpstot2) + ' sec to run.')
        print(result.summary())
        print('\n')

    if global_params.ns_algo == 'pymultinest':
        import pymultinest
        tmpstot1 = time.time()
        loglike_gp = lambda theta: loglike(theta, theta_index, global_params, main_file=main_file)
        prior_transform_gp = lambda theta: prior_transform(theta, theta_index, lim_param_grid, global_params)
        result = pymultinest.solve(
                        LogLikelihood=loglike_gp,
                        Prior=prior_transform_gp,
                        n_dims=n_free_parameters,
                        n_live_points=int(float(global_params.npoint)),
                        outputfiles_basename=global_params.result_path + '/result_' + global_params.ns_algo + '_RAW_',
                        verbose=True,
                        resume=False
                        )
        # Reformat the result file
        with open(global_params.result_path + '/result_' + global_params.ns_algo + '_RAW_stats.dat',
                  'rb') as open_dat:
            for l, line in enumerate(open_dat):
                if l == 0:
                    line = line.strip().split()
                    logz_multi = float(line[5])
                    logzerr_multi = float(line[7])
        sample_multi = []
        logl_multi = []
        logvol_multi = []
        with open(global_params.result_path + '/result_' + global_params.ns_algo + '_RAW_ev.dat',
                  'rb') as open_dat:
            for l, line in enumerate(open_dat):
                line = line.strip().split()
                points = []
                for p in line[:-3]:
                    points.append(float(p))
                sample_multi.append(points)
                logl_multi.append(float(line[-3]))
                logvol_multi.append(float(line[-2]))
        sample_multi = np.asarray(sample_multi)
        logl_multi = np.asarray(logl_multi)
        logvol_multi = np.asarray(logvol_multi)
        iter_multi = []
        weights_multi = []
        final_logl_multi = []
        final_logvol_multi = []
        with open(global_params.result_path + '/result_' + global_params.ns_algo + '_RAW_.txt',
                  'rb') as open_dat:
            for l, line in enumerate(open_dat):
                line = line.strip().split()
                points = []
                for p in line[2:]:
                    points.append(float(p))
                if points in sample_multi:
                    ind = np.where(sample_multi == points)
                    iter_multi.append(points)
                    weights_multi.append(float(line[0]))
                    final_logl_multi.append(logl_multi[ind[0][0]])
                    final_logvol_multi.append(logvol_multi[ind[0][0]])
        logz = [logz_multi, logzerr_multi]
        samples = np.asarray(iter_multi)
        weights = np.asarray(weights_multi)
        logvol = np.asarray(final_logvol_multi)
        logl = np.asarray(final_logl_multi)
        tmpstot2 = time.time()-tmpstot1
        print(' ')
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
        print('-> PyMultinest  ')
        print(' ')
        print('The code spent ' + str(tmpstot2) + ' sec to run.')
        print('The evidence is: %(logZ).1f +- %(logZerr).1f' % result)
        print('The parameter values are:')
        for name, col in zip(theta_index, result['samples'].transpose()):
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
        print('\n')

    # if global_params.ns_algo == 'ultranest':
    #     import ultranest, ultranest.stepsampler

    #     tmpstot1 = time.time()
        
    #     loglike_gp = lambda theta: loglike(theta, theta_index, global_params)

    #     prior_transform_gp = lambda theta: prior_transform(theta, theta_index, lim_param_grid, global_params)

    #     sampler = ultranest.ReactiveNestedSampler(theta_index,loglike=loglike_gp, transform=prior_transform_gp,
    #                                               wrapped_params=[False, False, False, False])#,
    #                                             #log_dir=global_params.result_path, resume=True)
    #     #result = sampler.run(min_num_live_points=100, max_ncalls=100000)

    #     # have to choose the number of steps the slice sampler should take
    #     # after first results, this should be increased and checked for consistency.
    #     nsteps = 2 * len(theta_index)
    #     # create step sampler:
    #     sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=nsteps,
    #                                                              generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
    #                                                              # adaptive_nsteps=False,
    #                                                              # max_nsteps=40
    #                                                              )
    #     sampler.print_results()
    #     #sampler.plot_corner()

    #     tmpstot2 = time.time()-tmpstot1
    #     print(' ')
    #     print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    #     print('-> Ultranest  ')
    #     print(' ')
    #     print('The code spent ' + str(tmpstot2) + ' sec to run.')
    #     print(result.summary())
    #     print('\n')
    
    # if global_params.ns_algo == 'dynesty':
    #     from dynesty import NestedSampler

    #     initialize our nested sampler
    #     sampler = NestedSampler(loglike, ptform, ndim)

    result_reformat = {"samples": samples,
                       "weights": weights,
                       "logl": logl,
                       "logvol": logvol,
                       "logz": logz,}

    with open(global_params.result_path + '/result_' + global_params.ns_algo + '.pic', "wb") as tf:
        pickle.dump(result_reformat, tf)

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
