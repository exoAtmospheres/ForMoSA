import numpy as np
import os
import glob
import nestle
import time
import xarray as xr
import pickle
from scipy.interpolate import interp1d

from ForMoSA.nested_sampling.nested_modif_spec import modif_spec
from ForMoSA.nested_sampling.nested_prior_functions import uniform_prior, loguniform_prior, gaussian_prior
from ForMoSA.nested_sampling.nested_logL_functions import *



def import_obsmod(global_params):
    """
    Function to import spectra (model and data) before the inversion

    Args:
        global_params    (object): Class containing every input from the .ini file.

    Returns:
        - main_file (list(array)): Return a list of lists with the wavelengths, flux, errors, covariance matrix,
                                transmission, star flux, systematics, grid indices and the grids for both spectroscopic and photometric data.

    Authors: Simon Petrus, Matthieu Ravet and Allan Denis
    """
    main_obs_path = global_params.main_observation_path

    main_file = []

    for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):

        # Recovery of the observational dictionnary
        global_params.observation_path = obs
        obs_name = os.path.splitext(os.path.basename(global_params.observation_path))[0]
        obs_dict = dict(np.load(os.path.join(global_params.result_path, f'spectrum_obs_{obs_name}.npz'), allow_pickle=True))

        # Recovery of the spectroscopy and photometry model
        path_grid_spectro = os.path.join(global_params.adapt_store_path, f'adapted_grid_spectro_{global_params.grid_name}_{obs_name}_nonan.nc')
        ds_spectro = xr.open_dataset(path_grid_spectro, decode_cf=False, engine='netcdf4')
        grid_spectro = ds_spectro['grid']
        path_grid_photo = os.path.join(global_params.adapt_store_path, f'adapted_grid_photo_{global_params.grid_name}_{obs_name}_nonan.nc')
        ds_photo = xr.open_dataset(path_grid_photo, decode_cf=False, engine='netcdf4')
        grid_photo = ds_photo['grid']

        # Emulator (if necessary)
        if global_params.emulator[0] != 'NA':
            # PCA or NMF
            mod_dict = dict(np.load(os.path.join(global_params.result_path, f'{global_params.emulator[0]}_mod_{obs_name}.npz'), allow_pickle=True))
        else:
            # Standard method
            mod_dict = {'wav_spectro': np.asarray(ds_spectro.coords['wavelength']), 'res_spectro': np.asarray(ds_spectro.attrs['res'])}
        ds_spectro.close()
        ds_photo.close()

        # Initiate indices tables for each sub-spectrum
        mask_mod_spectro = np.zeros(len(mod_dict['wav_spectro']), dtype=bool)
        mask_obs_spectro = np.zeros(len(obs_dict['wav_spectro']), dtype=bool)
        mask_photo = np.zeros(len(obs_dict['wav_photo']), dtype=bool)
        for ns_u_ind, ns_u in enumerate(global_params.wav_fit[indobs % len(global_params.wav_fit)].split('/')):
            min_ns_u = float(ns_u.split(',')[0])
            max_ns_u = float(ns_u.split(',')[1])
            # Indices of each model and data. Masks to have larger cuts of the spectroscopic grid if needed (if rv is defined)
            if global_params.rv[indobs*3 % len(global_params.rv)] == 'NA':
                mask_mod_spectro += (mod_dict['wav_spectro'] >= min_ns_u) & (mod_dict['wav_spectro'] <= max_ns_u)
            else:
                mask_mod_spectro += (mod_dict['wav_spectro'] >= 0.99 * min_ns_u) & (mod_dict['wav_spectro'] <= 1.01 * max_ns_u)
            mask_obs_spectro += (obs_dict['wav_spectro'] >= min_ns_u) & (obs_dict['wav_spectro'] <= max_ns_u)
            mask_photo += (obs_dict['wav_photo'] >= min_ns_u) & (obs_dict['wav_photo'] <= max_ns_u)

        # Cutting the data to a wavelength grid defined by the parameter 'wav_fit'
        for key in obs_dict:
            if obs_dict[key].size != 0:
                if key[-5:] == 'photo':
                    obs_dict[key] = obs_dict[key][mask_photo]
                elif key != 'inv_cov':
                    obs_dict[key] = obs_dict[key][mask_obs_spectro]
                else:
                    obs_dict[key] = obs_dict[key][np.ix_(mask_obs_spectro, mask_obs_spectro)]
            else:
                pass

        # Cutting the model to a wavelength grid defined by the parameter 'wav_fit'
        for key in mod_dict:
            if mod_dict[key].size != 0:
                if key[-5:] == 'photo':
                    if key[:7] == 'vectors':
                        mod_dict[key] = mod_dict[key][:,mask_photo]
                    else:
                        mod_dict[key] = mod_dict[key][mask_photo]
                else:
                    if key[:7] == 'vectors':
                        mod_dict[key] = mod_dict[key][:,mask_mod_spectro]
                    else:
                        mod_dict[key] = mod_dict[key][mask_mod_spectro]
            else:
                pass

        # Interpolating model resolution onto the data + computing determinants (if necessary)
        if len(obs_dict['wav_spectro']) != 0:
            interp_mod_to_obs = interp1d(mod_dict['wav_spectro'], mod_dict['res_spectro'], fill_value='extrapolate') 
            mod_dict['res_spectro'] = interp_mod_to_obs(obs_dict['wav_spectro'])
            if len(obs_dict['cov']) != 0:
                obs_dict['log_det_spectro'] = np.log(np.linalg.det(obs_dict['cov']))
            else:
                obs_dict['log_det_spectro'] = np.log(np.dot(obs_dict['err_spectro'],obs_dict['err_spectro']))
        if len(obs_dict['wav_photo']) != 0:
            obs_dict['log_det_photo'] = np.log(np.dot(obs_dict['err_photo'],obs_dict['err_photo']))

        # Cutting of the grid on the wavelength grid defined by the parameter 'wav_fit'
        if global_params.emulator[0] == 'NA':
            grid_spectro = grid_spectro.sel(wavelength=grid_spectro['wavelength'][mask_mod_spectro])
            grid_photo = grid_photo.sel(wavelength=grid_photo['wavelength'][mask_photo])

        main_file.append([obs_dict,
                          mod_dict,
                          grid_spectro,
                          grid_photo])

    return main_file


def loglike(theta, theta_index, global_params, main_file, for_plot='no'):
    """
    Function that calculates the logarithm of the likelihood.
    The evaluation depends on the choice of likelihood.
    (If this function is used on the plotting module, it returns the outputs of the modif_spec function)

    Args:
        theta                 (list): Parameter values randomly picked by the nested sampling
        theta_index           (list): Index for the parameter values randomly picked
        global_params       (object): Class containing every input from the .ini file.
        main_file       (list(list)): List containing the wavelengths, flux, errors, covariance, and grid information
        for_plot               (str): Default is 'no'. When this function is called from the plotting functions module, we use 'yes'

    Returns:
        - FINAL_logL     (float): Final evaluated loglikelihood for both spectra and photometry.

    Authors: Simon Petrus, Matthieu Ravet and Allan Denis
    """

    # Recovery of each observation spectroscopy and photometry data
    main_obs_path = global_params.main_observation_path
    FINAL_logL = 0


    for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):

        # Recovery of spectroscopy and photometry data and model (fixed)
        obs_dict = main_file[indobs][0]
        mod_dict = main_file[indobs][1]

        # Recovery of the spectroscopy and photometry grids
        grid_spectro = main_file[indobs][2]
        grid_photo = main_file[indobs][3]

        # Interpolation of the grid at the theta parameters set
        if global_params.par3[0] == 'NA':
            if len(obs_dict['wav_spectro']) != 0:
                interp_spectro = np.asarray(grid_spectro.interp(par1=theta[0], par2=theta[1],
                                                        method=global_params.method, kwargs={"fill_value": "extrapolate"}))
            else:
                interp_spectro = np.asarray([])
            if len(obs_dict['wav_photo']) != 0:
                interp_photo = np.asarray(grid_photo.interp(par1=theta[0], par2=theta[1],
                                                        method=global_params.method, kwargs={"fill_value": "extrapolate"}))
            else:
                interp_photo = np.asarray([])
        elif global_params.par4[0] == 'NA':
            if len(obs_dict['wav_spectro']) != 0:
                interp_spectro = np.asarray(grid_spectro.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                        method=global_params.method, kwargs={"fill_value": "extrapolate"}))
            else:
                interp_spectro = np.asarray([])
            if len(obs_dict['wav_photo']) != 0:
                interp_photo = np.asarray(grid_photo.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                        method=global_params.method, kwargs={"fill_value": "extrapolate"}))
            else:
                interp_photo = np.asarray([])
        elif global_params.par5[0] == 'NA':
            if len(obs_dict['wav_spectro']) != 0:
                interp_spectro = np.asarray(grid_spectro.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                        method=global_params.method, kwargs={"fill_value": "extrapolate"}))
            else:
                interp_spectro = np.asarray([])
            if len(obs_dict['wav_photo']) != 0:
                interp_photo = np.asarray(grid_photo.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                        method=global_params.method, kwargs={"fill_value": "extrapolate"}))
            else:
                interp_photo = np.asarray([])
        else:
            if len(obs_dict['wav_spectro']) != 0:
                interp_spectro = np.asarray(grid_spectro.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                        par5=theta[4],
                                                        method=global_params.method, kwargs={"fill_value": "extrapolate"}))
            else:
                interp_spectro = np.asarray([])
            if len(obs_dict['wav_photo']) != 0:
                interp_photo = np.asarray(grid_photo.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                        par5=theta[4],
                                                        method=global_params.method, kwargs={"fill_value": "extrapolate"}))
            else:
                interp_photo = np.asarray([])

        # Recreate the flux array
        if global_params.emulator[0] != 'NA':
            if global_params.emulator[0] == 'PCA':
                if len(mod_dict['vectors_spectro']) != 0:
                    flx_mod_spectro = (mod_dict['flx_mean_spectro']+mod_dict['flx_std_spectro'] * (interp_spectro[1:] @ mod_dict['vectors_spectro'])) * interp_spectro[0][np.newaxis]
                else:
                    flx_mod_spectro = np.asarray([])
                if len(mod_dict['vectors_photo']) != 0:
                    flx_mod_photo = (mod_dict['flx_mean_photo']+mod_dict['flx_std_photo'] * (interp_photo[1:] @ mod_dict['vectors_photo'])) * interp_photo[0][np.newaxis]
                else:
                    flx_mod_photo = np.asarray([])
            elif global_params.emulator[0] == 'NMF':
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
        modif_spec_LL = modif_spec(global_params, theta, theta_index,
                                      obs_dict, 
                                      flx_mod_spectro, flx_mod_photo, 
                                      mod_dict['wav_spectro'], mod_dict['res_spectro'],
                                      indobs=indobs)

        # Get back the modified arrays to compute the logL
        obs_dict_modif = modif_spec_LL[0]
        flx_mod_spectro_modif, flx_mod_photo_modif = modif_spec_LL[1], modif_spec_LL[2]
 
        # If you want to compute the full logL
        logL_full = global_params.logL_full[indobs % len(global_params.logL_full)]
    
        # Computation of the photometry logL
        if len(obs_dict_modif['wav_photo']) != 0:
            if global_params.logL_type[indobs % len(global_params.logL_type)] == 'chi2':
                logL_photo = logL_chi2(obs_dict_modif['flx_photo']-flx_mod_photo_modif, obs_dict_modif['err_photo'], obs_dict['log_det_photo'], full=logL_full)
            elif global_params.logL_type[indobs % len(global_params.logL_type)] == 'chi2_noisescaling':
                logL_photo = logL_chi2_noisescaling(obs_dict_modif['flx_photo']-flx_mod_photo_modif, obs_dict_modif['err_photo'], obs_dict['log_det_photo'], full=logL_full)
            else:
                print()
                print('WARNING: One or more dataset are not included when performing the inversion.')
                print('Please adapt your likelihood function choice.')
                print()
                exit()
        else:
            logL_photo = 0

        # Computation of the spectroscopy logL
        if len(obs_dict_modif['wav_spectro']) != 0:
            if global_params.logL_type[indobs % len(global_params.logL_type)] == 'chi2':
                logL_spectro = logL_chi2(obs_dict_modif['flx_spectro']-flx_mod_spectro_modif, obs_dict_modif['err_spectro'], obs_dict['log_det_spectro'], full=logL_full)
            elif global_params.logL_type[indobs % len(global_params.logL_type)] == 'chi2_covariance' and len(obs_dict_modif['inv_cov']) != 0:
                logL_spectro = logL_chi2_covariance(obs_dict_modif['flx_spectro']-flx_mod_spectro_modif, obs_dict_modif['inv_cov'], obs_dict['log_det_spectro'], full=logL_full)
            elif global_params.logL_type[indobs % len(global_params.logL_type)] == 'chi2_noisescaling':
                logL_spectro = logL_chi2_noisescaling(obs_dict_modif['flx_spectro']-flx_mod_spectro_modif, obs_dict_modif['err_spectro'], obs_dict['log_det_spectro'], full=logL_full)
            elif global_params.logL_type[indobs % len(global_params.logL_type)] == 'chi2_noisescaling_covariance' and len(obs_dict_modif['inv_cov']) != 0:
                logL_spectro = logL_chi2_noisescaling_covariance(obs_dict_modif['flx_spectro']-flx_mod_spectro_modif, obs_dict_modif['inv_cov'], obs_dict['log_det_spectro'], full=logL_full)
            elif global_params.logL_type[indobs % len(global_params.logL_type)] == 'CCF_Brogi':
                logL_spectro = logL_CCF_Brogi(obs_dict_modif['flx_spectro'], flx_mod_spectro_modif)
            elif global_params.logL_type[indobs % len(global_params.logL_type)] == 'CCF_Zucker':
                logL_spectro = logL_CCF_Zucker(obs_dict_modif['flx_spectro'], flx_mod_spectro_modif)
            elif global_params.logL_type[indobs % len(global_params.logL_type)] == 'CCF_custom':
                logL_spectro = logL_CCF_custom(obs_dict_modif['flx_spectro'], flx_mod_spectro_modif, obs_dict_modif['err_spectro'])
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
        return modif_spec_LL


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
    prior_law_par1 = global_params.par1[0]
    if prior_law_par1 != 'NA' and prior_law_par1 != 'constant':
        if prior_law_par1 == 'uniform':
            prior_par1 = uniform_prior([float(global_params.par1[1]), float(global_params.par1[2])], theta[0])
        if prior_law_par1 == 'loguniform':
            prior_par1 = loguniform_prior([float(global_params.par1[1]), float(global_params.par1[2])], theta[0])
        if prior_law_par1 == 'gaussian':
            prior_par1 = gaussian_prior([float(global_params.par1[1]), float(global_params.par1[2])], theta[0])
        if prior_par1 < lim_param_grid[0][0]:
            prior_par1 = lim_param_grid[0][0]
        elif prior_par1 > lim_param_grid[0][1]:
            prior_par1 = lim_param_grid[0][1]
        prior.append(prior_par1)
    prior_law_par2 = global_params.par2[0]
    if prior_law_par2 != 'NA' and prior_law_par2 != 'constant':
        if prior_law_par2 == 'uniform':
            prior_par2 = uniform_prior([float(global_params.par2[1]), float(global_params.par2[2])], theta[1])
        if prior_law_par2 == 'loguniform':
            prior_par2 = loguniform_prior([float(global_params.par2[1]), float(global_params.par2[2])], theta[1])
        if prior_law_par2 == 'gaussian':
            prior_par2 = gaussian_prior([float(global_params.par2[1]), float(global_params.par2[2])], theta[1])
        if prior_par2 < lim_param_grid[1][0]:
            prior_par2 = lim_param_grid[1][0]
        elif prior_par2 > lim_param_grid[1][1]:
            prior_par2 = lim_param_grid[1][1]
        prior.append(prior_par2)
    prior_law_par3 = global_params.par3[0]
    if prior_law_par3 != 'NA' and prior_law_par3 != 'constant':
        if prior_law_par3 == 'uniform':
            prior_par3 = uniform_prior([float(global_params.par3[1]), float(global_params.par3[2])], theta[2])
        if prior_law_par3 == 'loguniform':
            prior_par3 = loguniform_prior([float(global_params.par3[1]), float(global_params.par3[2])], theta[2])
        if prior_law_par3 == 'gaussian':
            prior_par3 = gaussian_prior([float(global_params.par3[1]), float(global_params.par3[2])], theta[2])
        if prior_par3 < lim_param_grid[2][0]:
            prior_par3 = lim_param_grid[2][0]
        elif prior_par3 > lim_param_grid[2][1]:
            prior_par3 = lim_param_grid[2][1]
        prior.append(prior_par3)
    prior_law_par4 = global_params.par4[0]
    if prior_law_par4 != 'NA' and prior_law_par4 != 'constant':
        if prior_law_par4 == 'uniform':
            prior_par4 = uniform_prior([float(global_params.par4[1]), float(global_params.par4[2])], theta[3])
        if prior_law_par4 == 'loguniform':
            prior_par4 = loguniform_prior([float(global_params.par4[1]), float(global_params.par4[2])], theta[3])
        if prior_law_par4 == 'gaussian':
            prior_par4 = gaussian_prior([float(global_params.par4[1]), float(global_params.par4[2])], theta[3])
        if prior_par4 < lim_param_grid[3][0]:
            prior_par4 = lim_param_grid[3][0]
        elif prior_par4 > lim_param_grid[3][1]:
            prior_par4 = lim_param_grid[3][1]
        prior.append(prior_par4)
    prior_law_par5 = global_params.par5[0]
    if prior_law_par5 != 'NA' and prior_law_par5 != 'constant':
        if prior_law_par5 == 'uniform':
            prior_par5 = uniform_prior([float(global_params.par5[1]), float(global_params.par5[2])], theta[4])
        if prior_law_par5 == 'loguniform':
            prior_par5 = loguniform_prior([float(global_params.par5[1]), float(global_params.par5[2])], theta[4])
        if prior_law_par5 == 'gaussian':
            prior_par5 = gaussian_prior([float(global_params.par5[1]), float(global_params.par5[2])], theta[4])
        if prior_par5 < lim_param_grid[4][0]:
            prior_par5 = lim_param_grid[4][0]
        elif prior_par5 > lim_param_grid[4][1]:
            prior_par5 = lim_param_grid[4][1]
        prior.append(prior_par5)

    # Extra-grid parameters
    prior_law_r = global_params.r[0]
    if prior_law_r != 'NA' and prior_law_r != 'constant':
        ind_theta_r = np.where(theta_index == 'r')
        if prior_law_r == 'uniform':
            prior_r = uniform_prior([float(global_params.r[1]), float(global_params.r[2])], theta[ind_theta_r[0][0]])
        if prior_law_r == 'loguniform':
            prior_r = loguniform_prior([float(global_params.r[1]), float(global_params.r[2])], theta[ind_theta_r[0][0]])
        if prior_law_r == 'gaussian':
            prior_r = gaussian_prior([float(global_params.r[1]), float(global_params.r[2])], theta[ind_theta_r[0][0]])
        prior.append(prior_r)
    prior_law_d = global_params.d[0]
    if prior_law_d != 'NA' and prior_law_d != 'constant':
        ind_theta_d = np.where(theta_index == 'd')
        if prior_law_d == 'uniform':
            prior_d = uniform_prior([float(global_params.d[1]), float(global_params.d[2])], theta[ind_theta_d[0][0]])
        if prior_law_d == 'loguniform':
            prior_d = loguniform_prior([float(global_params.d[1]), float(global_params.d[2])], theta[ind_theta_d[0][0]])
        if prior_law_d == 'gaussian':
            prior_d = gaussian_prior([float(global_params.d[1]), float(global_params.d[2])], theta[ind_theta_d[0][0]])
        prior.append(prior_d)

    # - - - - - - - - - - - - - - - - - - - - -

    # Individual parameters / observation
    main_obs_path = global_params.main_observation_path

    if len(global_params.alpha) > 3: # If you want separate alpha for each observations
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            prior_law_alpha = global_params.alpha[indobs*3] # Prior laws should be separeted by 2 values (need to be upgraded)
            if prior_law_alpha != 'NA' and prior_law_alpha != 'constant':
                ind_theta_alpha = np.where(theta_index == f'alpha_{indobs}')
                if prior_law_alpha == 'uniform':
                    prior_alpha = uniform_prior([float(global_params.alpha[indobs*3+1]), float(global_params.alpha[indobs*3+2])], theta[ind_theta_alpha[0][0]])
                if prior_law_alpha == 'loguniform':
                    prior_alpha = loguniform_prior([float(global_params.alpha[indobs*3+1]), float(global_params.alpha[indobs*3+2])], theta[ind_theta_alpha[0][0]])
                if prior_law_alpha == 'gaussian':
                    prior_alpha = gaussian_prior([float(global_params.alpha[indobs*3+1]), float(global_params.alpha[indobs*3+2])], theta[ind_theta_alpha[0][0]])
                prior.append(prior_alpha)
    else: # If you want 1 common alpha for all observations
        prior_law_alpha = global_params.alpha[0] 
        if prior_law_alpha != 'NA' and prior_law_alpha != 'constant':
            ind_theta_alpha = np.where(theta_index == 'alpha')
            if prior_law_alpha == 'uniform':
                prior_alpha = uniform_prior([float(global_params.alpha[1]), float(global_params.alpha[2])], theta[ind_theta_alpha[0][0]])
            if prior_law_alpha == 'loguniform':
                prior_alpha = loguniform_prior([float(global_params.alpha[1]), float(global_params.alpha[2])], theta[ind_theta_alpha[0][0]])
            if prior_law_alpha == 'gaussian':
                prior_alpha = gaussian_prior([float(global_params.alpha[1]), float(global_params.alpha[2])], theta[ind_theta_alpha[0][0]])
            prior.append(prior_alpha)
    if len(global_params.rv) > 3: # If you want separate rv for each observations
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            prior_law_rv = global_params.rv[indobs*3] # Prior laws should be separeted by 2 values (need to be upgraded)
            if prior_law_rv != 'NA' and prior_law_rv != 'constant':
                ind_theta_rv = np.where(theta_index == f'rv_{indobs}')
                if prior_law_rv == 'uniform':
                    prior_rv = uniform_prior([float(global_params.rv[indobs*3+1]), float(global_params.rv[indobs*3+2])], theta[ind_theta_rv[0][0]])
                if prior_law_rv == 'loguniform':
                    prior_rv = loguniform_prior([float(global_params.rv[indobs*3+1]), float(global_params.rv[indobs*3+2])], theta[ind_theta_rv[0][0]])
                if prior_law_rv == 'gaussian':
                    prior_rv = gaussian_prior([float(global_params.rv[indobs*3+1]), float(global_params.rv[indobs*3+2])], theta[ind_theta_rv[0][0]])
                prior.append(prior_rv)
    else: # If you want 1 common rv for all observations
        prior_law_rv = global_params.rv[0] 
        if prior_law_rv != 'NA' and prior_law_rv != 'constant':
            ind_theta_rv = np.where(theta_index == 'rv')
            if prior_law_rv == 'uniform':
                prior_rv = uniform_prior([float(global_params.rv[1]), float(global_params.rv[2])], theta[ind_theta_rv[0][0]])
            if prior_law_rv == 'loguniform':
                prior_rv = loguniform_prior([float(global_params.rv[1]), float(global_params.rv[2])], theta[ind_theta_rv[0][0]])
            if prior_law_rv == 'gaussian':
                prior_rv = gaussian_prior([float(global_params.rv[1]), float(global_params.rv[2])], theta[ind_theta_rv[0][0]])
            prior.append(prior_rv)
    if len(global_params.vsini) > 4: # If you want separate vsini for each observations
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            prior_law_vsini = global_params.vsini[4*indobs] # Prior laws should be separeted by 2 values (need to be upgraded)
            if prior_law_vsini != 'NA' and prior_law_vsini != 'constant':
                ind_theta_vsini = np.where(theta_index == f'vsini_{indobs}')
                if prior_law_vsini == 'uniform':
                    prior_vsini = uniform_prior([float(global_params.vsini[indobs*4+1]), float(global_params.vsini[indobs*4+2])], theta[ind_theta_vsini[0][0]])
                if prior_law_vsini == 'loguniform':
                    prior_vsini = loguniform_prior([float(global_params.vsini[indobs*4+1]), float(global_params.vsini[indobs*4+2])], theta[ind_theta_vsini[0][0]])
                if prior_law_vsini == 'gaussian':
                    prior_vsini = gaussian_prior([float(global_params.vsini[indobs*4+1]), float(global_params.vsini[indobs*4+2])], theta[ind_theta_vsini[0][0]])
                prior.append(prior_vsini)
    else: # If you want 1 common vsini for all observations
        prior_law_vsini = global_params.vsini[0]
        if prior_law_vsini != 'NA' and prior_law_vsini != 'constant':
            ind_theta_vsini = np.where(theta_index == 'vsini')
            if prior_law_vsini == 'uniform':
                prior_vsini = uniform_prior([float(global_params.vsini[1]), float(global_params.vsini[2])], theta[ind_theta_vsini[0][0]])
            if prior_law_vsini == 'loguniform':
                prior_vsini = loguniform_prior([float(global_params.vsini[1]), float(global_params.vsini[2])], theta[ind_theta_vsini[0][0]])
            if prior_law_vsini == 'gaussian':
                prior_vsini = gaussian_prior([float(global_params.vsini[1]), float(global_params.vsini[2])], theta[ind_theta_vsini[0][0]])
            prior.append(prior_vsini)
    if len(global_params.ld) > 3: # If you want separate ld for each observations
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            prior_law_ld = global_params.ld[3*indobs] # Prior laws should be separeted by 2 values (need to be upgraded)
            if prior_law_ld != 'NA' and prior_law_ld != 'constant':
                ind_theta_ld = np.where(theta_index == f'ld_{indobs}')
                if prior_law_ld == 'uniform':
                    prior_ld = uniform_prior([float(global_params.ld[indobs*3+1]), float(global_params.ld[indobs*3+2])], theta[ind_theta_ld[0][0]])
                if prior_law_ld == 'loguniform':
                    prior_ld = loguniform_prior([float(global_params.ld[indobs*3+1]), float(global_params.ld[indobs*3+2])], theta[ind_theta_ld[0][0]])
                if prior_law_ld == 'gaussian':
                    prior_ld = gaussian_prior([float(global_params.ld[indobs*3+1]), float(global_params.ld[indobs*3+2])], theta[ind_theta_ld[0][0]])
                prior.append(prior_ld)
    else: # If you want 1 common ld for all observations
        prior_law_ld = global_params.ld[0] # Prior laws should be separeted by 2 values (need to be upgraded)
        if prior_law_ld != 'NA' and prior_law_ld != 'constant':
            ind_theta_ld = np.where(theta_index == 'ld')
            if prior_law_ld == 'uniform':
                prior_ld = uniform_prior([float(global_params.ld[1]), float(global_params.ld[2])], theta[ind_theta_ld[0][0]])
            if prior_law_ld == 'loguniform':
                 prior_ld = loguniform_prior([float(global_params.ld[1]), float(global_params.ld[2])], theta[ind_theta_ld[0][0]])
            if prior_law_ld == 'gaussian':
                prior_ld = gaussian_prior([float(global_params.ld[1]), float(global_params.ld[2])], theta[ind_theta_ld[0][0]])
            prior.append(prior_ld)

    # - - - - - - - - - - - - - - - - - - - - -

    prior_law_av = global_params.av[0]
    if prior_law_av != 'NA' and prior_law_av != 'constant':
        ind_theta_av = np.where(theta_index == 'av')
        if prior_law_av == 'uniform':
            prior_av = uniform_prior([float(global_params.av[1]), float(global_params.av[2])], theta[ind_theta_av[0][0]])
        if prior_law_av == 'loguniform':
            prior_av = loguniform_prior([float(global_params.av[1]), float(global_params.av[2])], theta[ind_theta_av[0][0]])
        if prior_law_av == 'gaussian':
            prior_av = gaussian_prior([float(global_params.av[1]), float(global_params.av[2])], theta[ind_theta_av[0][0]])
        prior.append(prior_av)
    prior_law_bb_t = global_params.bb_t[0] 
    if prior_law_bb_t != 'NA' and prior_law_bb_t != 'constant':
        ind_theta_bb_t = np.where(theta_index == 'bb_t')
        if prior_law_bb_t == 'uniform':
            prior_bb_t = uniform_prior([float(global_params.bb_t[1]), float(global_params.bb_t[2])], theta[ind_theta_bb_t[0][0]])
        if prior_law_bb_t == 'loguniform':
            prior_bb_t = loguniform_prior([float(global_params.bb_t[1]), float(global_params.bb_t[2])], theta[ind_theta_bb_t[0][0]])
        if prior_law_bb_t == 'gaussian':
            prior_bb_t = gaussian_prior([float(global_params.bb_t[1]), float(global_params.bb_t[2])], theta[ind_theta_bb_t[0][0]])
        prior.append(prior_bb_t)
    prior_law_bb_r = global_params.bb_r[0] 
    if prior_law_bb_r != 'NA' and prior_law_bb_r != 'constant':
        ind_theta_bb_r = np.where(theta_index == 'bb_r')
        if prior_law_bb_r == 'uniform':
            prior_bb_r = uniform_prior([float(global_params.bb_r[1]), float(global_params.bb_r[2])], theta[ind_theta_bb_r[0][0]])
        if prior_law_bb_r == 'loguniform':
            prior_bb_r = loguniform_prior([float(global_params.bb_r[1]), float(global_params.bb_r[2])], theta[ind_theta_bb_r[0][0]])
        if prior_law_bb_r == 'gaussian':
            prior_bb_r = gaussian_prior([float(global_params.bb_r[1]), float(global_params.bb_r[2])], theta[ind_theta_bb_r[0][0]])
        prior.append(prior_bb_r)
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
        print(obs_name + ' will be computed with ' + global_params.logL_type[indobs % len(global_params.logL_type)])
        
        if global_params.logL_type[indobs % len(global_params.logL_type)] == 'CCF_Brogi' and global_params.res_cont[indobs % len(global_params.res_cont)] == 'NA' and global_params.hc_type[indobs % len(global_params.hc_type)] != 'nofit_rm_spec':
            print('WARNING. You cannot use CCF mappings without substracting the continuum')
            print()
            exit()
        elif global_params.logL_type[indobs % len(global_params.logL_type)] == 'CCF_Zucker' and global_params.res_cont[indobs % len(global_params.res_cont)] == 'NA' and global_params.hc_type[indobs % len(global_params.hc_type)] != 'nofit_rm_spec':
            print('WARNING. You cannot use CCF mappings without substracting the continuum')
            print()
            exit()
        elif global_params.logL_type[indobs % len(global_params.logL_type)] == 'CCF_custom' and global_params.res_cont[indobs % len(global_params.res_cont)] == 'NA' and global_params.hc_type[indobs % len(global_params.hc_type)] != 'nofit_rm_spec':
            print('WARNING. You cannot use CCF mappings without substracting the continuum')
            print()
            exit()

        print()
    print('Done !')
    print()

    theta_index = []
    lim_param_grid = []
    n_free_parameters = 0
    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine='netcdf4')
    # Count the number of free parameters and identify the parameter position in theta
    if global_params.par1[0] != 'NA':
        if global_params.par1[0] != 'constant':    
            lim_param_grid.append([min(ds['par1'].values), max(ds['par1'].values)])
            theta_index.append('par1')
            n_free_parameters += 1
        else:
            lim_param_grid.append([float(global_params.par1[1]), float(global_params.par1[1])])
    if global_params.par2[0] != 'NA':
        if global_params.par2[0] != 'constant':
            lim_param_grid.append([min(ds['par2'].values), max(ds['par2'].values)])
            theta_index.append('par2')
            n_free_parameters += 1
        else:
            lim_param_grid.append([float(global_params.par2[1]), float(global_params.par2[1])])
    if global_params.par3[0] != 'NA':
        if global_params.par3[0] != 'constant':
            lim_param_grid.append([min(ds['par3'].values), max(ds['par3'].values)])
            theta_index.append('par3')
            n_free_parameters += 1
        else:
            lim_param_grid.append([float(global_params.par3[1]), float(global_params.par3[1])])
    if global_params.par4[0] != 'NA':
        if global_params.par4[0]!= 'constant':
            lim_param_grid.append([min(ds['par4'].values), max(ds['par4'].values)])
            theta_index.append('par4')
            n_free_parameters += 1
        else:
            lim_param_grid.append([float(global_params.par4[1]), float(global_params.par4[1])])
    if global_params.par5[0] != 'NA':
        if global_params.par5[0] != 'constant':
            lim_param_grid.append([min(ds['par5'].values), max(ds['par5'].values)])
            theta_index.append('par5')
            n_free_parameters += 1
        else:
            lim_param_grid.append([float(global_params.par5[1]), float(global_params.par5[1])])
            
    if global_params.r[0] != 'NA' and global_params.r[0] != 'constant':
        n_free_parameters += 1
        theta_index.append('r')
    if global_params.d[0] != 'NA' and global_params.d[0] != 'constant':
        n_free_parameters += 1
        theta_index.append('d')

    # - - - - - - - - - - - - - - - - - - - - -

    # Individual parameters / observation
    main_obs_path = global_params.main_observation_path

    if len(global_params.alpha) > 3:
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            if global_params.alpha[indobs*3] != 'NA' and global_params.alpha[indobs*3] != 'constant': # Check if the idobs is different from constant
                n_free_parameters += 1
                theta_index.append(f'alpha_{indobs}')
    else:
        if global_params.alpha[0] != 'NA' and global_params.alpha[0] != 'constant':
            n_free_parameters += 1
            theta_index.append(f'alpha')
    if len(global_params.rv) > 3:
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            if global_params.rv[indobs*3] != 'NA' and global_params.rv[indobs*3] != 'constant': # Check if the idobs is different from constant
                n_free_parameters += 1
                theta_index.append(f'rv_{indobs}')
    else:
        if global_params.rv[0] != 'NA' and global_params.rv[0] != 'constant':
            n_free_parameters += 1
            theta_index.append(f'rv')
    if len(global_params.vsini) > 4:
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            if global_params.vsini[indobs*4] != 'NA' and global_params.vsini[indobs*4] != 'constant': # Check if the idobs is different from constant
                n_free_parameters += 1
                theta_index.append(f'vsini_{indobs}')
    else:
        if global_params.vsini[0] != 'NA' and global_params.vsini[0] != 'constant':
            n_free_parameters += 1
            theta_index.append('vsini')
    if len(global_params.ld) > 3:
        for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
            if global_params.ld[indobs*3] != 'NA' and global_params.ld[indobs*3] != 'constant': # Check if the idobs is different from constant
                n_free_parameters += 1
                theta_index.append(f'ld_{indobs}')
    else:
        if global_params.ld[0] != 'NA' and global_params.ld[0] != 'constant':
            n_free_parameters += 1
            theta_index.append('ld')

    # - - - - - - - - - - - - - - - - - - - - -

    if global_params.av[0] != 'NA' and global_params.av[indobs] != 'constant':
        n_free_parameters += 1
        theta_index.append('av')
    ## adding cpd
    if global_params.bb_t[0] != 'NA' and global_params.bb_t[indobs] != 'constant':
        n_free_parameters += 1
        theta_index.append('bb_t')
    if global_params.bb_r[0] != 'NA' and global_params.bb_r[indobs] != 'constant':
        n_free_parameters += 1
        theta_index.append('bb_r')
    theta_index = np.asarray(theta_index)
    

    # Import all the data (only done once)
    main_file = import_obsmod(global_params)

    if global_params.ns_algo == 'nestle':
        os.makedirs(global_params.result_path + '/nestle/', exist_ok=True)
        tmpstot1 = time.time()
        loglike_gp = lambda theta: loglike(theta, theta_index, global_params, main_file=main_file)
        prior_transform_gp = lambda theta: prior_transform(theta, theta_index, lim_param_grid, global_params)
        result = nestle.sample(
                               loglike_gp, prior_transform_gp, n_free_parameters,
                               npoints=global_params.npoint,
                               method=global_params.n_method,
                               update_interval=global_params.n_update_interval,
                               npdim=global_params.n_npdim,
                               maxiter=global_params.n_maxiter,
                               maxcall=global_params.n_maxcall,
                               dlogz=global_params.n_dlogz,
                               decline_factor=global_params.n_decline_factor,
                               rstate=global_params.n_rstate,
                               callback=nestle.print_progress
                               )
        # Reformat the result file
        with open(global_params.result_path + '/nestle/RAW.pic', 'wb') as f1:
            pickle.dump(result, f1)
        logz = [result['logz'], result['logzerr']]
        samples = result['samples']
        weights = result['weights']
        logvol = result['logvol']
        logl = result['logl']
        tmpstot2 = time.time()-tmpstot1
        if tmpstot2 < 60:
            time_spent = f'{tmpstot2:.1f} sec'
        elif tmpstot2 < 3600:
            time_spent = f'{tmpstot2/60:.1f} min'
        else:
            time_spent = f'{tmpstot2/3600:.1f} hours'

        print(' ')
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
        print('-> Nestle  ')
        print(' ')
        print(f'The code spent {time_spent} to run.')
        print(result.summary())
        print('\n')

    if global_params.ns_algo == 'pymultinest':
        os.makedirs(global_params.result_path + '/pymultinest/', exist_ok=True)
        import pymultinest
        tmpstot1 = time.time()
        loglike_gp = lambda theta: loglike(theta, theta_index, global_params, main_file=main_file)
        prior_transform_gp = lambda theta: prior_transform(theta, theta_index, lim_param_grid, global_params)
        result = pymultinest.solve(
                        LogLikelihood=loglike_gp,
                        Prior=prior_transform_gp,
                        n_dims=n_free_parameters,
                        n_clustering_params=global_params.pm_n_clustering_params,
                        wrapped_params=global_params.pm_wrapped_params,
                        importance_nested_sampling=global_params.pm_importance_nested_sampling,
                        multimodal=global_params.pm_multimodal,
                        const_efficiency_mode=global_params.pm_const_efficiency_mode,
                        n_live_points=global_params.npoint,
                        evidence_tolerance=global_params.pm_evidence_tolerance,
                        sampling_efficiency=global_params.pm_sampling_efficiency,
                        n_iter_before_update=global_params.pm_n_iter_before_update,
                        null_log_evidence=global_params.pm_null_log_evidence,
                        max_modes=global_params.pm_max_modes,
                        mode_tolerance=global_params.pm_mode_tolerance,
                        outputfiles_basename=global_params.result_path + '/pymultinest/' + 'RAW_',
                        seed=global_params.pm_seed,
                        verbose=global_params.pm_verbose,
                        resume=global_params.pm_resume,
                        context=global_params.pm_context,
                        log_zero=global_params.pm_log_zero,
                        max_iter=global_params.pm_max_iter,
                        init_MPI=global_params.pm_init_MPI,
                        dump_callback=global_params.pm_dump_callback,
                        use_MPI=global_params.pm_use_MPI
                        )
        # Reformat the result file
        with open(global_params.result_path + '/pymultinest/' + 'RAW_stats.dat',
                  'rb') as open_dat:
            for l, line in enumerate(open_dat):
                if l == 0:
                    line = line.strip().split()
                    logz_multi = float(line[5])
                    logzerr_multi = float(line[7])
        sample_multi = []
        logl_multi = []
        logvol_multi = []
        with open(global_params.result_path + '/pymultinest/' + 'RAW_ev.dat',
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
        with open(global_params.result_path + '/pymultinest/' + 'RAW_.txt',
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
        if tmpstot2 < 60:
            time_spent = f'{tmpstot2:.1f} sec'
        elif tmpstot2 < 3600:
            time_spent = f'{tmpstot2/60:.1f} min'
        else:
            time_spent = f'{tmpstot2/3600:.1f} hours'

        print(' ')
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
        print('-> PyMultinest  ')
        print(' ')
        print(f'The code spent {time_spent} to run.')
        print('The evidence is: %(logZ).1f +- %(logZerr).1f' % result)
        print('The parameter values are:')
        for name, col in zip(theta_index, result['samples'].transpose()):
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
        print('\n')

    if global_params.ns_algo == 'ultranest':
        os.makedirs(global_params.result_path + '/ultranest/', exist_ok=True)
        import ultranest
        from ultranest import integrator
        tmpstot1 = time.time()
        loglike_gp = lambda theta: loglike(theta, theta_index, global_params, main_file=main_file)
        prior_transform_gp = lambda cube: prior_transform(cube, theta_index, lim_param_grid, global_params)
        sampler = ultranest.ReactiveNestedSampler(
                                param_names=list(theta_index),
                                loglike=loglike_gp,
                                transform=prior_transform_gp,
                                log_dir=global_params.result_path + '/ultranest/',
                                resume=global_params.u_resume,
                                wrapped_params=global_params.u_wrapped_params,
                                run_num=global_params.u_run_num,
                                num_test_samples=global_params.u_num_test_samples,
                                draw_multiple=global_params.u_draw_multiple,
                                num_bootstraps=global_params.u_num_bootstraps,
                                vectorized=global_params.u_vectorized,
                                ndraw_min=global_params.u_ndraw_min,
                                ndraw_max=global_params.u_ndraw_max,
                                storage_backend=global_params.u_storage_backend,
                                warmstart_max_tau=global_params.u_warmstart_max_tau
                                )
        _ = sampler.run(
                min_num_live_points=global_params.npoint,
                update_interval_volume_fraction=global_params.u_update_interval_volume_fraction,
                update_interval_ncall=global_params.u_update_interval_ncall,
                log_interval=global_params.u_log_interval,
                show_status=global_params.u_show_status,
                viz_callback=global_params.u_viz_callback,
                dlogz=global_params.u_dlogz,
                dKL=global_params.u_dKL,
                frac_remain=global_params.u_frac_remain,
                Lepsilon=global_params.u_Lepsilon,
                max_iters=global_params.u_max_iters,
                max_ncalls=global_params.u_max_ncalls,
                max_num_improvement_loops=global_params.u_max_num_improvement_loops,
                cluster_num_live_points=global_params.u_cluster_num_live_points,
                insertion_test_zscore_threshold=global_params.u_insertion_test_zscore_threshold,
                insertion_test_window=global_params.u_insertion_test_window,
                widen_before_initial_plateau_num_warn=global_params.u_widen_before_initial_plateau_num_warn,
                widen_before_initial_plateau_num_max=global_params.u_widen_before_initial_plateau_num_max
                )
        result = integrator.read_file(
                                global_params.result_path + '/ultranest/',
                                x_dim=len(theta_index),
                                num_bootstraps=global_params.u_num_bootstraps
                                )
        # Reformat the result file
        with open(global_params.result_path + '/ultranest/RAW.pic', 'wb') as f1:
            pickle.dump(result, f1)
        logz = [result[-1]['logz'], result[-1]['logzerr']]
        samples = result[-1]['samples']
        weights = result[-1]['weighted_samples']['weights']
        logvol = result[0]['logvol']  # Not always used in UltraNest
        logl = result[0]['logl']
        tmpstot2 = time.time() - tmpstot1
        if tmpstot2 < 60:
            time_spent = f'{tmpstot2:.1f} sec'
        elif tmpstot2 < 3600:
            time_spent = f'{tmpstot2/60:.1f} min'
        else:
            time_spent = f'{tmpstot2/3600:.1f} hours'

        print(' ')
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
        print('-> UltraNest')
        print(' ')
        print(f'The code spent {time_spent} to run.')
        print(f"Final logZ = {logz[0]:.3f} ± {logz[1]:.3f}")
        print('\n')

    result_reformat = {"samples": samples,
                       "weights": weights,
                       "logl": logl,
                       "logvol": logvol,
                       "logz": logz,}

    with open(global_params.result_path + '/result_' + global_params.ns_algo + '.pic', "wb") as tf:
        pickle.dump(result_reformat, tf)

    return

# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    from ForMoSA.global_file import GlobFile

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
