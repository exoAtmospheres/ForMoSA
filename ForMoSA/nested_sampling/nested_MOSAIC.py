import numpy as np
import os
import glob
import xarray as xr

from nested_sampling.nested_modif_spec import modif_spec
from nested_sampling.nested_prior_function import uniform_prior, gaussian_prior
from nested_sampling.nested_logL_functions import *
from main_utilities import yesno, diag_mat


def MOSAIC_logL(theta, theta_index, global_params, main_file):
    """
    Function that calculates the logarithm of the likelihood for MOSAIC version. 
    Main difference with standard logL function is that here we calculate logL by adding the different observations.
    
    Args:
        theta           (list): Parameter values randomly picked by the nested sampling
        theta_index     (list): Index for the parameter values randomly picked
        global_params (object): Class containing every input from the .ini file.
        main_file (list(list)): List containing the wavelengths, flux, errors, covariance, and grid information
        
    Returns:
        FINAL_logL     (float): Final evaluated loglikelihood for both spectra and photometry. 

    Authors: Simon Petrus and Matthieu Ravet
    """
    
    # Recovery of each observation spectroscopy and photometry data

    main_obs_path = global_params.main_observation_path
    FINAL_logL = 0

    for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
        
        # Recovery of spectroscopy and photometry data
        wav_obs_merge = main_file[indobs][0][0]
        wav_obs_phot = main_file[indobs][0][1]
        flx_obs_merge = main_file[indobs][1][0]
        flx_obs_phot = main_file[indobs][1][1]
        err_obs_merge = main_file[indobs][2][0]
        err_obs_phot = main_file[indobs][2][1]
        inv_cov_obs_merge = main_file[indobs][3]
        transm_obs_merge = main_file[indobs][4]
        star_flx_obs_merge = main_file[indobs][5]

        # Recovery of the spectroscopy and photometry model
        grid_merge = main_file[indobs][6]
        grid_phot = main_file[indobs][7]

        # Calculation of the likelihood for each sub-spectrum defined by the parameter 'wav_fit'
        for ns_u_ind, ns_u in enumerate(global_params.wav_fit[indobs].split('/')):
            
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
                    flx_mod_merge_cut = np.asarray(grid_merge_cut.interp(par1=theta[0], par2=theta[1],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_merge_cut = np.asarray([])
                if len(grid_phot_cut['wavelength']) != 0:
                    flx_mod_phot_cut = np.asarray(grid_phot_cut.interp(par1=theta[0], par2=theta[1],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_phot_cut = np.asarray([])
            elif global_params.par4 == 'NA':
                if len(grid_merge_cut['wavelength']) != 0:
                    flx_mod_merge_cut = np.asarray(grid_merge_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_merge_cut = np.asarray([])
                if len(grid_phot_cut['wavelength']) != 0:
                    flx_mod_phot_cut = np.asarray(grid_phot_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_phot_cut = np.asarray([])
            elif global_params.par5 == 'NA':
                if len(grid_merge_cut['wavelength']) != 0:
                    flx_mod_merge_cut = np.asarray(grid_merge_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_merge_cut = np.asarray([])
                if len(grid_phot_cut['wavelength']) != 0:
                    flx_mod_phot_cut = np.asarray(grid_phot_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_phot_cut = np.asarray([])
            else:
                if len(grid_merge_cut['wavelength']) != 0:
                    flx_mod_merge_cut = np.asarray(grid_merge_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            par5=theta[4],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_merge_cut = np.asarray([])
                if len(grid_phot_cut['wavelength']) != 0:
                    flx_mod_phot_cut = np.asarray(grid_phot_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            par5=theta[4],
                                                            method="linear", kwargs={"fill_value": "extrapolate"}))
                else:
                    flx_mod_phot_cut = np.asarray([])


            # Re-merging of the data and interpolated synthetic spectrum to a wavelength grid defined by the parameter 'wav_fit'
            ind_merge = np.where((wav_obs_merge >= min_ns_u) & (wav_obs_merge <= max_ns_u))
            ind_phot = np.where((wav_obs_phot >= min_ns_u) & (wav_obs_phot <= max_ns_u))
            if ns_u_ind == 0:
                wav_obs_merge_ns_u = wav_obs_merge[ind_merge]
                flx_obs_merge_ns_u = flx_obs_merge[ind_merge]
                err_obs_merge_ns_u = err_obs_merge[ind_merge]
                flx_mod_merge_ns_u = flx_mod_merge_cut
                if len(inv_cov_obs_merge) != 0:  # Add covariance in the loop (if necessary)
                    inv_cov_obs_merge_ns_u = inv_cov_obs_merge[np.ix_(ind_merge[0],ind_merge[0])]
                else:
                    inv_cov_obs_merge_ns_u = np.asarray([])
                if len(transm_obs_merge) != 0:
                    transm_obs_merge_ns_u = transm_obs_merge[ind_merge]
                else:
                    transm_obs_merge_ns_u = np.asarray([])
                if len(star_flx_obs_merge) != 0:
                    star_flx_obs_merge_ns_u = star_flx_obs_merge[ind_merge]
                else:
                    star_flx_obs_merge_ns_u = np.asarray([])
                wav_obs_phot_ns_u = wav_obs_phot[ind_phot]
                flx_obs_phot_ns_u = flx_obs_phot[ind_phot]
                err_obs_phot_ns_u = err_obs_phot[ind_phot]
                flx_mod_phot_ns_u = flx_mod_phot_cut
            else:
                wav_obs_merge_ns_u = np.concatenate((wav_obs_merge_ns_u, wav_obs_merge[ind_merge]))
                flx_obs_merge_ns_u = np.concatenate((flx_obs_merge_ns_u, flx_obs_merge[ind_merge]))
                err_obs_merge_ns_u = np.concatenate((err_obs_merge_ns_u, err_obs_merge[ind_merge]))
                flx_mod_merge_ns_u = np.concatenate((flx_mod_merge_ns_u, flx_mod_merge_cut))
                if len(inv_cov_obs_merge_ns_u) != 0: # Merge the covariance matrices (if necessary)
                    inv_cov_obs_merge_ns_u = diag_mat([inv_cov_obs_merge_ns_u, inv_cov_obs_merge[np.ix_(ind_merge[0],ind_merge[0])]])
                if len(transm_obs_merge_ns_u) != 0:
                    transm_obs_merge_ns_u = np.concatenate((transm_obs_merge_ns_u, transm_obs_merge[ind_merge]))
                if len(star_flx_obs_merge_ns_u) != 0:    
                    star_flx_obs_merge_ns_u = np.concatenate((star_flx_obs_merge_ns_u, star_flx_obs_merge[ind_grid_merge_sel]))
                wav_obs_phot_ns_u = np.concatenate((wav_obs_phot_ns_u, wav_obs_phot[ind_phot]))
                flx_obs_phot_ns_u = np.concatenate((flx_obs_phot_ns_u, flx_obs_phot[ind_phot]))
                err_obs_phot_ns_u = np.concatenate((err_obs_phot_ns_u, err_obs_phot[ind_phot]))
                flx_mod_phot_ns_u = np.concatenate((flx_mod_phot_ns_u, flx_mod_phot_cut))
        
                    
        # Modification of the synthetic spectrum with the extra-grid parameters
        modif_spec_LL = modif_spec(global_params, theta, theta_index,
                                    wav_obs_merge_ns_u,  flx_obs_merge_ns_u,  err_obs_merge_ns_u,  flx_mod_merge_ns_u,
                                    wav_obs_phot_ns_u,  flx_obs_phot_ns_u, err_obs_phot_ns_u,  flx_mod_phot_ns_u, 
                                    transm_obs_merge_ns_u, star_flx_obs_merge_ns_u, indobs=indobs)
        
        flx_obs, flx_obs_phot = modif_spec_LL[1], modif_spec_LL[5]
        flx_mod, flx_mod_phot = modif_spec_LL[3], modif_spec_LL[7]
        err, err_phot = modif_spec_LL[2], modif_spec_LL[6]
        inv_cov = inv_cov_obs_merge_ns_u
        ck = modif_spec_LL[8]

        # Computation of the photometry logL
        if len(flx_mod_phot) != 0:
            logL_phot = logL_chi2_classic(flx_obs_phot-flx_mod_phot, err_phot)
        else:
            logL_phot = 0

        # Computation of the spectroscopy logL
        if len(flx_obs) != 0:
            if global_params.logL_type[indobs] == 'chi2_classic':
                logL_spec = logL_chi2_classic(flx_obs-flx_mod, err)
            elif global_params.logL_type[indobs] == 'chi2_covariance' and len(inv_cov) != 0:
                logL_spec = logL_chi2_covariance(flx_obs-flx_mod, inv_cov)
            elif global_params.logL_type[indobs] == 'CCF_Brogi':
                logL_spec = logL_CCF_Brogi(flx_obs, flx_mod)
            elif global_params.logL_type[indobs] == 'CCF_Zucker':
                logL_spec = logL_CCF_Zucker(flx_obs, flx_mod)
            elif global_params.logL_type[indobs] == 'CCF_custom':
                logL_spec = logL_CCF_custom(flx_obs, flx_mod, err)
            else:
                print()
                print('WARNING: One or more dataset are not included when performing the inversion.')
                print('Please adapt your likelihood function choice.')
                print()
                exit()
        else:
            logL_spec = 0

        # Compute the final logL (sum of all likelihood under the hypothesis of independent instruments)
        FINAL_logL = logL_phot + logL_spec + FINAL_logL

    return FINAL_logL