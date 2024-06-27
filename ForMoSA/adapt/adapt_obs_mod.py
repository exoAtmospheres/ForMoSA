from __future__ import print_function, division
import numpy as np
import os
import xarray as xr
from scipy.interpolate import interp1d

from adapt.extraction_functions import extract_observation
from adapt.adapt_grid import adapt_grid
from main_utilities import diag_mat
import glob
# import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------


def launch_adapt(global_params, justobs='no'):
    """
    Adapt the synthetic spectra of a grid to make them comparable with the data.
    
    Args:
        global_params  (object): Class containing each parameter
        justobs    ('yes'/'no'): 'no' by default to also adapt the grid
    Returns:
        None

    Author: Simon Petrus / Adapted: Matthieu Ravet, Paulina Palma-Bifani and Allan Denis
    """

    # Get back the grid information from the config file
    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine="netcdf4")
    wav_mod_nativ = ds["wavelength"].values
    attr = ds.attrs
    res_mod_nativ = attr['res']
    ds.close()

    # Extract the data from the observation files
    main_obs_path = global_params.main_observation_path


    for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
        
        global_params.observation_path = obs
        obs_name = os.path.splitext(os.path.basename(global_params.observation_path))[0]

        # Estimate and subtract the continuum (if needed) + check-ups
        if global_params.continuum_sub[indobs] != 'NA':
            print()
            print(obs_name + ' will have a R=' + global_params.continuum_sub[indobs] + ' continuum removed using a ' 
                + global_params.wav_for_continuum[indobs] + ' wavelength range')
            print()
            obs_spectro, obs_photo, obs_spectro_ins, obs_photo_ins, obs_opt = extract_observation(global_params, wav_mod_nativ, res_mod_nativ, 'yes', 
                                                                                                  obs_name=obs_name, indobs=indobs)
        else:
            obs_spectro, obs_photo, obs_spectro_ins, obs_photo_ins, obs_opt = extract_observation(global_params, wav_mod_nativ, res_mod_nativ,
                                                                                                   obs_name=obs_name, indobs=indobs)


        # Merging of each sub-spectrum and interpolating the grid
        for c, cut in enumerate(obs_spectro):

            # Interpolate the resolution onto the wavelength of the data
            ind_mod_obs = np.where((wav_mod_nativ <= cut[0][-1]) & (wav_mod_nativ > cut[0][0]))
            wav_mod_cut = wav_mod_nativ[ind_mod_obs]
            res_mod_cut = res_mod_nativ[ind_mod_obs]
            interp_mod_to_obs = interp1d(wav_mod_cut, res_mod_cut, fill_value='extrapolate')
            res_mod_cut = interp_mod_to_obs(cut[0])

            if c == 0:
                wav_obs_extract = obs_spectro[c][0]
                flx_obs_extract = obs_spectro[c][1]
                err_obs_extract = obs_spectro[c][2]
                res_obs_extract = obs_spectro[c][3]
                cov_obs_extract = obs_opt[c][0]
                transm_obs_extract = obs_opt[c][1]
                star_flx_obs_extract = obs_opt[c][2]
                system_obs_extract = obs_opt[c][3]
                # Save the interpolated resolution of the grid
                res_mod_obs_merge = [res_mod_cut]

            else:
                wav_obs_extract = np.concatenate((wav_obs_extract, obs_spectro[c][0]))
                flx_obs_extract = np.concatenate((flx_obs_extract, obs_spectro[c][1]))
                err_obs_extract = np.concatenate((err_obs_extract, obs_spectro[c][2]))
                res_obs_extract = np.concatenate((res_obs_extract, obs_spectro[c][3]))
                if len(cov_obs_extract) != 0:
                    cov_obs_extract = diag_mat([cov_obs_extract, obs_opt[c][0]])
                if len(transm_obs_extract) != 0:
                    transm_obs_extract = np.concatenate((transm_obs_extract, obs_opt[c][1]))
                if len(star_flx_obs_extract) != 0:
                    star_flx_obs_extract = np.concatenate((star_flx_obs_extract, obs_opt[c][2]))
                if len(system_obs_extract) != 0:
                    system_obs_extract = np.concatenate((system_obs_extract, obs_opt[c][3]), axis=0)
                # Save the interpolated resolution of the grid
                res_mod_obs_merge.append(res_mod_cut)


            # Compute the inverse of the merged covariance matrix (note: inv(C1, C2) = (in(C1), in(C2)) if C1 and C2 are block matrix on the diagonal)
            # if necessary
            if len(cov_obs_extract) != 0:
                inv_cov_obs_extract = np.linalg.inv(cov_obs_extract)
            else:
                inv_cov_obs_extract = np.asarray([])

            # Compile everything and changing data type to object to allow for different array sizes
            obs_spectro_merge = np.asarray([wav_obs_extract, flx_obs_extract, err_obs_extract, res_obs_extract])
            obs_spectro = np.asarray(obs_spectro, dtype=object)
            obs_spectro_ins = np.asarray(obs_spectro_ins, dtype=object)
            obs_photo = np.asarray(obs_photo, dtype=object)
            obs_photo_ins = np.asarray(obs_photo_ins, dtype=object)
            obs_opt_merge = np.asarray([inv_cov_obs_extract, transm_obs_extract, star_flx_obs_extract, system_obs_extract], dtype=object)

            # Check-ups and warnings for negative values in the diagonal of the covariance matrix
            if len(cov_obs_extract) != 0 and any(np.diag(cov_obs_extract) < 0):
                print()
                print("WARNING: Negative value(s) is(are) present on the diagonal of the covariance matrix.") 
                print("Operation aborted.")
                print()
                exit()

        # Save the new data spectrum
        np.savez(os.path.join(global_params.result_path, f'spectrum_obs_{obs_name}.npz'),
                    obs_spectro_merge=obs_spectro_merge,
                    obs_spectro=obs_spectro,
                    obs_spectro_ins=obs_spectro_ins,
                    obs_photo=obs_photo,
                    obs_photo_ins=obs_photo_ins,
                    obs_opt_merge=obs_opt_merge) # Optional arrays kept separatly
        
        # Adaptation of the model grid
        if justobs == 'no':
            # Creation of the repertory to store the adapted grid (if needed)
            if os.path.isdir(global_params.adapt_store_path):
                pass
            else:
                os.mkdir(global_params.adapt_store_path)

            print()
            print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
            print("-> To compare synthetic spectra with the observation we need to manage them. The following actions are performed:")
            print("- extraction -")
            print("- resizing on the observation's wavelength range -")
            print("- adjustement of the spectral resolution -")
            print("- substraction of the continuum (if needed) -")
            print()
            print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
            print(f"-> Sarting the adaptation of {obs_name}")

            adapt_grid(global_params, obs_spectro_merge[0], obs_photo[0], res_mod_obs_merge, obs_name=obs_name, indobs=indobs)
        

# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    from main_utilities import GlobFile

    # USER configuration path
    print()
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    print('-> Configuration of environment')
    print('Where is your configuration file?')
    config_file_path = input()
    print()

    # CONFIG_FILE reading and defining global parameters
    global_params = GlobFile(config_file_path)  # To access any param.: global_params.parameter_name

    launch_adapt(global_params, 'no')
