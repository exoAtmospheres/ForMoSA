from __future__ import print_function, division
import numpy as np
import os,sys
import xarray as xr
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.abspath('../'))

from adapt.extraction_functions import extract_observation
from adapt.adapt_grid import adapt_grid
import glob
# ----------------------------------------------------------------------------------------------------------------------


def launch_adapt(global_params, justobs='no'):
    """
    Adapt the synthetic spectra of a grid to make them comparable with the data.
    
    Args:
        global_params  (object): Class containing each parameter
        justobs    ('yes'/'no'): 'no' by default to also adapt the grid
    Returns:
        None

    Author: Simon Petrus, Matthieu Ravet, Paulina Palma-Bifani and Allan Denis
    """

    # Get back the grid information from the config file
    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine="netcdf4")
    wav_mod_nativ = ds["wavelength"].values
    attr = ds.attrs
    res_mod_nativ = attr['res']
    ds.close()

    # Check if the grid is Nyquist-sampled, else set the resolution to R = wav / 2 Deltawav to make sure we are adding any info
    dwav = np.abs(wav_mod_nativ - np.roll(wav_mod_nativ, 1))
    dwav[0] = dwav[1]
    res_Nyquist = wav_mod_nativ / (2 * dwav)
    res_mod_nativ[(res_mod_nativ > res_Nyquist)] = res_Nyquist[(res_mod_nativ > res_Nyquist)]

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
            obs_spectro, obs_photo, obs_photo_ins, obs_opt, res_mod_obs  = extract_observation(global_params, wav_mod_nativ, res_mod_nativ, 'yes', 
                                                                                                indobs=indobs)
        else:
            obs_spectro, obs_photo, obs_photo_ins, obs_opt, res_mod_obs  = extract_observation(global_params, wav_mod_nativ, res_mod_nativ,
                                                                                                indobs=indobs)

        wav_obs_spectro, res_obs_spectro, wav_obs_photo = obs_spectro[0], obs_spectro[3], obs_photo[0]

        # - - - - - - - - 

        # Masks to cut the spectroscopic grid
        if len(obs_spectro[0]) != 0:
            # Extract larger windows but cuts the grid before extraction to speed-up the code
            # Redifine grid shape if needed
            if len(global_params.rv) > 3:
                if global_params.rv[indobs*3] == 'NA':
                    wav_grid_spectro = wav_obs_spectro
                else:
                    mask_mod_obs = (wav_mod_nativ <= 1.01 * obs_spectro[0][-1]) & (wav_mod_nativ >= 0.99 * obs_spectro[0][0])   # 1.01 corresponds to a value of 3000 km/s for the RV so we do no risk to lose data on the edges when applying the RV correction
                    wav_grid_spectro = wav_mod_nativ[mask_mod_obs]
            else:
                if global_params.rv == 'NA':
                    wav_grid_spectro = wav_obs_spectro
                else:
                    mask_mod_obs = (wav_mod_nativ <= 1.01 * obs_spectro[0][-1]) & (wav_mod_nativ >= 0.99 * obs_spectro[0][0])   # 1.01 corresponds to a value of 3000 km/s for the RV so we do no risk to lose data on the edges when applying the RV correction
                    wav_grid_spectro = wav_mod_nativ[mask_mod_obs]  
        else:
            wav_grid_spectro = wav_obs_spectro

        wav_grid_photo = wav_obs_photo


        # Check-ups and warnings for negative values in the diagonal of the covariance matrix
        if len(obs_opt[0]) != 0 and any(np.diag(obs_opt[0]) < 0):
            print()
            print("WARNING: Negative value(s) is(are) present on the diagonal of the covariance matrix.") 
            print("Operation aborted.")
            print()
            exit()
            
        # Save the new data spectrum
        np.savez(os.path.join(global_params.result_path, f'spectrum_obs_{obs_name}.npz'),
                    obs_spectro=obs_spectro,
                    obs_photo=obs_photo,
                    obs_photo_ins=obs_photo_ins,
                    obs_opt=obs_opt) # Optional arrays kept separatly
        
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

            adapt_grid(global_params, wav_grid_spectro, wav_grid_photo, res_mod_obs, wav_obs_spectro, res_obs_spectro, obs_photo_ins, obs_name, indobs)
        
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
