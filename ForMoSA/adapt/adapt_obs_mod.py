from __future__ import print_function, division
import numpy as np
import os
import glob
import xarray as xr
from scipy.interpolate import interp1d

from ForMoSA.adapt.adapt_grid import adapt_grid
from ForMoSA.adapt.adapt_extraction_functions import adapt_observation

# ----------------------------------------------------------------------------------------------------------------------


def launch_adapt(global_params, adapt_model=True):
    """
    Adapt the synthetic spectra of a grid to make them comparable with the data.
    
    Args:
        global_params  (object): Class containing each parameter
        adapt_model      (bool): True by default to also adapt the grid
    Returns:
        None

    Author: Simon Petrus, Matthieu Ravet, Paulina Palma-Bifani and Allan Denis
    """

    # Get back the grid information from the config file
    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine="netcdf4")
    wav_mod_nativ = ds["wavelength"].values
    attr = ds.attrs
    res_mod_nativ = attr['res']

    # Check if the grid is Nyquist-sampled, else set the resolution to R = wav / 2 Deltawav to make sure we not are adding any info
    dwav = np.abs(wav_mod_nativ - np.roll(wav_mod_nativ, 1))
    dwav[0] = dwav[1]
    res_Nyquist = wav_mod_nativ / (2 * dwav)
    res_mod_nativ[(res_mod_nativ > res_Nyquist)] = res_Nyquist[(res_mod_nativ > res_Nyquist)]


    # Extract the data from the observation files
    main_obs_path = global_params.main_observation_path

    for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
        
        global_params.observation_path = obs
        obs_name = os.path.splitext(os.path.basename(global_params.observation_path))[0]

        obs_dict = adapt_observation(global_params, wav_mod_nativ, res_mod_nativ, obs_name, indobs=indobs)

        # Check-ups and warnings for negative values in the diagonal of the covariance matrix
        if len(obs_dict['wav_spectro']) != 0 and any(np.diag(obs_dict['cov']) < 0):
            print()
            print("WARNING: Negative value(s) is(are) present on the diagonal of the covariance matrix.") 
            print("Operation aborted.")
            print()
            exit()
            
        # Save the data
        np.savez(os.path.join(global_params.result_path, f'spectrum_obs_{obs_name}.npz'), **obs_dict)
        

        # - - - - - - - - 


        # Adaptation of the model grid
        if adapt_model == True:
            # Creation of the repertory to store the adapted grid (if needed)
            if os.path.isdir(global_params.adapt_store_path):
                pass
            else:
                os.mkdir(global_params.adapt_store_path)
            if len(obs_dict['wav_spectro']) != 0:
                # Setup target wavelength and resolution for the observation and the model
                if global_params.target_res_mod[indobs % len(global_params.target_res_mod)] == 'mod': # Kepping the model's resolution
                    target_wav_mod = wav_mod_nativ
                    target_res_mod = res_mod_nativ
                elif global_params.target_res_mod[indobs % len(global_params.target_res_mod)] == 'obs': # Using the observation's resolution except where its higher than the model's
                    target_wav_mod = obs_dict['wav_spectro']
                    target_res_mod = obs_dict['res_spectro']
                else:                                             # Using a custom resolution except where its higher than the model's
                    res_custom = np.full(len(wav_mod_nativ), float(global_params.target_res_mod[indobs]))
                    target_wav_mod = wav_mod_nativ
                    target_res_mod = np.min([res_mod_nativ, res_custom], axis=0)

                # Masks to have larger cuts of the spectroscopic grid if needed (if rv is defined)
                if global_params.rv[indobs*3 % len(global_params.rv)] == 'NA':
                    mask_mod_obs = (target_wav_mod <= obs_dict['wav_spectro'][-1]) & (target_wav_mod >= obs_dict['wav_spectro'][0]) 
                    target_wav_mod = target_wav_mod[mask_mod_obs]
                    target_res_mod = target_res_mod[mask_mod_obs]
                else:
                    mask_mod_obs = (target_wav_mod <= 1.01 * obs_dict['wav_spectro'][-1]) & (target_wav_mod >= 0.99 * obs_dict['wav_spectro'][0])   # 1.01 corresponds to a value of 3000 km/s for the RV so we do no risk to lose data on the edges when applying the RV correction
                    target_wav_mod = target_wav_mod[mask_mod_obs]
                    target_res_mod = target_res_mod[mask_mod_obs]

                # Interpolate the resolution of the model onto the wavelength of the data to properly decrease the resolution if necessary
                interp_mod_to_obs = interp1d(wav_mod_nativ, res_mod_nativ, fill_value='extrapolate')
                res_mod_nativ_interp = interp_mod_to_obs(target_wav_mod)

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

            adapt_grid(global_params, obs_dict, res_mod_nativ_interp, target_wav_mod, target_res_mod, obs_name, indobs)
        
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

    launch_adapt(global_params, True)
