from __future__ import print_function, division
import numpy as np
import os
import xarray as xr
# import matplotlib.pyplot as plt

from adapt.extraction_functions import extract_observation
from adapt.adapt_grid import adapt_grid
from main_utilities import yesno, diag_mat
import glob
# ----------------------------------------------------------------------------------------------------------------------


def launch_adapt(global_params, justobs='no'):
    """
    Adapt the synthetic spectra of a grid to make them comparable with the data.
    
    Args:
        global_params (object): Class containing each parameter
        wav_obs_spec   (array): Merged wavelength grid of the data
        wav_obs_phot   (array): Wavelengths of the photometry points
        obs_name         (str): Name of the current observation looping (only relevant in MOSAIC, else set to '')
        indobs           (int): Index of the current observation looping (only relevant in MOSAIC, else set to 0)
    Returns:
        None

    Author: Simon Petrus / Adapted: Matthieu Ravet & Paulina Palma-Bifani
    """

    # Get back information from the config file
    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine="netcdf4")
    wav_mod_nativ = ds["wavelength"].values
    attr = ds.attrs
    ds.close()

    # Extract the data from the observation file
    obs_cut, obs_pho, obs_cut_ins, obs_pho_ins, obs_cut_cov = extract_observation(global_params, wav_mod_nativ, attr['res'])

    # Estimate and subtraction of the continuum (if needed)
    if global_params.continuum_sub != 'NA':
        obs_cut_c, obs_pho_c, obs_cut_ins_c, obs_pho_ins_c, obs_cut_cov = extract_observation(global_params, wav_mod_nativ,
                                                                                 attr['res'], 'yes')
        for c, cut in enumerate(obs_cut):
            obs_cut[c][1] -= obs_cut_c[c][1]
    # Merging of each sub-spectrum
    for c, cut in enumerate(obs_cut):
        if c == 0:
            wav_obs_extract = obs_cut[c][0]
            flx_obs_extract = obs_cut[c][1]
            err_obs_extract = obs_cut[c][2]
            res_obs_extract = np.array(obs_cut[c][3], dtype=float) # New addition (may need to be corrected)
            transm_obs_extract = obs_cut[c][4]
            star_flx_obs_extract = obs_cut[c][5]
            ins_obs_extract = obs_cut_ins[c]
            if len(obs_cut_cov) != 0: # Extract sub-covariance (if necessary)
                cov_obs_extract = obs_cut_cov[c]
            else:
                cov_obs_extract = []

        else:
            wav_obs_extract = np.concatenate((wav_obs_extract, obs_cut[c][0]))
            flx_obs_extract = np.concatenate((flx_obs_extract, obs_cut[c][1]))
            err_obs_extract = np.concatenate((err_obs_extract, obs_cut[c][2]))
            res_obs_extract = np.concatenate((res_obs_extract, np.array(obs_cut[c][3], dtype=float))) # New addition (may need to be corrected)
            transm_obs_extract = np.concatenate((transm_obs_extract, obs_cut[c][4]))
            star_flx_obs_extract = np.concatenate((star_flx_obs_extract, obs_cut[c][5]))
            ins_obs_extract = np.concatenate((ins_obs_extract, obs_cut_ins[c]))
            if len(cov_obs_extract) != 0:
                cov_obs_extract = diag_mat([cov_obs_extract, obs_cut_cov[c]])

        # Compute the inverse of the merged covariance matrix (note: inv(C1, C2) = (in(C1), in(C2)) if C1 and C2 are block matrix on the diagonal)
        # if necessary
        if len(cov_obs_extract) != 0:
            inv_cov_obs = np.linalg.inv(cov_obs_extract)
        else:
            inv_cov_obs = []

        # Compile everything
        obs_merge = [wav_obs_extract, flx_obs_extract, err_obs_extract, res_obs_extract, transm_obs_extract, star_flx_obs_extract]

        # Check-ups and warnings for negative values in the diagonal of the covariance matrix
        if len(cov_obs_extract) != 0 and any(np.diag(cov_obs_extract) < 0):
            print()
            y_n_par = yesno("WARNING: Negative value(s) is(are) present on the diagonal of the covariance matrix. Do you still want to run the inversion? (y/n)") 
        else:
            y_n_par = 'y' 
        if y_n_par != 'y':
            print("Operation aborted.")
            print()
            exit()
        else:
            print()
            print("Continuing...")

    # Save the new data spectrum
    np.savez(global_params.result_path + '/spectrum_obs',
             obs_merge=obs_merge,
             obs_cut=obs_cut,
             obs_cut_ins=obs_cut_ins,
             obs_merge_ins=ins_obs_extract,
             obs_pho=obs_pho,
             obs_pho_ins=obs_pho_ins,
             inv_cov_obs=inv_cov_obs)

    # Adaptation of the model grid
    if justobs == 'no':
        # Creation of the repertory to store the adapted grid (if needed)
        if os.path.isdir(global_params.adapt_store_path):
            pass
        else:
            os.mkdir(global_params.adapt_store_path)

        print()
        print()
        print("-> To compare synthetic spectra with the observation we need to manage them. The following actions are performed:")
        print("- extraction -")
        print("- resizing on the observation's wavelength range -")
        print("- adjustement of the spectral resolution -")
        print("- substraction of the continuum (if needed) -")
        print()

        adapt_grid(global_params, obs_merge[0], obs_pho[0])


# ----------------------------------------------------------------------------------------------------------------------


def launch_adapt_MOSAIC(global_params, justobs='no'):
    """
    Adapt the synthetic spectra of a grid to make them comparable with the data.
    
    Args:
        global_params (object): Class containing each parameter
        wav_obs_spec   (array): Merged wavelength grid of the data
        wav_obs_phot   (array): Wavelengths of the photometry points
        obs_name         (str): Name of the current observation looping (only relevant in MOSAIC, else set to '')
        indobs           (int): Index of the current observation looping (only relevant in MOSAIC, else set to 0)
    Returns:
        None

    Author: Simon Petrus / Adapted: Matthieu Ravet & Paulina Palma-Bifani
    """

    # Get back information from the config file
    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine="netcdf4")
    wav_mod_nativ = ds["wavelength"].values
    attr = ds.attrs
    ds.close()

    # Extract the data from the observation files
    main_obs_path = global_params.main_observation_path

    print()
    print()
    print()
    print()
    print("         > Starting MOSAIC <             ")
    print()
    print()
    print()
    print()


    for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
        
        global_params.observation_path = obs
        obs_name = os.path.splitext(os.path.basename(global_params.observation_path))[0]
        
        obs_cut, obs_pho, obs_cut_ins, obs_pho_ins, obs_cut_cov = extract_observation(global_params, wav_mod_nativ, attr['res'], obs_name=obs_name,
                                                                                      indobs=indobs)

        # Estimate and subtraction of the continuum (if needed) + check-ups
        if global_params.continuum_sub[indobs] != 'NA':
            print()
            print(obs_name + ' will have a R=' + global_params.continuum_sub[indobs] + ' continuum removed using a ' 
                  + global_params.wav_for_continuum[indobs] + ' wavelength range')
            print()
            y_n_par = yesno('Is this what you want ? (y/n)')
            if y_n_par == 'n':
                print('Please input the desired spectral resolution (or NA if you do not want to remove the continuum):')
                global_params.continuum_sub[indobs] = input()
            print()
            print()
            print()
        if global_params.continuum_sub[indobs] != 'NA':
            obs_cut_c, obs_pho_c, obs_cut_ins_c, obs_pho_ins_c, obs_cut_cov = extract_observation(global_params, wav_mod_nativ,
                                                                                    attr['res'], 'yes', obs_name=obs_name, indobs=indobs)
            for c, cut in enumerate(obs_cut):
                obs_cut[c][1] -= obs_cut_c[c][1]
        # Merging of each sub-spectrum
        for c, cut in enumerate(obs_cut):
            if c == 0:
                wav_obs_extract = obs_cut[c][0]
                flx_obs_extract = obs_cut[c][1]
                err_obs_extract = obs_cut[c][2]
                res_obs_extract = np.array(obs_cut[c][3], dtype=float) # New addition (may need to be corrected)
                transm_obs_extract = obs_cut[c][4]
                star_flx_obs_extract = obs_cut[c][5]
                ins_obs_extract = obs_cut_ins[c]
                if len(obs_cut_cov) != 0: # Extract sub-covariance (if necessary)
                    cov_obs_extract = obs_cut_cov[c]
                else:
                    cov_obs_extract = []

            else:
                wav_obs_extract = np.concatenate((wav_obs_extract, obs_cut[c][0]))
                flx_obs_extract = np.concatenate((flx_obs_extract, obs_cut[c][1]))
                err_obs_extract = np.concatenate((err_obs_extract, obs_cut[c][2]))
                res_obs_extract = np.concatenate((res_obs_extract, np.array(obs_cut[c][3], dtype=float))) # New addition (may need to be corrected)
                transm_obs_extract = np.concatenate((transm_obs_extract, obs_cut[c][4]))
                star_flx_obs_extract = np.concatenate((star_flx_obs_extract, obs_cut[c][5]))
                ins_obs_extract = np.concatenate((ins_obs_extract, obs_cut_ins[c]))
                if len(cov_obs_extract) != 0:
                    cov_obs_extract = diag_mat([cov_obs_extract, obs_cut_cov[c]])


            # Compute the inverse of the merged covariance matrix (note: inv(C1, C2) = (in(C1), in(C2)) if C1 and C2 are block matrix on the diagonal)
            # if necessary
            if len(cov_obs_extract) != 0:
                inv_cov_obs = np.linalg.inv(cov_obs_extract)
            else:
                inv_cov_obs = []

            # Compile everything
            obs_merge = [wav_obs_extract, flx_obs_extract, err_obs_extract, res_obs_extract, transm_obs_extract, star_flx_obs_extract]

            # Check-ups and warnings for negative values in the diagonal of the covariance matrix
            if len(cov_obs_extract) != 0 and any(np.diag(cov_obs_extract) < 0):
                print()
                y_n_par = yesno("WARNING: Negative value(s) is(are) present on the diagonal of the covariance matrix. Do you still want to run the inversion? (y/n)") 
            else:
                y_n_par = 'y' 
            if y_n_par != 'y':
                print("Operation aborted.")
                print()
                exit()
            else:
                print()
                print("Continuing...")

        # Save the new data spectrum
        np.savez(os.path.join(global_params.result_path, f'spectrum_obs_{obs_name}.npz'),
                obs_merge=obs_merge,
                obs_cut=obs_cut,
                obs_cut_ins=obs_cut_ins,
                obs_merge_ins=ins_obs_extract,
                obs_pho=obs_pho,
                obs_pho_ins=obs_pho_ins,
                inv_cov_obs=inv_cov_obs)
        
        # Adaptation of the model grid
        if justobs == 'no':
            # Creation of the repertory to store the adapted grid (if needed)
            if os.path.isdir(global_params.adapt_store_path):
                pass
            else:
                os.mkdir(global_params.adapt_store_path)

            print()
            print()
            print("-> To compare synthetic spectra with the observation we need to manage them. The following actions are performed:")
            print("- extraction -")
            print("- resizing on the observation's wavelength range -")
            print("- adjustement of the spectral resolution -")
            print("- substraction of the continuum (if needed) -")
            print()
            print(f"-> Sarting adaptation of {obs_name}")

            adapt_grid(global_params, obs_merge[0], obs_pho[0], obs_name=obs_name, indobs=indobs)





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
