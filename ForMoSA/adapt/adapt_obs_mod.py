from __future__ import print_function, division
import numpy as np
import os
import xarray as xr
# import matplotlib.pyplot as plt

from adapt.extraction_functions import extract_observation
from adapt.adapt_grid import adapt_grid

# ----------------------------------------------------------------------------------------------------------------------


def launch_adapt(global_params, justobs='no'):
    """
    Extract and adapt (wavelength resampling, resolution decreasing, continuum subtracting) the data and the synthetic
    spectra from a model grid.

    Args:
        global_params: Class containing each parameter
        justobs: If the grid need to be adapted justobs='no'
    Returns:

    Author: Simon Petrus
    """

    # Get back information from the config file
    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine="netcdf4")
    wav_mod_nativ = ds["wavelength"].values
    attr = ds.attrs
    ds.close()

    # Extract the data from the observation file
    obs_cut, obs_pho, obs_cut_ins, obs_pho_ins = extract_observation(global_params, wav_mod_nativ, attr['res'])

    # Estimate and subtraction of the continuum (if needed)
    if global_params.continuum_sub != 'NA':
        obs_cut_c, obs_pho_c, obs_cut_ins_c, obs_pho_ins_c = extract_observation(global_params, wav_mod_nativ,
                                                                                 attr['res'], 'yes')
        for c, cut in enumerate(obs_cut):
            obs_cut[c][1] -= obs_cut_c[c][1]
    # Merging of each sub-spectrum
    for c, cut in enumerate(obs_cut):
        if c == 0:
            wav_obs_extract = obs_cut[c][0]
            flx_obs_extract = obs_cut[c][1]
            err_obs_extract = obs_cut[c][2]
            res_obs_extract = obs_cut[c][3]
            ins_obs_extract = obs_cut_ins[c]

        else:
            wav_obs_extract = np.concatenate((wav_obs_extract, obs_cut[c][0]))
            flx_obs_extract = np.concatenate((flx_obs_extract, obs_cut[c][1]))
            err_obs_extract = np.concatenate((err_obs_extract, obs_cut[c][2]))
            res_obs_extract = np.concatenate((res_obs_extract, obs_cut[c][3]))
            ins_obs_extract = np.concatenate((ins_obs_extract, obs_cut_ins[c]))

    obs_merge = [wav_obs_extract, flx_obs_extract, err_obs_extract, res_obs_extract]

    # Save the new data spectrum
    np.savez(global_params.result_path + '/spectrum_obs',
             obs_merge=obs_merge,
             obs_cut=obs_cut,
             obs_cut_ins=obs_cut_ins,
             obs_merge_ins=ins_obs_extract,
             obs_pho=obs_pho,
             obs_pho_ins=obs_pho_ins)

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


if __name__ == '__main__':
    from master_main_utilities import GlobFile

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
