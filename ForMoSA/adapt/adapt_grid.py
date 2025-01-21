from __future__ import print_function, division
import numpy as np
import xarray as xr
import os, sys
import ctypes
import multiprocessing as mp

from tqdm import tqdm
from multiprocessing.pool import ThreadPool

sys.path.insert(0, os.path.abspath('../'))

from adapt.extraction_functions import adapt_model


# ----------------------------------------------------------------------------------------------------------------------
def array_to_numpy(shared_array, shape, dtype):
    '''
    Return a numpy array from a shared array

    Args:
        shared_array     (mp.RawArray): Raw shared array
        shape                  (tuple): Shape of the array
        dtype            (numpy dtype): Data type of the array
    Returns
        numpy_array       (np.ndarray): Numpy array mapped to shared array

    Author: Arthur Vigan
    '''
    if shared_array is None:
        return None

    numpy_array = np.frombuffer(shared_array, dtype=dtype)
    if shape is not None:
        numpy_array.shape = shape

    return numpy_array


def tpool_adapt_init(grid_input_shape_i, grid_input_data_i, grid_spectro_shape_i, grid_spectro_data_i, grid_photo_shape_i, grid_photo_data_i):
    '''
    Thread pool init function for the parallelisation process of adapt_model()

    This function initializes the global variables stored as shared arrays
    '''

    # global variables
    global grid_input_shape, grid_input_data, grid_spectro_shape, grid_spectro_data, grid_photo_shape, grid_photo_data

    grid_input_shape   = grid_input_shape_i
    grid_input_data    = grid_input_data_i
    grid_spectro_shape = grid_spectro_shape_i
    grid_spectro_data  = grid_spectro_data_i
    grid_photo_shape   = grid_photo_shape_i
    grid_photo_data    = grid_photo_data_i

# global_params, wav_mod_nativ, flx_mod_nativ, res_mod_obs, wav_obs_spectro, res_obs_spectro, obs_photo_ins

def tpool_adapt(idx, global_params, wav_mod_nativ, res_mod_obs, wav_obs_spectro, res_obs_spectro, obs_photo_ins, obs_name, indobs, keys, titles, values):
    '''
    Worker function for the parallelisation process of adapt_model()

    Args:
        idx               (tuple): Index of the current model
        global_params    (object): Class containing each parameter 
        wav_mod_nativ     (array): Wavelength of the input models
        res_mod_obs       (array): Spectral resolution of the model interpolated at wav_obs_spectro
        wav_obs_spectro   (array): Merged wavelength array of the data
        res_obs_spectro   (array): Merged resolution array of the data
        obs_photo_ins     (array): List containing different filters used for the data (1 per photometric point). [filter_phot_1, filter_phot_2, ..., filter_phot_n]
        obs_name            (str): Name of the current observation looping
        indobs              (int): Index of the current observation looping
        keys               (list): Attribute keys
        titles             (list): Attribute titles
        values             (dict): Values for each attribute
    Returns:
        None

    Author: Arthur Vigan
    '''

    # global variables
    global grid_input_shape, grid_input_data, grid_spectro_shape, grid_spectro_data, grid_photo_shape, grid_photo_data

    grid_input   = array_to_numpy(grid_input_data, grid_input_shape, float)
    grid_spectro = array_to_numpy(grid_spectro_data, grid_spectro_shape, float)
    grid_photo   = array_to_numpy(grid_photo_data, grid_photo_shape, float)

    model_to_adapt = grid_input[(..., ) + idx]
    nan_mod = np.isnan(model_to_adapt)

    if np.any(nan_mod):
        msg = 'Extraction of model failed : '
        for i, (key, title) in enumerate(zip(keys, titles)):
            msg += f'{title}={values[key][idx[i]]}, '
        print(msg)
    else:
        mod_spectro, mod_photo = adapt_model(global_params, wav_mod_nativ, model_to_adapt, res_mod_obs, wav_obs_spectro, res_obs_spectro, obs_photo_ins, obs_name=obs_name, indobs=indobs)
        grid_spectro[(..., ) + idx] = mod_spectro
        grid_photo[(..., ) + idx]   = mod_photo


def adapt_grid(global_params, res_mod_obs, wav_obs_spectro, res_obs_spectro, wav_obs_photo, obs_photo_ins, obs_name='', indobs=0):
    """
    Adapt the synthetic spectra of a grid to make them comparable with the data.

    Args:
        global_params    (object): Class containing each parameter
        res_mod_obs       (array): Spectral resolution of the model interpolated at wav_obs_spectro 
        wav_obs_spectro   (array): Merged wavelength array of the data
        res_obs_spectro   (array): Merged resolution array of the data
        wav_obs_photo     (array): Wavelengths of the photometry points
        obs_photo_ins     (array): List containing different filters used for the data (1 per photometric point). [filter_phot_1, filter_phot_2, ..., filter_phot_n]
        obs_name            (str): Name of the current observation looping
        indobs              (int): Index of the current observation looping
        parallel           (bool): Specify if parallelisation is used for adaptation
    Returns:
        None

    Author: Simon Petrus, Matthieu Ravet, Paulina Palma-Bifani and Arthur Vigan
    """

    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine="netcdf4")
    wav_mod_nativ = ds["wavelength"].values
    grid = ds['grid']
    attr = ds.attrs
    grid_np = grid.to_numpy()

    # create arrays without any assumptions on the number of parameters
    shape_spectro = [len(wav_obs_spectro)]
    shape_photo = [len(wav_obs_photo)]
    values = {}
    for key in attr['key']:
        shape_spectro.append(len(grid[key].values))
        shape_photo.append(len(grid[key].values))
        values[key] = grid[key].values

    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

    #
    # Shared arrays of image intensities at all wavelengths
    #

    grid_input_shape   = grid_np.shape
    grid_input_data    = mp.RawArray(ctypes.c_double, int(np.prod(grid_input_shape)))
    grid_input_np      = array_to_numpy(grid_input_data, grid_input_shape, float)
    grid_input_np[:]   = grid_np
    del grid_np

    grid_spectro_shape = shape_spectro
    grid_spectro_data  = mp.RawArray(ctypes.c_double, int(np.prod(grid_spectro_shape)))
    grid_spectro_np    = array_to_numpy(grid_spectro_data, grid_spectro_shape, float)
    grid_spectro_np[:] = np.nan

    grid_photo_shape   = shape_photo
    grid_photo_data    = mp.RawArray(ctypes.c_double, int(np.prod(grid_photo_shape)))
    grid_photo_np      = array_to_numpy(grid_photo_data, grid_photo_shape, float)
    grid_photo_np[:]   = np.nan

    #
    # parallel grid adaptation
    #
    shape = grid_input_shape[1:]
    pbar = tqdm(total=np.prod(shape), leave=False)

    def update(*a):
        pbar.update()

    if global_params.parallel:
        ncpu = mp.cpu_count()
        with ThreadPool(processes=ncpu, initializer=tpool_adapt_init, initargs=(grid_input_shape, grid_input_data, grid_spectro_shape, grid_spectro_data, grid_photo_shape, grid_photo_data)) as pool:
            for idx in np.ndindex(shape):
                pool.apply_async(tpool_adapt, args=(idx, global_params, wav_mod_nativ, res_mod_obs, wav_obs_spectro, res_obs_spectro, obs_photo_ins, obs_name, indobs, attr['key'], attr['title'], values), callback=update)

            pool.close()
            pool.join()
    else:
        tpool_adapt_init(grid_input_shape, grid_input_data, grid_spectro_shape, grid_spectro_data, grid_photo_shape, grid_photo_data)

        for idx in np.ndindex(shape):
            tpool_adapt(idx, global_params, wav_mod_nativ, res_mod_obs, wav_obs_spectro, res_obs_spectro, obs_photo_ins, obs_name, indobs, attr['key'], attr['title'], values)
            update()

    # create final datasets
    vars = ["wavelength"]
    for key in attr['key']:
        vars.append(key)

    coords_spectro = {"wavelength": wav_obs_spectro}
    coords_photo   = {"wavelength": wav_obs_photo}
    for key in attr['key']:
        coords_spectro[key] = grid[key].values
        coords_photo[key]   = grid[key].values

    ds_spectro_new = xr.Dataset(data_vars=dict(grid=(vars, grid_spectro_np)), coords=coords_spectro, attrs=attr)
    ds_photo_new   = xr.Dataset(data_vars=dict(grid=(vars, grid_photo_np)), coords=coords_photo, attrs=attr)

    print()
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    print('-> The possible holes in the grid are interpolated: ')
    print()
    nkey = len(attr['key'])
    for idx, (key, title) in enumerate(zip(attr['key'], attr['title'])):
        print(f'{idx+1}/{nkey} - {title}')
        ds_spectro_new = ds_spectro_new.interpolate_na(dim=key, method="linear", fill_value="extrapolate", limit=None, max_gap=None)
        ds_photo_new = ds_photo_new.interpolate_na(dim=key, method="linear", fill_value="extrapolate", limit=None, max_gap=None)

    ds_spectro_new.to_netcdf(os.path.join(global_params.adapt_store_path, f'adapted_grid_spectro_{global_params.grid_name}_{obs_name}_nonan.nc'),
                             format='NETCDF4',
                             engine='netcdf4',
                             mode='w')
    ds_photo_new.to_netcdf(os.path.join(global_params.adapt_store_path, f'adapted_grid_photo_{global_params.grid_name}_{obs_name}_nonan.nc'),
                           format='NETCDF4',
                           engine='netcdf4',
                           mode='w')

    print('The possible holes have been interpolated!')

    return None
