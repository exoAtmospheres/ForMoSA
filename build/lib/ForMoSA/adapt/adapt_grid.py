from __future__ import print_function, division
import numpy as np
import xarray as xr
import os
import ctypes
import multiprocessing as mp
from scipy.interpolate import interp1d

from tqdm import tqdm
from multiprocessing.pool import ThreadPool

from ForMoSA.utils import format_grid
from ForMoSA.adapt.adapt_extraction_functions import adapt_model


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

def tpool_adapt(idx, global_params, obs_dict, wav_mod_nativ, res_mod_nativ, target_wav_mod, target_res_mod, indobs, keys, titles, values):
    '''
    Worker function for the parallelisation process of adapt_model()

    Args:
        idx               (tuple): Index of the current model
        global_params    (object): Class containing each parameter 
        wav_mod_nativ     (array): Native wavelength array of the grid
        res_mod_nativ       (array): Native resolution of the grid (might have changed if Nyquist is not respected or resample at wav_obs_spectro)
        target_wav_mod    (array): Targeted wavelength grid of the final grid
        target_res_mod    (array): Targeted spectral resolution of the final grid
        indobs              (int): Index of the current observation looping
        keys               (list): Attribute keys
        titles             (list): Attribute titles
        values             (dict): Values for each attribute
    Returns:
        None

    Author: Arthur Vigan
    '''
    # global variables
    try:
        global grid_input_shape, grid_input_data, grid_spectro_shape, grid_spectro_data, grid_photo_shape, grid_photo_data
        grid_input   = array_to_numpy(grid_input_data, grid_input_shape, float)
        grid_spectro = array_to_numpy(grid_spectro_data, grid_spectro_shape, float)
        grid_photo   = array_to_numpy(grid_photo_data, grid_photo_shape, float)
        
        model_to_adapt = grid_input[(..., ) + idx]
        nan_mod = np.isnan(model_to_adapt)
        msg = ''

        if np.any(nan_mod):
            msg = 'Extraction of model failed : '
            for i, (key, title) in enumerate(zip(keys, titles)):
                msg += f'{title}={values[key][idx[i]]}, '
            print(msg)
        else:
            mod_spectro, mod_photo = adapt_model(global_params, obs_dict, wav_mod_nativ, model_to_adapt, res_mod_nativ, target_wav_mod, target_res_mod, indobs)
            grid_spectro[(..., ) + idx] = mod_spectro
            grid_photo[(..., ) + idx]   = mod_photo
        
    except Exception as e:
        print(f'Error in task: {e}')


def adapt_grid(global_params, obs_dict, res_mod_nativ, target_wav_mod, target_res_mod, obs_name, indobs=0):
    """
    Adapt the synthetic spectra of a grid to make them comparable with the data.

    Args:
        global_params      (object): Class containing each parameter
        obs_dict             (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)
        res_mod_nativ       (array): Native resolution of the grid (might have changed if Nyquist is not respected or resample at wav_obs_spectro)
        target_wav_mod      (array): Targeted wavelength grid of the final grid
        target_res_mod      (array): Targeted spectral resolution of the final grid
        obs_name              (str): Name of the current observation looping
        indobs                (int): Index of the current observation looping
    Returns:
        None

    Author: Simon Petrus, Matthieu Ravet, Paulina Palma-Bifani, Arthur Vigan and Allan Denis
    """

    # Open raw grid
    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine="netcdf4")
    wav_mod_nativ = ds["wavelength"].values
    grid = ds['grid']
    attr = ds.attrs
    grid_np = grid.to_numpy()
    attr = ds.attrs
    ds.close()

    # Create arrays without any assumptions on the number of parameters
    shape_spectro = [len(target_wav_mod)]
    shape_photo = [len(obs_dict['wav_photo'])]
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

    try: # Parallel if possible
        ncpu = mp.cpu_count()
        with ThreadPool(processes=ncpu, initializer=tpool_adapt_init, initargs=(grid_input_shape, grid_input_data, grid_spectro_shape, grid_spectro_data, grid_photo_shape, grid_photo_data)) as pool:
            for idx in np.ndindex(shape):
                pool.apply_async(tpool_adapt, args=(idx, global_params, obs_dict, wav_mod_nativ, res_mod_nativ, target_wav_mod, target_res_mod, indobs, attr['key'], attr['title'], values), callback=update)

            pool.close()
            pool.join()
    except:
        tpool_adapt_init(grid_input_shape, grid_input_data, grid_spectro_shape, grid_spectro_data, grid_photo_shape, grid_photo_data)

        for idx in np.ndindex(shape):
            tpool_adapt(idx, global_params, obs_dict, wav_mod_nativ, res_mod_nativ, target_wav_mod, target_res_mod, indobs, attr['key'], attr['title'], values)
            update()

    # create final datasets
    vars = ["wavelength"]
    for key in attr['key']:
        vars.append(key)

    # Save new attributes and coords
    attr['res'] = target_res_mod
    coords_spectro = {"wavelength": target_wav_mod}
    coords_photo   = {"wavelength": obs_dict['wav_photo']}

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
    
    interp_kwargs = {
    "method": global_params.method,
    "fill_value": "extrapolate",
    "limit": None,
    "max_gap": None,
    }
    
    for idx, (key, title) in enumerate(zip(attr['key'], attr['title'])):
        print(f'{idx+1}/{nkey} - {title}')
        if ds_spectro_new.isnull().any(dim=key):  # Check is there is any nan in the grid
            ds_spectro_new = ds_spectro_new.interpolate_na(dim=key, **interp_kwargs)
        if ds_photo_new.isnull().any(dim=key):
            ds_photo_new = ds_photo_new.interpolate_na(dim=key, **interp_kwargs)


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Chose the method of adaptation of the grid. It can either be :
    # "NA" = we keep the spectra xarray grid and interpolate it during the inversion
    # "PCA" = we use PCA to decompose the grid into eigenspectra and weight and keep the weigths grid and interpolate it during the inversion
    # "NMF" = we use NMF to decompose the grid into H (~eigenspectra) W (~weights) and keep the weigths grid and interpolate it during the inversion
    if global_params.emulator[0] != 'NA':
        from ForMoSA.adapt.adapt_emulators import emulator_PCA, emulator_NMF

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

        if global_params.emulator[0] == 'PCA':
            print()
            print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
            print('-> Decomposing the grid using PCA... ')
            print()

            # PCA components
            ncomp = int(global_params.emulator[1])

            # Check if the grid is not empty and compute PCA 
            if len(target_wav_mod) != 0:
                flx_grid_mean_spectro, flx_grid_std_spectro, vectors_spectro, weights_spectro = emulator_PCA(ds_spectro_new, ncomp)
            else:
                flx_grid_mean_spectro, flx_grid_std_spectro, vectors_spectro, weights_spectro = np.asarray([]), np.asarray([]), np.empty((0,0), dtype=object), np.empty((0,) + grid.shape[1:], dtype=object)
            if len(obs_dict['wav_photo']) != 0:
                flx_grid_mean_photo, flx_grid_std_photo, vectors_photo, weights_photo = emulator_PCA(ds_photo_new, ncomp)
            else:
                flx_grid_mean_photo, flx_grid_std_photo, vectors_photo, weights_photo = np.asarray([]), np.asarray([]), np.empty((0,0), dtype=object), np.empty((0,) + grid.shape[1:], dtype=object)
            
            # Save the new grids and PCA outputs
            mod_dict = {'wav_spectro': target_wav_mod,
                       'res_spectro': target_res_mod,
                       'flx_mean_spectro': flx_grid_mean_spectro,
                       'flx_std_spectro': flx_grid_std_spectro,
                       'vectors_spectro': vectors_spectro,
                       'flx_mean_photo': flx_grid_mean_photo,
                       'flx_std_photo': flx_grid_std_photo,
                       'vectors_photo': vectors_photo}
            np.savez(os.path.join(global_params.result_path, f'PCA_mod_{obs_name}.npz'), **mod_dict)
            
            # Format the new grids in xarray
            attr.pop('res') # You don't need the res array anymore
            ds_spectro_new = format_grid(grid, attr, weights_spectro.shape[0], weights_spectro)
            ds_photo_new = format_grid(grid, attr, weights_photo.shape[0], weights_photo)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

        elif global_params.emulator[0] == 'NMF':
            print()
            print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
            print('-> Decomposing the grid using NMF... ')
            print()

            #NMF components
            ncomp = int(global_params.emulator[1])

            # Check if the grid is not empty and compute PCA 
            if len(target_wav_mod) != 0:
                vectors_spectro, weights_spectro = emulator_NMF(ds_spectro_new, ncomp)
            else:
                vectors_spectro, weights_spectro = np.empty((0,0), dtype=object), np.empty((0,) + grid.shape[1:], dtype=object)
            if len(obs_dict['wav_photo']) != 0:
                vectors_photo, weights_photo = emulator_NMF(ds_photo_new, ncomp)
            else:
                vectors_photo, weights_photo = np.empty((0,0), dtype=object), np.empty((0,) + grid.shape[1:], dtype=object)
            
            # Save the new grids and NMF outputs
            mod_dict = {'wav_spectro': target_wav_mod,
                       'res_spectro': target_res_mod,
                       'vectors_spectro': vectors_spectro,
                       'vectors_photo': vectors_photo}
            np.savez(os.path.join(global_params.result_path, f'NMF_mod_{obs_name}.npz'), **mod_dict)
            
            # Format the new grids in xarray
            attr.pop('res') # You don't need the res array anymore
            ds_spectro_new = format_grid(grid, attr, weights_spectro.shape[0], weights_spectro)
            ds_photo_new = format_grid(grid, attr, weights_photo.shape[0], weights_photo)
    else:
        pass

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
