import numpy as np
import xarray as xr
import os 
import glob

# ----------------------------------------------------------------------------------------------------------------------


def yesno(text):
    '''
    Function to interact with the terminal and decide for different options when running ForMoSA (Loop to repeat question if answer is different to 'y' or 'n).

    Args:
        text    (str): (y/n) answer in the terminall in interactive mode
    Returns:
        asw     (str): answer y or n

    Author: Simon Petrus
    '''
    print(text)
    asw = input()
    if asw in ['y', 'n']:
        return asw
    else:
        return yesno()

# ----------------------------------------------------------------------------------------------------------------------


def decoupe(second):
    """
    Re-arranged a number of seconds in the hours-minutes-seconds format.

    Args:
        second (float): number of second
    Returns:
        - float     : hours
        - float     : minutes
        - float     : seconds

    Author: Simon Petrus
    """

    hour = second / 3600
    second %= 3600
    minute = second / 60
    second %= 60

    return hour, minute, second


# ----------------------------------------------------------------------------------------------------------------------


def find_nearest(array, value):
    '''
    Return the indice of the closest values from a desire value in an array.

    Parameters
    ----------
    array (array): Array to explore
    value (float): Desire value
    
    Returns:
        - idx (int)          : Indice of the closest values from the desire value

    Author: Simon Petrus
    '''
    
    idx = (np.abs(array - value)).argmin()

    return idx


# ----------------------------------------------------------------------------------------------------------------------


def format_grid(grid, attr, free_comp, weights):
    '''
    Format PCA or NMF outputs into a single xarray
    
    Parameters
    ----------
    grid              (np.ndarray): Original grid 
    attr                    (dict): Original grid attributs
    free_comp                (int): Number of free components in the new grid (= PCA component used during PCA + 1 (nfs))
    weights           (np.ndarray): PCA or NMF weights grid
    
    Returns:
        - ds_weights              (xarray): Xarray of the PCA or NMF weights grid

    Author: Matthieu Ravet
    '''
    
    # Format the new grids in xarray
    vars_nfs_ws = ["eigen_indices"]
    for key in attr['key']:
        vars_nfs_ws.append(key)
    coords_nfs_ws = {"eigen_indices": np.arange(free_comp)} # The first columns are the normalization factors so you need to add 1
    for key in attr['key']:
        coords_nfs_ws[key] = grid[key].values
    ds_weights= xr.Dataset(data_vars=dict(grid=(vars_nfs_ws, weights)), coords=coords_nfs_ws, attrs=attr)

    return ds_weights

# ----------------------------------------------------------------------------------------------------------------------


def check_format(*params, type_expected):
    '''
    Check that all the components defined in params are in the expected formats

    Args
        *params            : list of parameters
        type_expeced (type): Expected type (list, str, tuple, ...)
        
    Author: Allan Denis
    '''
    
    wrong_format = []
    for param in params:
        if not(isinstance(param, type_expected)):
            wrong_format.append(param)
    
    return wrong_format


# ----------------------------------------------------------------------------------------------------------------------


def weighted_quantile(values, quantiles, weights = None):
    '''
    Compute quantiles for weighted data.
    '''
    values = np.asarray(values)
    quantiles = np.atleast_1d(quantiles)
    
    if weights is None:
        sample_weight = np.ones(len(values))
        
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cumulative_weights = np.cumsum(weights)
    cumulative_weights /= cumulative_weights[-1]
   
    return np.interp(quantiles, cumulative_weights, values)


# ----------------------------------------------------------------------------------------------------------------------


def find_filter_file(filter_name: str) -> str | None:
    '''
    Find a filter file .npz given a filter name, ignoring lowercase and uppercase letters.
    
    Parameters
    ----------
    filter_name (str): Name of the filter ('Keck_NIRC2_H', 'NACO_Lp', 'MIRI_F1065', ...)

    Returns:
        file_path (str | None): Path of the file if it exists, None otherwise
    
    Authors: Allan Denis
    '''
    
    path_list = __file__.split("/")[:-1]
    filter_dir = '/'.join(path_list) + '/phototeque/'
    
    for file_path in glob.glob(os.path.join(filter_dir, '*.npz')):
        root = os.path.basename(file_path).split('.')[0]
        if root.lower() == filter_name.lower():
            return file_path
    
    return None


# ----------------------------------------------------------------------------------------------------------------------


def scale_to_one_significant_digit(flux):
    '''
    Returns a tuple (scaled_flux, factor) such that flux ≈ scaled_flux * 10**factor
    
    Authors: Allan Denis
    '''
    
    if len(flux) == 0:
        return 0, 0

    factor = int(np.floor(np.log10(abs(np.mean(flux)))))
    scaled_flux = flux / (10 ** factor)
    
    return scaled_flux, factor


# ----------------------------------------------------------------------------------------------------------------------


def determine_continuum_types(hc_type: str, res_cont: str | float) -> str:
    '''
    Method to determine the star continuum type ("estimate", "remove", "NA") from the high contrast function type

    Parameters
    ----------
    hc_type          (str): high-contrast function type
    res_cont (str | float): Resolution of the continuum

    Returns:
        - star_continuum    (str): star_continuum type ("estimate", "remove" or "NA")
        - remove_continuum (bool): Whether to remove the continuum of the grid models

    Authors: Allan Denis
    '''
    
    if res_cont != 'NA':
        if hc_type != 'NA':
            star_continuum = 'estimate'
            remove_continuum = False
        else:
            star_continuum = 'remove'
            remove_continuum = True
    else:
        star_continuum = 'NA'
        remove_continuum = False
    
    return star_continuum, remove_continuum
