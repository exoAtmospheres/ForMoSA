import numpy as np
import xarray as xr
import os
import glob

from pathlib import Path

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

    Authors: Allan Denis
    '''

    wrong_format = []
    for param in params:
        if not(isinstance(param, type_expected)):
            wrong_format.append(param)

    return wrong_format


# ----------------------------------------------------------------------------------------------------------------------


def get_weighted_percentile(n, data, weights=None):
    '''
    Return the weighted nth percentile(s) of the data

    Args:
        n (float | list | array): Percentile(s) between 0 and 100
    data                 (array): Data array, shape (N,) or (N, M)
    weights      (array | None): Weights array of shape (N,). If None, uniform weights are used.

    Returns
    -------
        percentiles (array): Weighted percentile values.
            - shape (len(n),) if data is 1D
            - shape (len(n), M) if data is 2D

    Authors: Allan Denis
    '''

    # Convert n to array
    n = np.atleast_1d(n).astype(float)


    if weights is None:
        weights = np.ones(data.shape[0])
    else:
        weights = np.asarray(weights)

    # Normalize weights
    #weights = weights / np.sum(weights)

    if data.ndim == 1:
        # Sort data and weights
        sorter = np.argsort(data)
        sorted_data = data[sorter]
        sorted_weights = weights[sorter]

        cumweights = np.cumsum(sorted_weights)
        cumweights /= cumweights[-1]
        # Interpolate
        percentiles = np.interp(n/100, cumweights, sorted_data)

        if percentiles.shape[0] == 1:
            return percentiles[0]  # return scalar if only one percentile
        else:
            return percentiles

    elif data.ndim == 2:
        n_cols = data.shape[1]
        percentiles = np.zeros((len(n), n_cols))

        for i in range(n_cols):
            column = data[:, i]
            sorter = np.argsort(column)
            sorted_data = column[sorter]
            sorted_weights = weights[sorter]

            cumweights = np.cumsum(sorted_weights)
            cumweights /= cumweights[-1]
            percentiles[:, i] = np.interp(n/100, cumweights, sorted_data)

        if len(n) == 1:
            return percentiles[0, :]
        else:
            return percentiles

    else:
        raise ValueError("Data must be 1D or 2D.")




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

    path_list = Path(__file__)
    filter_dir = path_list.parent.parent / 'phototeque'

    filters = filter_dir.glob('*.npz')
    for filter in filters:
        if filter.stem.lower() == filter_name.lower():
            return filter

    return None


# ----------------------------------------------------------------------------------------------------------------------


def scale_to_one_significant_digit(flux):
    '''
    Returns a tuple (scaled_flux, factor) such that flux ≈ scaled_flux * 10**factor

    Authors: Allan Denis
    '''

    if len(flux) == 0:
        return 0, 0

    factor = int(np.floor(np.log10((np.sqrt(np.sum(flux**2))))))
    scaled_flux = flux / (10 ** factor)

    return scaled_flux, factor


