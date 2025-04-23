import numpy as np
import xarray as xr

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
    """
    Return the indice of the closest values from a desire value in an array.

    Args:
        array (array): Array to explore
        value (float): Desire value
    Returns:
        - idx (int)          : Indice of the closest values from the desire value

    Author: Simon Petrus
    """
    idx = (np.abs(array - value)).argmin()

    return idx


# ----------------------------------------------------------------------------------------------------------------------


def format_grid(grid, attr, free_comp, weights):
    """
    Format PCA or NMF outputs into a single xarray
    
    Args:
        - grid              (np.ndarray): Original grid 
        - attr                    (dict): Original grid attributs
        - free_comp                (int): Number of free components in the new grid (= PCA component used during PCA + 1 (nfs))
        - weights           (np.ndarray): PCA or NMF weights grid
    Returns:
        ds_weights              (xarray): Xarray of the PCA or NMF weights grid

    Author: Matthieu Ravet
    """
    # Format the new grids in xarray
    vars_nfs_ws = ["eigen_indices"]
    for key in attr['key']:
        vars_nfs_ws.append(key)
    coords_nfs_ws = {"eigen_indices": np.arange(free_comp)} # The first columns are the normalization factors so you need to add 1
    for key in attr['key']:
        coords_nfs_ws[key] = grid[key].values
    ds_weights= xr.Dataset(data_vars=dict(grid=(vars_nfs_ws, weights)), coords=coords_nfs_ws, attrs=attr)

    return ds_weights

# ----------------------------------------------------------------------------------------------------------------------