import numpy as np
import xarray as xr
from astropy.io import fits
from collections.abc import Mapping

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.enums import ObservationKeys

# ----------------------------------------------------------------------------------------------------------------------


def decoupe(second):
    """
    Re-arranged a number of seconds in the hours-minutes-seconds format.

    Parameters
    ----------
        second : float
            number of second
    Returns
    -------
        - float     : hours
        - float     : minutes
        - float     : seconds

    Notes
    -----
    Authors: Simon Petrus
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
    array : array
        Array to explore
    value : float
        Desire value

    Returns
    -------
        int
            Index of the closest values from the desire value

    Notes
    -----
    Authors: Simon Petrus
    '''

    idx = (np.abs(array - value)).argmin()

    return idx


# ----------------------------------------------------------------------------------------------------------------------


def format_grid(grid, attr, free_comp, weights):
    '''
    Format PCA or NMF outputs into a single xarray

    Parameters
    ----------
    grid : np.ndarray
        Original grid
    attr : dict
        Original grid attributs
    free_comp : int
        Number of free components in the new grid (= PCA component used during PCA + 1 (nfs))
    weights : np.ndarray
        PCA or NMF weights grid

    Returns
    -------
        xarray
            Xarray of the PCA or NMF weights grid

    Notes
    -----
    Authors: Matthieu Ravet
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


def get_weighted_percentile(n, data, weights=None):
    '''
    Return the weighted nth percentile(s) of the data

    Parameters
    ----------
        n : float | list | array
            Percentile(s) between 0 and 100
    data : array
        Data array, shape (N,) or (N, M)
    weights : array | None
        Weights array of shape (N,). If None, uniform weights are used.

    Returns
    -------
        percentiles : array
            Weighted percentile values.
            - shape (len(n),) if data is 1D
            - shape (len(n), M) if data is 2D

    Notes
    -----
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


def scale_to_one_significant_digit(flux):
    '''
    Returns a tuple (scaled_flux, factor) such that flux ≈ scaled_flux * 10**factor

    Notes
    -----
    Authors: Allan Denis
    '''

    if len(flux) == 0:
        return 0, 0

    factor = int(np.floor(np.log10((np.sqrt(np.sum(flux**2))))))
    scaled_flux = flux / (10 ** factor)

    return scaled_flux, factor


# ----------------------------------------------------------------------------------------------------------------------


def from_recarray_to_dic(data: fits.fitsrec.FITS_rec):
    '''
    Transform a recarray (from a fits file) into a dictionary

    Parameters
    ----------
    data : fits.fitsred.FITS_rec
        Data from a fits file

    Returns
    -------
    data_dict : dic
        Dictionary representation of the data

    Notes
    -----
    Authors: Allan Denis
    '''

    return {name: np.asarray(data[name]) for name in data.names}


# -----------------------------------------------------------------------------------------------------------------------


def normalize_list(value, name: str, converter=None):
    '''
    Normalize value to a list and apply optional conversion.

    Parameters
    ----------
    value : Any
        Value to normalize
    name : str
        Parameter name (for error messages)
    converter : callable
        Function applied to each element if provided

    Returns
    -------
    list
        Normalized list

    Notes
    -----
    Authors: Allan Denis
    '''

    # Ensure list
    if isinstance(value, list):
        values = value
    elif isinstance(value, (str, int, float)):
        values = [value]
    else:
        raise ForMoSAError(f" {name} must be str, int, float or list, got {type(value)}")

    # Apply converter if provided
    if converter is not None:
        out = []
        for v in values:
            try:
                out.append(converter(v))
            except Exception:
                raise ForMoSAError(f" Invalid value for {name}: {v} ({type(v)})")
        return out

    return values


# -----------------------------------------------------------------------------------------------------------------------


def to_float_if_possible(v):
    '''
    Convert value to float if possible.

    Parameters
    ----------
    v : Any
        Value to convert

    Returns
    -------
    float
        Float if conversion succeeds, otherwise original value

    Notes
    -----
    Authors: Allan Denis
    '''

    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    raise ForMoSAError(f" Invalid value type: {type(v)}")