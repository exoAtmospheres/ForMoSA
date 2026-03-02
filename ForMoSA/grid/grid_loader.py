import os
import xarray as xr
from pathlib import Path
import numpy as np

from ForMoSA.core.errors import ForMoSAError

class GridLoader:
    '''
    Class responsible for Grid loading.

    Authors: Allan Denis
    '''

    @staticmethod
    def _validate_model_grid_dataset(ds):
        '''
        Validate that an xarray.Dataset conforms to the ForMoSA grid specifications

        Authors: Allan Denis
        '''

        # Grid key
        if "grid" not in ds:
            raise ForMoSAError("Dataset must contain 'grid'")

        # Attributes
        required_attrs = ["key", "par", "title", "unit", "res"]

        for attr in required_attrs:
            if attr not in ds.attrs:
                raise ForMoSAError(f"Missing grid attribute: {attr}")

        # Wavelength
        dims = ds["grid"].dims

        if dims[0] != "wavelength":
            raise ForMoSAError("First grid dimension must be 'wavelength'")

        # Grid dimensions
        param_keys = ds.attrs["key"]

        if list(dims[1:]) != list(param_keys):
            raise ForMoSAError(f"Grid dimensions {dims[1:]} do not match attrs['key']={param_keys}")

        # Resolution
        if len(ds["wavelength"]) != len(ds.attrs["res"]):
            raise ForMoSAError("attrs['res'] length must match wavelength size")

        # Wavelength unit
        if "wave_unit" in ds.attrs:
            if not isinstance(ds.attrs['wave_unit'], str):
                raise ForMoSAError(f"Invalid type for 'wave_unit' attribute: {type(ds.attrs['wave_unit'])}. Must be a string")

    @staticmethod
    def _from_file(path: str | os.PathLike) -> xr.Dataset:
        '''
        Load and validate a ForMoSA model grid from a NetCDF file.

        Parameters
        ----------
        path (str | os.PathLike): Path to the NetCDF grid file.

        Returns
        -------
        xr.Dataset: Validated ForMoSA grid dataset.

        Authors: Allan Denis
        '''

        path = Path(path)

        # Check path
        if not path.exists():
            raise ForMoSAError(f"Grid file does not exist: {path}")

        if path.suffix != ".nc":
            raise ForMoSAError(f"Grid file must be a .nc file: {path}")

        # Open dataset
        try:
            ds = xr.open_dataset(path, decode_cf=False)
            ds.attrs['res'] = np.atleast_1d(ds.attrs['res']).astype(float)
        except Exception as e:
            raise ForMoSAError(f"Failed to open grid file: {e}")

        # Validation
        GridLoader._validate_model_grid_dataset(ds)

        return ds

    @staticmethod
    def _from_data(data: np.ndarray, coords: dict, attrs: dict) -> xr.Dataset:
        '''
        Build and validate a ForMoSA model grid from raw arrays.

        Parameters
        ----------
        data   (np.ndarray): Grid data array. Expected shape: (n_wavelength, n_par1, ..., n_parN)
        coords       (dict): Coordinate dictionary. Must include 'wavelength' and parameter coordinates.
        attrs        (dict): Grid attributes required by ForMoSA.

        Returns
        -------
        xr.Dataset: Validated ForMoSA grid dataset.

        Authors: Allan Denis
        '''

        # Data type
        if not isinstance(data, np.ndarray):
            raise ForMoSAError("data must be a numpy.ndarray")

        # Wavelength
        if "wavelength" not in coords:
            raise ForMoSAError("coords must include 'wavelength'")

        # Keys
        param_keys = attrs.get("key")
        if param_keys is None:
            raise ForMoSAError("attrs must include 'key'")

        # Data shape
        dims = ["wavelength"] + param_keys
        expected_shape = tuple(len(coords[d]) for d in dims)

        if data.shape != expected_shape:
            raise ForMoSAError(f"Data shape {data.shape} does not match expected {expected_shape}")

        # Grid generation
        ds = xr.Dataset(data_vars=dict(grid=(dims, data)), coords=coords, attrs=attrs)

        # Validation
        GridLoader._validate_model_grid_dataset(ds)

        return ds
