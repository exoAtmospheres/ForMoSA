import os
import copy
import logging
import numpy as np
import xarray as xr
from pathlib import Path
import astropy.units as u

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.enums import WavelengthUnit
from ForMoSA.core.loggings import setup_logging
from ForMoSA.grid.grid_loader import GridLoader

class ModelGrid:
    '''
    Representation of a model grid with metadata.
    Handles loading from file and exposing basic properties.

    Parameters
    ----------
    path                (str | Path): Path to the model grid file
    logger         (logging.Logger) : Logger instance for logging
    log_level                  (str): Level of the logger
    display_unit    (WavelengthUnit): Unit of the wavelength to display

    Authors: Allan Denis
    '''

    def __init__(self, dataset: xr.Dataset, model_path: str | os.PathLike | None = None, logger: logging.Logger | None = None, log_level: str = "INFO", display_unit: WavelengthUnit = WavelengthUnit.MICROMETER) -> None:

        self._logger = logger or setup_logging(log_level)

        # Validation
        try:
            GridLoader._validate_model_grid_dataset(dataset)
        except ForMoSAError as e:
            raise ForMoSAError(e, self.logger)

        self._grid = dataset
        self._model_path = model_path
        self._display_unit = display_unit
        self._grid.attrs['grid_type'] = self.GridType

        if 'wave_unit' not in self.attrs.keys():
            self._grid.attrs['wave_unit'] = str(WavelengthUnit.MICROMETER.unit)
            self._logger.warning(f"Wavelength unit not found in grid attributes. Setting to default: {WavelengthUnit.MICROMETER.unit}")

        self.logger.info(f'    Grid dimensions: {tuple(self.dimensions)} {tuple(self.dims)}')

    # ================================================
    # Representation
    # ================================================

    def __repr__(self):
        return f"<ModelGrid name={self.grid_name} path={self.model_path} shape={self.grid['grid'].shape}>"

    # ================================================
    # Properties
    # ================================================

    @property
    def suffix(self) -> str:                                        # Suffix used for saving (Overriden in subgrid)
        return 'native'

    @property
    def GridType(self) -> str:                                      # OGrid type (overriden in subgrid)
        return 'native'

    @property
    def grid(self) -> xr.Dataset:                                    # Grid as xr.Dataset
        return self._grid

    @property
    def grid_as_dataarray(self) -> xr.DataArray:                     # Grid as xr.DataArray (more tailored to manipulations)
        da = self._grid["grid"].copy()
        # Propagate attributes of the Dataset
        da.attrs.update(self.attrs)
        return da

    @property
    def model_path(self) -> Path | str:                              # Path to the model
        return Path(self._model_path).expanduser() if self._model_path is not None else 'in-memory-grid'

    @property
    def grid_name(self) -> str:                                      # Name of the grid (Overriden in subgrid)
        return str(self.model_path).split('/')[-1].split('.nc')[0]

    @property
    def logger(self) -> logging.Logger:                              # Logger
        return self._logger

    @property
    def native_unit(self) -> u.core.PrefixUnit:                      # Native unit of the wavelength
        return WavelengthUnit[self.attrs['wave_unit']].unit

    @property
    def unit(self) -> u.core.PrefixUnit:                             # Unit of the wavelength to display
        return self._display_unit.unit

    @property
    def wave(self) -> np.ndarray:                                    # Wavelength of the grid
        return ((self.grid.coords['wavelength'].values * self.native_unit).to(self.unit)).value

    @property
    def res(self) -> np.ndarray:                                    # Resolution of the grid
        return self.attrs['res']

    @grid.setter
    def grid(self, grid_array):                                      # Grid setter
        self._grid = grid_array
        return grid_array

    @property
    def attrs(self) -> dict:                                         # Dictionary of attributes of the grid
        return dict(self.grid.attrs)

    @property
    def keys(self) -> list:                                          # Keys of the grid parameters
        return self.attrs['key']

    @property
    def titles(self) -> list:                                        # Names of the grid parameters
        return self.attrs['title']

    @property
    def key_values(self) -> dict:                                    # Values taken by the grid parameters
        values = {}
        for key in self.keys:
            values[key] = np.atleast_1d(self.grid[key].values)
        return values

    @property
    def lims_params_grid(self):                                       # Limits of grid parameters
        return {par : [min(self.key_values[par]), max(self.key_values[par])] for par in self.keys}

    @property
    def nyquist(self) -> np.ndarray:                                 # Nyquist sampling
        if self.wave is None or len(self.wave) < 2:
            return self.wave

        diff = np.diff(self.wave, append=(2*self.wave[-1]-self.wave[-2]))
        return self.wave / (2 * diff)

    @property
    def effective_resolution(self) -> np.ndarray:                    # Effective resolution beeing the minimum between Nyquist sampling and resolution
        return np.minimum(self.res, self.nyquist)

    @property
    def n_grids(self) -> int:                                        # Number of grids
        return self.grid.grid[0,].size

    @property
    def size(self) -> int:                                           # Size of the grid
        return self.grid.data_vars['grid'].size

    @property
    def dims(self) -> list[str]:                                     # List of names for each dimension of the grid
        return list(self.grid.dims.keys())

    @property
    def dimensions(self) -> list[int]:                               # List of number of points for each dimension
        return list(self.grid.dims.values())

    # ================================================
    # Class methods
    # ================================================

    @classmethod
    def from_file(cls, path: str | os.PathLike, logger: logging.Logger | None = None, log_level: str = 'INFO', display_unit: WavelengthUnit = WavelengthUnit.MICROMETER):
        '''
        Generate grid from file.

        Parameters
        ----------
        path         (str | os.PathLike): Path to the grid
        logger         (logging.Logger) : Logger instance for logging
        log_level                  (str): Level of the logger
        display_unit    (WavelengthUnit): Unit of the wavelength to display

        Returns
        -------
        ModelGrid: Instance of :class:~ModelGrid

        Usage: grid = ModelGrid._from_file(path)

        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name = 'ModelGrid')
        logger.debug(f'Loading ModelGrid from file {path}')
        ds = GridLoader._from_file(path)
        logger.info(f'    ModelGrid generated from {path}')

        return cls(ds, model_path=path, logger=logger, display_unit=display_unit)

    @classmethod
    def _from_attributes(cls, data: np.ndarray, coords: dict, attrs: dict, logger: logging.Logger | None = None, log_level: str = 'INFO', display_unit: WavelengthUnit = WavelengthUnit.MICROMETER):
        '''
        Generate grid from attributes

        Parameters
        ----------
        data                (np.ndarray): Array of the grid
        coords                    (dict): Dictionnary of coordinates
        attrs                     (dict): Dictionnary of attributes
        logger         (logging.Logger) : Logger instance for logging
        log_level                  (str): Level of the logger
        display_unit    (WavelengthUnit): Unit of the wavelength to display

        Returns
        -------
        ModelGrid: Instance of :class:~ModelGrid

        Usage: grid = ModelGrid._from_attributes(data, coords, attrs)

        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name='ModelGrid')
        logger.debug('<Generate ModelGrid from attributes>')
        ds = GridLoader._from_data(data, coords, attrs)
        logger.info('ModelGrid generated')

        return cls(ds, logger=logger, display_unit=display_unit)

    # ================================================
    # Methods
    # ================================================

    def _set_unit(self, unit: WavelengthUnit):
        '''
        Method to set the unit used for the wavelength

        Parameters
        ----------
        unit (WavelengthUnit): unit used (micrometer', 'nanometer', 'angstrom')

        Authors: Allan Denis
        '''

        if not isinstance(unit, WavelengthUnit):
            raise ForMoSAError(f"<Unit must be a WavelengthUnit Enum, got {type(unit)}>", self.logger)

        self._logger.info(f"<Convert the unit used for grid {self.grid_name} from {self.native_unit} to {unit.unit}>")
        self._display_unit = unit

    def _load_model_at_specific_index(self, idx: tuple) -> xr.DataArray:
        '''
        Load a model at a specific index

        Parameters
        ----------
        idx (tuple): Index of model to be loaded (e.g. (5, 0, 6, 3) for a 4-parameter grid)

        Returns
        -------
        model_to_return (xr.DataArray): Model at the specific index

        Authors: Allan Denis
        '''

        if not isinstance(idx, tuple):
            raise ForMoSAError(f'<Index is type{type(idx)}. It should be a tuple>', self.logger)

        model_to_return = self.grid_as_dataarray[(..., ) + idx]
        if np.any(np.isnan(model_to_return)):
            msg = 'Extraction of model failed : '
            for i, (key, title) in enumerate(zip(self.keys, self.titles)):
                msg += f'{title}={self.key_values[key][idx[i]]}, '
            self._logger.warning(f' {msg}')
        return model_to_return

    def _interpolate_missing_values(self, method: str = "linear", limit: int = None, fill_value: str = 'extrapolate', max_gap: int = None) -> None:
        '''
        Interpolate missing (NaN) values in the grid.

        Parameters
        ----------
        method      (str): Interpolation method to use.
        limit       (int): Maximum number of consecutive NaNs to fill.
        fill_value  (str): Method to fill in points outside of data range
        max_gap     (int): Maximum size of gap, a continuous sequence of NaNs, that will be filled

        Authors: Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        self.logger.info(f'    Interpolating between holes of the grid {self.grid_name}')

        interp_kwargs = {"method": method, "fill_value": fill_value}

        if limit is not None:
            interp_kwargs["limit"] = limit
        if max_gap is not None:
            interp_kwargs["max_gap"] = max_gap

        self._logger.info(f" {self.grid_name}")

        for idx, (key, title) in enumerate(zip(self.keys, self.titles)):
            self.logger.info(f' {idx + 1}/{len(self.keys)} - {title}')

            if self.grid.isnull().any(dim=key).any():
                self._grid = self.grid.interpolate_na(dim=key, **interp_kwargs)

    def _nan_interpolated_grid(self):
        '''
        Return a 1D interpolated grid filled with NaNs,
        with the same structure as a valid interpolated grid.

        Authors: Allan Denis
        '''

        ref_params = {k: 0.5 * (self.lims_params_grid[k][0] + self.lims_params_grid[k][1]) for k in self.keys}

        interp_kwargs = dict(ref_params)
        interp_kwargs["method"] = "nearest"
        interp_kwargs["kwargs"] = {"fill_value": np.nan}

        grid_1d = self._grid.interp(**interp_kwargs)

        # Replace values by nans
        return grid_1d * np.nan

    def _interpolate_between_gridpoints(self, grid_params: dict[str, float], method: str = "linear", print_logger: bool = False):
        '''
        Interpolate between gridpoints in the adapted spectroscopic and photometric grids.

        Parameters
        ----------
        grid_params   (dict[str, float]): Dictionary of grid parameter values
        method                     (str): Interpolation method
        print_logger              (bool): Whether to print logger info

        Returns
        -------
        xr.Dataset: Interpolated 1D grid. If parameters are out-of-bounds, returns a 1D grid filled with NaNs.

        Authors
        -------
        Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        if print_logger:
            self.logger.info('Interpolate between gridpoints in the grid')

        if not isinstance(grid_params, dict):
            raise ForMoSAError(f'<Wrong type for grid_params: {type(grid_params)}. Expected a dictionary>', self.logger)

        # ==================================================
        # Keys validation
        # ==================================================
        expected_keys = set(self.keys)
        provided_keys = set(grid_params.keys())

        if provided_keys != expected_keys:
            missing = expected_keys - provided_keys
            extra = provided_keys - expected_keys
            raise ForMoSAError(f"Grid parameter mismatch. Missing: {missing}, Extra: {extra}", self.logger)

        # ==================================================
        # Bounds checks
        # ==================================================
        out_of_bounds = False
        for name, value in grid_params.items():
            min_val, max_val = self.lims_params_grid[name]
            if not (min_val <= value <= max_val):
                out_of_bounds = True
                self.logger.warning(f"Grid parameter '{name}'={value} outside bounds [{min_val}, {max_val}]. Returning NaN grid")

        # ==================================================
        #  out-of-bounds
        # ==================================================
        if out_of_bounds:
            return self._nan_interpolated_grid()

        # ==================================================
        # Normal interpolation
        # ==================================================
        interp_kwargs = dict(grid_params)
        interp_kwargs["method"] = method
        interp_kwargs["kwargs"] = {"fill_value": np.nan}

        if not isinstance(self.grid, xr.Dataset):
            raise ForMoSAError("Grid is not a valid xarray.Dataset", self.logger)

        if print_logger:
            self.logger.debug('<Interpolation>')

        return self.grid.interp(**interp_kwargs)

    def save_grid(self, store_path : str | os.PathLike) -> None:
        '''
        Save the grid to a specified directory.

        Parameters
        ----------
        store_path (str | os.PathLike): Path where to store the grid

        Authors: Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        if not isinstance(store_path, (str, os.PathLike)):
            raise ForMoSAError(f'<Wrong type for store_path: {type(store_path)}. Expected a string or os.PathLike>', self.logger)

        store_path = Path(store_path).expanduser()

        if not store_path.exists():
            self.logger.warning(f'{store_path} does not exist. Creating it')
            store_path.mkdir(exist_ok=True, parents=True)

        self.logger.info(f"    Saving Grid {self.suffix}_{self.grid_name}.nc to {store_path}")

        filename = f"{self.suffix}_{self.grid_name}.nc"
        self.grid.to_netcdf(store_path / filename, format="NETCDF4", engine="netcdf4", mode="w")

    def _load_grid(self, store_path: str | os.PathLike, grid_name: str = 'in-memory-grid', suffix: str = 'native') -> xr.Dataset:
        '''
        Method to load a grid from a nc file

        Parameters
        ----------
        store_path (str | os.PathLike): Path where to store the grid
        grid_name                (str): Name of the grid
        suffix                   (str): Suffix of the grid

        Returns
        -------
        xr.Dataset: Loaded grid

        Authors: Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        if not isinstance(store_path, (str, os.PathLike)):
            raise ForMoSAError(f'<Wrong type for store_path: {type(store_path)}. Expected a string or os.PathLike>', self.logger)

        store_path = Path(store_path).expanduser()

        self.logger.info(f'<Load adapted grid {suffix}_{grid_name} from the store_path {store_path}>')

        filename = f"{suffix}_{grid_name}.nc"
        grid_file = store_path / filename

        self.logger.debug(f'<Open grid file {grid_file}>')

        try:
            grid = GridLoader._from_file(grid_file)
        except ForMoSAError as e:
            raise ForMoSAError(e, self.logger)

        return grid

    def _restricted_grid(self, windows: str | None = None, print_logger: bool=True, extension: float = 0.0) -> "ModelGrid":
        '''
        Returns a version of the grid restricted to the given wavelength range

        Parameters
        ----------
        windows        (str): Windows in the format 'wmin1,wmax1 / wmin2,wmax2 / ...'
        print_logger  (bool): Whether to print the Logger
        extension    (float): Extension factor of the windows

        Returns
        -------
        'ModelGrid': An instance of class ModelGrid

        Authors: Allan Denis
        '''

        if not isinstance(windows, str):
            raise ForMoSAError(f'<Wrong type for windows: {type(windows)}. Requires a string>', self.logger)

        if windows is None:
            windows = f'{self.wavelength[0]}, {self.wavelength[1]}'

        if print_logger:
            self.logger.debug(f'Restrict grid {self.grid_name} onto windows {windows}')

        ind = np.array([], dtype=int)
        for window in windows.split("/"):
            wmin, wmax = map(float, window.split(","))
            indices = np.where((self.wave >= wmin * (1 - extension)) & (self.wave <= wmax * (1 + extension)))[0]
            ind = np.concatenate((ind, indices))
        ind = np.unique(ind)

        restricted_grid = copy.deepcopy(self)
        # New restriced grid
        restricted_grid._grid = restricted_grid.grid.isel(wavelength=ind)
        restricted_grid.grid._attrs['res'] = restricted_grid.res[ind]

        # Final validation
        try:
            GridLoader._validate_model_grid_dataset(restricted_grid.grid)
        except ForMoSAError as e:
            raise ForMoSAError(e, self.logger)

        if print_logger:
            self.logger.info(f'    Generated restricted Grid. Former grid length: {len(self.wave)}. New grid length: {len(restricted_grid.wave)}')

        return restricted_grid