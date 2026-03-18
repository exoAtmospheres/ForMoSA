import os
import json
import logging
import xarray as xr
from pathlib import Path

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.grid.subgrid_base import SubGrid
from ForMoSA.grid.model_grid import ModelGrid
from ForMoSA.core.enums import ObservationType
from ForMoSA.core.loggings import setup_logging
from ForMoSA.grid.subgrid_photometry import SubGridPhotometry
from ForMoSA.grid.subgrid_spectroscopy import SubGridSpectroscopy

class SubGridSet(object):
    '''
    Container for a set of subgrids (spectroscopic or photometric).

    Can be initialized empty or directly from an ObservationSet.

    Parameters
    ----------
    parent_grid : ModelGrid
        Parent model grid
    logger : logging.Logger | None
        Logger
    log_level : str
        Logger level

    Notes
    -----
    Authors: Allan Denis
    '''

    def __init__(self, parent_grid: ModelGrid, logger: logging.Logger | None = None, log_level: str = "INFO") -> None:

        self._logger = logger or setup_logging(log_level)

        if not isinstance(parent_grid, ModelGrid):
            raise ForMoSAError(" parent_grid must be a ModelGrid instance", self.logger)
        self._parent_grid = parent_grid
        self._subgrids: list[SubGrid] = []

    # ======================================
    # Representation
    # ======================================

    def __repr__(self) -> str:
        return f" SubGridSet parent_grid={self.parent_grid.grid_name}, n_subgrids={len(self._subgrids)}"

    # ======================================
    # Properties
    # ======================================

    @property
    def logger(self) -> logging.Logger:
        """Logger."""
        return self._logger

    @property
    def parent_grid(self) -> ModelGrid:
        """Parent grid."""
        return self._parent_grid

    @property
    def subgrids(self) -> list[SubGrid]:
        """List of adapted subgrids."""
        return self._subgrids

    @property
    def n_subgrids(self) -> int:
        """Number of subgrids."""
        return len(self._subgrids)

    @property
    def is_empty(self) -> bool:
        """Whether the subgrid set is empty."""
        return len(self._subgrids) == 0

    @property
    def has_spectroscopy(self) -> bool:
        """Whether the subgrid set contains spectroscopic subgrids."""
        return any(sg.is_spectroscopic for sg in self._subgrids)

    @property
    def has_photometry(self) -> bool:
        """Whether the subgrid set contains photometric subgrids."""
        return any(sg.is_photometric for sg in self._subgrids)

    @property
    def spectroscopic_subgrids(self) -> list[SubGridSpectroscopy]:
        """List of spectroscopic subgrids."""
        return [sg for sg in self._subgrids if sg.is_spectroscopic]

    @property
    def photometric_subgrids(self) -> list[SubGridPhotometry]:
        """List of photometric subgrids."""
        return [sg for sg in self._subgrids if sg.is_photometric]

    @property
    def wavelength_range(self) -> tuple:
        """Global wavelength range."""
        if self.is_empty:
            return (0.0, 0.0)
        wmins = [sg.wavelength_range[0] for sg in self._subgrids]
        wmaxs = [sg.wavelength_range[1] for sg in self._subgrids]
        return (min(wmins), max(wmaxs))

    @property
    def subgrid_names(self) -> list[str]:
        """List of grid names."""
        return [grid.grid_name for grid in self.subgrids]

    # ======================================
    # Class methods
    # ======================================

    @classmethod
    def from_path(cls, path: str | os.PathLike, parent_grid: ModelGrid, logger: logging.Logger | None = None, log_level: str = 'INFO') -> "SubGridSet":
        '''
        Generate an instance of SubGridSet from the folder containing all the subgrids.

        Parameters
        ----------
        path : str | os.PathLike
            Path containing the subgrids
        parent_grid : ModelGrid
            Instance of ModelGrid corresponding to the parent grid
        logger : logging.Logger
            Logger
        log_level : str
            Level of the Logger

        Returns
        -------
        "SubGridSet"
            Instance of SubGridSet

        Notes
        -----
        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name='SubGridSet')

        logger.debug(f'Building instance of SubGridSet from the path {path}')

        # Initial checking
        if not isinstance(path, (str, os.PathLike)):
            raise ForMoSAError(f' Wrong type for path: {type(path)}. Expected a str or os.PathLike', logger)

        if not isinstance(parent_grid, ModelGrid):
            raise ForMoSAError(f' Wrong type for parent_grid: {type(parent_grid)}. Epected a ModelGrid', logger)

        # Creating instance
        subgrid_set = cls(parent_grid=parent_grid, logger=logger)
        subgrid_path = Path(path).expanduser() / 'SubGrids'
        subgrid_files = [subgrid for subgrid in os.listdir(subgrid_path) if subgrid.lower().endswith('.nc')]

        if len(subgrid_files) == 0:
            raise ForMoSAError(f' {subgrid_path} does not contain any nc file', logger)

        # Generate ordered subgrids
        order_file = subgrid_path / "subgrid_order.json"

        if order_file.exists():
            with open(order_file, "r") as f:
                ordered_names = json.load(f)
                subgrid_files = [f"adapted_{name}.nc" for name in ordered_names]
        else:
            logger.warning("No order_file found. The extracted subgrid order likely won't match the initial subgrid order")

        # Filling instance
        for file in subgrid_files:
            file = os.path.join(subgrid_path, file)
            if file.endswith('.nc'):
                subgrid_set.add_subgrid(file)

        logger.info('    Set of SubGrids generated')
        return subgrid_set

    # ======================================
    # Methods
    # ======================================

    def add_subgrid(self, *args, **kwargs) -> None:
        '''
        Add a subgrid to the set based on the type of data provided.

        Parameters
        ----------
        args :
            - If a SubGrid object is provided, directly add it
            - If a `.nc` file is provided, provide a single argument (str | Path)
            - If an xr.Dataset is provided, provide a single argument

        Notes
        -----
        Authors: Allan Denis
        '''

        if len(args) == 1:
            if isinstance(args[0], SubGrid):
                # If the argument is a SubGrid
                subgrid = args[0]
            elif isinstance(args[0], (str, os.PathLike)):
                # If the argument is a .nc file
                subgrid = SubGrid.from_file(args[0], self.parent_grid, logger=self.logger)
                # If the argument is a xarray.Dataset
            elif isinstance(args[0], xr.Dataset):
                subgrid = SubGrid.from_dataset(args[0], self.parent_grid, logger=self.logger)
            else:
                raise ForMoSAError(f"Unrecognized input type for subgrid: {type(args[0])}", self.logger)
        else:
            raise ForMoSAError("No valid data provided to add a subgrid", self.logger)

        self.logger.info(f'      Adding {subgrid.GridType} Subgrid with name {subgrid.name} to the set of subgrids')
        self._subgrids.append(subgrid)

    def interpolate_all(self, method: str) -> None:
        '''
        Interpolate missing values of all the subgrids.

        Parameters
        ----------
        method : str
            Interpolation method

        Notes
        -----
        Authors: Allan Denis
        '''

        if not isinstance(method, str):
            raise ForMoSAError(f' Wrong type for method: {type(method)}. Expected a string', self.logger)

        for subgrid in self.subgrids:
            subgrid._interpolate_missing_values(method=method)

    def save_all(self, path: str | os.PathLike) -> None:
        '''
        Save all the grids of the set of subgrids to the directort store_path.

        Parameters
        ----------
        store_path : str | os.PathLike
            Directory where to save the subgrids

        Notes
        -----
        Authors: Allan Denis
        '''

        path = Path(path).expanduser() / 'Subgrids'
        if not path.exists():
            self.logger.warning(f'path {path} does not exist. Creating it')
            path.mkdir(exist_ok=True, parents=True)

        self.logger.info(f'    Saving all the subgrids {self.subgrid_names} to path {path}')
        for subgrid in self.subgrids:
            subgrid.save_grid(path)

        # Save order
        order_file = path / "subgrid_order.json"
        with open(order_file, "w") as f:
            json.dump(self.subgrid_names, f, indent=4)