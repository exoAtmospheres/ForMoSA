import os
import logging
import traceback
import numpy as np
import xarray as xr
from tqdm import tqdm
import astropy.units as u
import multiprocessing as mp
from functools import partial
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.grid.model_grid import ModelGrid
from ForMoSA.core.loggings import setup_logging
from ForMoSA.grid.grid_loader import GridLoader
from ForMoSA.parameter.parameter import Parameter
from ForMoSA.observation.observation_base import Observation
from ForMoSA.transform.observed import ObservedModel, ObservedParameters
from ForMoSA.core.enums import ObservationType, WavelengthUnit, ParameterKind, LogLikelihoodType


class SubGrid(ModelGrid, ABC):
    '''
    Base class for any subgrid (spectroscopic or photometric). Inherits from the ModelGrid class.

    Parameters
    ----------
    grid : xr.Dataset
        Grid
    parent_grid : ModelGrid
        Parent model grid
    target_resolution : str | float
        Target resolution to reach for the model ('obs', 'mod' or float)
    logger : logging.Logger | None
        Logger
    log_level : str
        Level of the Logger
    name : str
        Name of the subgrid
    display_unit : WavelengthUnit
        Unit of the wavelength

    Notes
    -----
    Authors: Allan Denis
    '''

    def __init__(self, grid: xr.Dataset, parent_grid: ModelGrid, logger: logging.Logger | None = None, log_level: str = "INFO", name: str = 'Unknown', display_unit: WavelengthUnit = WavelengthUnit.MICROMETER):

        # Initialization from the ModelGrid
        super().__init__(grid, logger=logger, log_level=log_level, display_unit=display_unit)

        self._name = name
        self._parent_grid = parent_grid

        self._grid.attrs['name'] = name

        self.logger.info(f'    Setting wavelength unit of subgrid {self.name} to {self.unit}>')

    # =========================================
    # Abstract methods
    # (Subclasses must implement these methods)
    # =========================================

    @property
    @abstractmethod
    def GridType(self) -> ObservationType.obstype:
        """Grid type (spectroscopic or photometric)."""
        pass

    @property
    @abstractmethod
    def relevant_parameter_kinds(self) -> list[ParameterKind]:
        """List of relevant parameter kinds the subgrid applies to."""
        return [
            ParameterKind.ALPHA,
            ParameterKind.RADIUS,
            ParameterKind.DISTANCE,
            ParameterKind.AV,
            ParameterKind.BB_T,
            ParameterKind.BB_R
        ]

    @abstractmethod
    def _adapt_model(self, model: xr.DataArray) -> xr.DataArray:
        '''
        Adapt a single model to the target wavelength and resolution.
        Optionally remove the continuum.

        Parameters
        ----------
        model : xr.DataArray
            Model to adapt

        Returns
        -------
        xr.DataArray
            Adapted model

        Notes
        -----
        Authors: Allan Denis
        '''

        pass

    @abstractmethod
    def _get_restriction_bounds(self) -> tuple[float, float]:
        '''
        Get restriction bounds for computing the restricted subgrid before the adaptation of the subgrid.

        Returns
        -------
        tuple[float, float]
            Minimumn and maximum wavelengths of the restricted subgrid

        Notes
        -----
        Authors: Allan Denis
        '''

        pass

    @classmethod
    @abstractmethod
    def _apply_physics_effects(model: ObservedModel, params: dict[Parameter, float]) -> ObservedModel:
        '''
        Apply the relevant physics effects for the given model type.

        Parameters
        ----------
        model : ObservedModel
            Instance of class ObservedModel to transform
        params : dict[Parameter, float]
            Dictionary of parameters values.

        Returns
        -------
        ObservedModel
            Instance of class ObservedModel transformed

        Notes
        -----
        Authors: Allan Denis
        '''

        pass

    @classmethod
    @abstractmethod
    def _apply_observational_effects(model: ObservedModel, obs: Observation, bounds: tuple[float, float] | None = None) -> ObservedModel:
        '''
        Apply the relevant observational effects for the given model type.

        Parameters
        ----------
        observed_model : ObservedModel
            Instance of class ObservedModel to transform
        obs : Observation
            Instance of class Observation
        bounds : tuple[float, float]
            Bounds for the least squares

        Returns
        -------
        ObservedModel
            Instance of class ObservedModel transformed

        Notes
        -----
        Authors: Allan Denis
        '''

        pass

    @staticmethod
    @abstractmethod
    def _compute_loglike(model: ObservedModel, obs: Observation, logL_type: LogLikelihoodType) -> float:
        '''
        Compute the loglikelihood given a transformed model, an observation and a loglikelihood function

        Parameters
        ----------
        model : ObservedModel
            Instance of class ObservedModel
        obs : Observation
            Observation
        logL_type : LogLikelihoodType
            Loglikelihood function

        Returns
        -------
        float
            logL value

        Notes
        -----
        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        pass

    # =========================================
    # Common properties
    # =========================================

    @property
    def parent_grid(self) -> ModelGrid:
        """Parent grid."""
        return self._parent_grid

    @property
    def suffix(self) -> str:
        """Suffix used for saving."""
        return 'adapted'

    @property
    def name(self) -> str:
        """Name of the subgrid."""
        return self._name

    @property
    def grid_name(self) -> str:
        """Name of the grid."""
        return f'{self.parent_grid.grid_name}_{self.name}_{self.GridType}'

    @property
    def wavelength_range(self) -> tuple:
        """Wavelength range of the subgrid."""
        return float(self.wave.min()), float(self.wave.max())

    @property
    def unit(self) -> u.core.PrefixUnit:
        """Unit of the wavelength to display."""
        return self._display_unit.unit

    @property
    def is_spectroscopic(self) -> bool:
        """Whether subgrid is spectroscopic."""
        return self.GridType == ObservationType.SPECTROSCOPIC.obstype

    @property
    def is_photometric(self) -> bool:
        """Whether subgrid is photometric."""
        return self.GridType == ObservationType.PHOTOMETRIC.obstype

    # ==========================
    # Class methods
    # ==========================

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, parent_grid: ModelGrid, logger: logging.Logger | None = None, log_level: str = 'INFO', display_unit: WavelengthUnit = WavelengthUnit.MICROMETER) -> 'SubGrid':
        '''
        Generate SubGrid from dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the parameters of the subgrid
        parent_grid : ModelGrid
            Parent model grid
        logger : logging.Logger
            Logger
        log_level : str
            Level of the Logger

        Returns
        -------
        'SubGrid'
            An instance of class SubGrid

        Notes
        -----
        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name="Observation")
        logger.debug('Extracting SubGrid from dataset')

        from ForMoSA.grid.subgrid_spectroscopy import SubGridSpectroscopy
        from ForMoSA.grid.subgrid_photometry import SubGridPhotometry

        grid_type = ds.attrs.get("grid_type")

        if grid_type is None:
            raise ForMoSAError("Dataset has no 'grid_type' attribute", logger)

        if grid_type == ObservationType.SPECTROSCOPIC.value:
            return SubGridSpectroscopy.from_grid(ds, parent_grid=parent_grid, logger=logger, display_unit=display_unit)

        elif grid_type == ObservationType.PHOTOMETRIC.value:
            return SubGridPhotometry.from_grid(ds, parent_grid=parent_grid, logger=logger, display_unit=display_unit)

        raise ForMoSAError(f"Unrecognized grid_type: {grid_type}. Expected {[grid_type.value for grid_type in ObservationType]}", logger)

    @classmethod
    def from_file(cls, path: str | os.PathLike, parent_grid: ModelGrid, logger: logging.Logger | None = None, log_level: str = 'INFO', display_unit: WavelengthUnit = WavelengthUnit.MICROMETER) -> 'SubGrid':
        '''
        Generate SubGrid from file.

        Parameters
        ----------
        path : str | os.PathLike
            Path to the .nc file
        parent_grid : ModelGrid
            Parent model grid
        logger : logging.Logger
            Logger
        log_level : str
            Level of the Logger
        display_unit : WavelengthUnit
            Unit to display for wavelength

        Returns
        -------
        'SubGrid'
            An instance of class SubGrid

        Notes
        -----
        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name='SubGrid')
        logger.debug(f'Loading grid from file {path}')

        try:
            ds = GridLoader._from_file(path)
        except ForMoSAError as e:
            raise ForMoSAError(e, logger)

        logger.info(f'    {ds.attrs.get("grid_type")} grid detected')

        return cls.from_dataset(ds=ds, parent_grid=parent_grid, logger=logger, display_unit=display_unit)

    # ==========================
    # Methods
    # ==========================

    def _build_empty_adapted_grid(self, target_wavelength: np.ndarray, target_resolution: np.ndarray) -> xr.Dataset:
        '''
        Build en ampty grid from the parent grid and the targets wavelength and resolution

        Parameters
        ----------
        target_wavelength : np.ndarray
            Wavelength to reach for the subgrid
        target_resolution : np.ndarray
            Resolution to reach for the subgrid

        Returns
        -------
        xr.Dataset
            Empty grid

        Notes
        -----
        Authors: Allan Denis
        '''

        self._logger.debug(f'Building empty grid from the native grid {self.parent_grid.grid_name}')

        target_wavelength = np.atleast_1d(target_wavelength).astype(float)
        target_resolution = np.atleast_1d(target_resolution).astype(float)

        parent_wavelength = self.parent_grid.grid["wavelength"].values

        wl_min_parent = np.nanmin(parent_wavelength)
        wl_max_parent = np.nanmax(parent_wavelength)

        wl_min_target = np.nanmin(target_wavelength)
        wl_max_target = np.nanmax(target_wavelength)

        if (wl_min_target < wl_min_parent) or (wl_max_target > wl_max_parent):
            raise ForMoSAError(f"target_wavelength={target_wavelength} is outside the parent grid [{wl_min_parent}, {wl_max_parent}]>", self.logger)

        # Shape of the native grid
        data_shape = len(target_wavelength), *tuple(len(self.parent_grid.grid[key]) for key in self.parent_grid.keys)

        # Generate empty grid with the attributes and coordinates of the native grid
        empty_data = np.full(data_shape, np.nan)

        coords = {"wavelength": target_wavelength}
        coords.update({key: self.parent_grid.grid[key].values for key in self.parent_grid.keys})
        attrs = self.parent_grid.attrs.copy()
        attrs["res"] = target_resolution

        try:
            # Create grid and check consistency between the parameters of the grid
            return GridLoader._from_data(empty_data, coords, attrs)
        except ForMoSAError as e:
            raise ForMoSAError(e, self.logger) from e

        self.logger.info(f"    Initialized empty subgrid with shape {self._grid['grid'].shape}")

    def adapt_grid(self) -> None:
        '''
        Adapt the entire grid to the observation.

        Notes
        -----
        Authors: Arthur Vigan and Allan Denis
        '''

        # Get a restricted version of the parent grid to spped up the adaptation
        wmin, wmax = self._get_restriction_bounds()
        restricted_grid = self.parent_grid._restricted_grid(f'{wmin}, {wmax}')

        # Build indices
        shape = self.grid.grid.values.shape[1:]
        indices = list(np.ndindex(shape))
        total = len(indices)

        self._logger.debug(f'Adapting grid {self.grid_name}')
        self._logger.info("    Parallel adaptation")

        def worker(idx):
            model = restricted_grid._load_model_at_specific_index(idx)
            result = self._adapt_model(model)
            return idx, result

        try:
            n_threads = min(8, mp.cpu_count())  # Limit number of threads to avoid memory overload

            with ThreadPool(processes=n_threads) as pool:
                results = pool.imap_unordered(worker, indices, chunksize=10)

                with tqdm(total=total, leave=False) as pbar:
                    for idx, result in results:
                        self._grid.grid[(...,) + idx] = result
                        pbar.update(1)

        except Exception:
            self._logger.warning("ThreadPool adaptation failed, fallback to serial\n" + traceback.format_exc())

            try:
                for idx in tqdm(indices):
                    model = restricted_grid._load_model_at_specific_index(idx)
                    result = self._adapt_model(model)
                    self._grid.grid[(...,) + idx] = result
            except Exception as e:
                self._logger.error(f"Non parallel adaptation produced the following error: {e}")


    def _compute_loglike_for_obs(self, observed_model: ObservedModel, observation: "Observation", logL_type: LogLikelihoodType = LogLikelihoodType.CHI2, interp_method: str = 'linear', bounds_lsq: tuple[float, float] = (-np.inf, np.inf)) -> float:
        '''
        Evaluate the loglike given a dictionary of parameters and their associated values

        Parameters
        ----------
        observed_model : ObservedModel
            Instance of class ObservedModel
        observation : Observation
            Observation
        logL_type                   LogLikelihoodType): Loglikelihood function
        interp_method : str
            Interpolation method
        bounds_lsq : tuple[float, float]
            (lower, higher) bounds for the Least Squares

        Returns
        -------
        float
            Loglikelihood

        Notes
        -----
        Authors: Allan Denis
        '''

        # Initial check
        if not isinstance(observed_model, ObservedModel):
            raise ForMoSAError(f'Wrong type for observed_model: {type(observed_model)}. Expected an ObservedModel')

        if np.any(np.isnan(observed_model.flux)):
            # NaN are produced by out-of-grid parameters. The loglike must be -inf in this case
            return -float('inf')

        # Compute loglikelihood
        loglike = self._compute_loglike(observed_model, observation, logL_type = logL_type)

        return loglike

    def _build_model_from_params(self, observed_params: ObservedParameters, observation: Observation, interp_method: str = 'linear', bounds_lsq: tuple[float, float] = (-np.inf, np.inf)) -> ObservedModel:
        '''
        Build a model from a dictionarty of parameters and their associated values

        Parameters
        ----------
        observed_params    ObservedParameters): Instance of class ObservedParameters
        observation : Observation
            Observation
        interp_method : str
            Interpolation method
        bounds : tuple[float, float]
            Bounds for the least squares

        Returns
        -------
        ObservedModel
            Model build with the values of the parameters

        Notes
        -----
        Authors: Allan Denis
        '''

        # Split parameters
        grid_params = observed_params.grid
        physics_params  = observed_params.physics

        # Evaluation of the model at grid points
        observed_model = self.evaluate_at_gridpoints(grid_params, interp_method = interp_method)

        if np.any(np.isnan(observed_model.flux)):
            # NaNs produced because grid parameter is outside of the grid bounds, so we directly return the model
            return observed_model

        # Apply physics transformations
        try:
            observed_model = self._apply_physics_effects(observed_model, physics_params)
        except ForMoSAError as e:
            raise ForMoSAError(e, self.logger)

        # Apply observational transformations
        try:
            observed_model = self._apply_observational_effects(observed_model, observation, bounds_lsq = bounds_lsq)
        except ForMoSAError as e:
            raise ForMoSAError(e, self.logger)

        return observed_model

    def evaluate_at_gridpoints(self, grid_params: ObservedParameters, interp_method: str = 'linear') -> ObservedModel:
        '''
        Evaluate model given a list of parameters and their associated values.

        Parameters
        ----------
        params : Dict[Parameter, float]
            Dictionary of grid parameters and their associated values
        interp_method : str
            Interpolation method

        Returns
        -------
        observed_model : ObservedModel
            Instance of class ObservedModel

        Notes
        -----
        Authors: Allan Denis
        '''

        # Initial checks
        if not isinstance(grid_params, ObservedParameters):
            raise ForMoSAError(f'<Wrong type for grid_params: {type(grid_params)}. Expected an ObservedParameters>', self.logger)

        if (not grid_params.has_grid) or (grid_params.has_physics):
            raise ForMoSAError("You should use only grid parameters in grid_params", self.logger)

        grid_values = grid_params.values_by_name

        # Inteprolation
        model = self._interpolate_between_gridpoints(grid_values, method = interp_method)
        observed_model = ObservedModel(model.coords['wavelength'], model.grid.values, model.attrs['res'])

        return observed_model