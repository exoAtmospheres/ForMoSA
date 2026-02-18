import logging
import numpy as np
import xarray as xr

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.grid.subgrid_base import SubGrid
from ForMoSA.grid.model_grid import ModelGrid
from ForMoSA.core.loggings import setup_logging
from ForMoSA.grid.grid_loader import GridLoader
from ForMoSA.filter.filter import PhotometryFilter
from ForMoSA.observation.observation_base import Observation
from ForMoSA.transform.photometric_effects import PhotometricEffects
from ForMoSA.transform.observed import ObservedModel, ObservedParameters
from ForMoSA.core.enums import ObservationType, WavelengthUnit, ParameterKind, LogLikelihoodType

class SubGridPhotometry(SubGrid):
    '''
    Photometric subgrid class, which implements adaptation to a specific filter.

    Parameters
    ----------
    parent_grid          (ModelGrid): Parent model grid
    Filter        (PhotometryFilter): Instance of :class:~PhotometryFilter corresponding to the photometric filter
    logger          (logging.Logger): Logger
    log_level                  (str): Level of the logger
    name                       (str): Name of the subgrid

    Authors: Allan Denis
    '''

    def __init__(self, grid: xr.Dataset, parent_grid: ModelGrid, Filter: PhotometryFilter, logger: logging.Logger | None = None, log_level: str = "INFO", display_unit: WavelengthUnit = WavelengthUnit.MICROMETER, name: str = 'Unknown'):
        super().__init__(grid, parent_grid = parent_grid, logger=logger, log_level=log_level, name=name, display_unit=display_unit)

        self._Filter = Filter
        self.Filter._set_unit(display_unit)

        self._validate_photometry()

    # ================================================
    # Representation
    # ================================================

    def __repr__(self) -> str:
        return f"<SubGridPhotometry name={self.name} filter_id={self.Filter.filter_id}>"

    # ================================================
    # properties
    # ================================================

    @property
    def GridType(self) -> ObservationType:                            # Observation type
        return ObservationType.PHOTOMETRIC.obstype

    @property
    def wave_cont(self) -> np.ndarray | None:                         # Wavelengths for continuum removal
        return None

    @property
    def res_cont(self) -> float | None:                               # Resolutions for continuum removal
        return None

    @property
    def remove_continuum(self) -> bool:                               # Whether to remove continuum
        return False

    @property
    def Filter(self) -> PhotometryFilter:                             # Filter
        if not isinstance(self._Filter, PhotometryFilter):
            raise ForMoSAError(f'Wrong instance for Filter: {type(self._Filter)}. Expecting an instance of PhotometryFilter>', self.logger)
        return self._Filter

    @property
    def relevant_parameter_kinds(self) -> list[ParameterKind]:        # List of relevant parameter kinds the subgrid applies to
        return [
            ParameterKind.GRID,
            ParameterKind.ALPHA,
            ParameterKind.RADIUS,
            ParameterKind.DISTANCE,
            ParameterKind.AV,
            ParameterKind.BB_T,
            ParameterKind.BB_R
        ]

    # ======================================================
    # Class methods
    # ======================================================

    @classmethod
    def from_parent(cls, parent_grid: ModelGrid, Filter: PhotometryFilter, name: str = 'unknown', logger: logging.Logger | None = None, log_level: str = 'INFO', display_unit: WavelengthUnit = WavelengthUnit.MICROMETER) -> 'SubGridPhotometry':
        '''
        Build Photometric subgrid from the parent grid, target_wavelength.

        Parameters
        ----------
        parent_grid (ModelGrid): Instance of ModelGrid
        Filter        (PhotometryFilter): Instance of :class:~PhotometryFilter corresponding to the photometric filter
        name                     (str): Name of the subgrid
        logger (logging.Logger | None): Logger
        log_level                (str): Level of the Logger
        display_unit  (WavelengthUnit): Unit to display for the wavelength

        Returns
        -------
        SubGridPhotometryy: Instance of SubGridPhotometry

        Usage: subgrid = SubGridPhotometry.from_parent(parent_grid, Filter, name, logger, log_level, display_unit)

        Authors: Allan Denis
        '''

        subgrid = cls(
            grid=parent_grid.grid,
            parent_grid=parent_grid,
            Filter=Filter,
            name=name,
            logger=logger,
            log_level=log_level,
            display_unit=display_unit,
        )

        subgrid._grid = subgrid._build_empty_adapted_grid(target_wavelength=Filter.central_wavelength, target_resolution=0)
        subgrid.adapt()

        return subgrid

    @classmethod
    def from_grid(cls, ds: xr.Dataset, parent_grid: ModelGrid, logger: logging.Logger | None = None, log_level: str = 'INFO', display_unit: WavelengthUnit = WavelengthUnit.MICROMETER) -> 'SubGridPhotometry':
        '''
        Retrieve photometric subgrid from grid.

        Parameters
        ----------
        dx                (xr.Dataset): Grid data
        parent_grid        (ModelGrid): Instance of ModelGrid
        logger (logging.Logger | None): Logger
        log_level                (str): Level of the Logger
        display_unit  (WavelengthUnit): Unit to display for the wavelength

        Returns
        -------
        SubGridPhotometry: Instance of SubGridPhotometry

        Usage: subgrid = SubGridPhotometry.from_grid(ds, parent_grid, logger, log_level, display_unit)

        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name='SubGridPhotometry')

        # Validation
        try:
            GridLoader._validate_model_grid_dataset(ds)
        except ForMoSAError as e:
            raise ForMoSAError(e, logger)

        if ds.attrs.get("grid_type") != ObservationType.PHOTOMETRIC.value:
            raise ForMoSAError(f"Wrong grid type: {ds.attrs.get('grid_type')}. Expected {ObservationType.PHOTOMETRIC.value}")

        name = ds.attrs.get('name', 'unknown')
        filter_name = ds.attrs.get("filter_name")
        Filter = PhotometryFilter._from_filter_name(filter_name)

        logger.info('    Creating photometric SubGrid from dataset')

        return cls(
            grid=ds,
            parent_grid=parent_grid,
            Filter=Filter,
            name=name,
            logger=logger,
            log_level=log_level,
            display_unit=display_unit,
        )

    # ======================================================
    # Methods
    # ======================================================

    def _validate_photometry(self) -> None:
        '''
        Check the consistency between the target wavelength and the wavelengths of the filter.

        Authors: Allan Denis
        '''

        pass

    def adapt(self) -> None:
        '''
        Adapt the native grid to the target wavelength and resolution.

        Authors: Allan Denis
        '''

        self.adapt_grid()
        self._grid.attrs['filter_name'] = self.Filter.name

    def _get_restriction_bounds(self) -> tuple[float, float]:
        '''
        Get restriction bounds for computing the restricted subgrid before the adaptation of the subgrid.

        Returns
        -------
        tuple[float, float]: Minimumn and maximum wavelengths of the restricted subgrid

        Authors: Allan Denis
        '''

        margin = 0.05   # 5% margin
        return self.Filter.wavelength_min * (1 - margin), self.Filter.wavelength_max * (1 + margin)

    def _adapt_model(self, model_to_adapt: xr.DataArray):
        '''
        Method to adapt a specific model to the photometric filter.

        Parameters
        ----------
        model_to_adapt         (xr.DataArray): Model to adapt

        Returns
        -------
        model_adapted

        Authors: Allan Denis
        '''

        model_adapted = self.integrate_filter_curve(model_to_adapt)

        return model_adapted

    def _apply_physics_effects(self, observed_model: ObservedModel, observed_params: ObservedParameters) -> xr.DataArray:
        '''
        Apply the physics effects relevant to photometry.

        Parameters
        ----------
        observed_model  (ObservedModel): Model to modify.
        params     (ObservedParameters): Dictionary of parameters values.

        Returns
        -------
        (ObservedModel): observed_model transformed by the physics effects

        Authors: Allan Denis
        '''

        # Parameters relevant to spectroscopy
        allowed_kinds = set(self.relevant_parameter_kinds)

        # Check provided parameters
        invalid = [p.kind for p in observed_params.values.keys() if p.kind not in allowed_kinds]

        if invalid:
            raise ForMoSAError(f"<Parameters {invalid} are not relevant for photometric subgrid>")

        return PhotometricEffects._apply_physics(observed_model, observed_params)

    def integrate_filter_curve(self, model_to_adapt: xr.DataArray, print_logger: bool = False) -> float:
        '''
        Method to integrate the filter curve on a spectrum

        Parameters
        ----------
        model_to_adapt (xr.DataArray): Model to integrate

        Returns
        -------
        xr.DataArray: Integrated value under the filter curve

        Authors: Allan Denis
        '''

        if print_logger:
            self.logger.debug(f' Integrate filter curve of {self.Filter.name} on the spectrum')

        # Model wavelength
        wave_model = model_to_adapt["wavelength"]

        # Filter as DataArray
        filter_da = xr.DataArray(
            self.Filter.transmission,
            coords={"wavelength": self.Filter.wavelength},
            dims=("wavelength",),
        )

        # Interpolate filter transmission on model wavelength grid
        filter_interp = filter_da.interp(
            wavelength=wave_model,
            kwargs={"fill_value": 0.0},
        )

        # Restrict to filter support
        filter_interp = filter_interp.where(filter_interp > 0, drop=True)
        model_sub = model_to_adapt.sel(wavelength=filter_interp.wavelength)

        # Weighted integration
        numerator = (model_sub * filter_interp).integrate("wavelength")
        denominator = filter_interp.integrate("wavelength")

        return (numerator / denominator).item()

    def _apply_observational_effects(self, observed_model: ObservedModel, obs: Observation, bounds_lsq: tuple[float, float] =(-np.inf, np.inf)) -> ObservedModel:
        '''
        Apply the observational effects relevant to photometry.

        Parameters
        ----------
        observed_model     (ObservedModel): Instance of class ObservedModel to transform
        obs                  (Observation): Instance of class Observation
        bounds_lsq   (tuple[float, float]): Bounds for the least squares

        Returns
        -------
        ObservedModel: Instance of class ObservedModel transformed

        Authors: Allan Denis
        '''

        return PhotometricEffects._apply_observation(observed_model, obs, bounds_lsq)

    def _compute_loglike(self, observed_model: ObservedModel, obs: Observation, logL_type: LogLikelihoodType) -> float:
        '''
        Compute the loglikelihood given an ObservedModel, an Observation and a loglikelihood function.

        Parameters
        ----------
        observed_model (ObservedModel): Observed model
        obs              (Observation): Observation
        logL_type  (LogLikelihoodType): Loglikelihood function

        Returns
        -------
        float: logL value

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''

        return PhotometricEffects._compute_loglike(observed_model, obs, logL_type)


    # def _determine_target_wavelength_and_resolution(self, target_resolution: str | float) -> tuple[np.ndarray, np.ndarray]:
    #     '''
    #     Determine the target wavelength and resolution grids.

    #     Parameters
    #     ----------
    #     target_resolution (str | float): Target resolution to reach ('obs', 'mod', float)

    #     Returns
    #     -------
    #     tuple[np.ndarray, np.ndarray]: Target wavelength and resolution arrays

    #     Authors: Allan Denis
    #     '''

    #     # Update the native unit to the unit of the observation
    #     self._native_unit = self.observation._display_unit
    #     return self.observation.wave, np.zeros(len(self.observation.wave))


