import logging
import numpy as np
import xarray as xr

import ForMoSA.utils.spec as us
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.grid.subgrid_base import SubGrid
from ForMoSA.grid.model_grid import ModelGrid
from ForMoSA.core.loggings import setup_logging
from ForMoSA.grid.grid_loader import GridLoader
from ForMoSA.observation.observation_base import Observation
from ForMoSA.transform.spectroscopic_effects import SpectralEffects
from ForMoSA.transform.observed import ObservedModel, ObservedParameters
from ForMoSA.core.enums import ObservationType, WavelengthUnit, ParameterKind, LogLikelihoodType

class SubGridSpectroscopy(SubGrid):
    '''
    Spectral subgrid class, which implements adaptation to a specific wavelength and resolution.

    Parameters
    ----------
    grid                (xr.Dataset): Grid
    parent_grid          (ModelGrid): Parent model grid
    remove_continuum          (bool): Whether to remove the continuum
    wave_cont                  (str): Wavelengths for continuum removal ('window1 / window2 / windw3 / ...' where window{i} = 'wave{i}, wave{i+1}')
    res_cont                 (float): Resolutions for continuum removal
    logger          (logging.Logger): Logger
    log_level                  (str): Level of the logger
    name                       (str): Name of the subgrid

    Authors: Allan Denis
    '''
    def __init__(self, grid: xr.Dataset, parent_grid: ModelGrid | None = None, remove_continuum: bool = False, wave_cont: str | None = None, res_cont: float | None = None, logger: logging.Logger | None = None, log_level: str = "INFO", display_unit: WavelengthUnit = WavelengthUnit.MICROMETER, name: str = 'Unknown'):
        super().__init__(grid, parent_grid = parent_grid, logger = logger, log_level = log_level, name = name, display_unit = display_unit)

        self._wave_cont = wave_cont
        self._res_cont = res_cont
        self._remove_continuum = bool(remove_continuum)

        self._validate_spectral()

    # ================================================
    # Representation
    # ================================================

    def __repr__(self) -> str:
        return f" SubGridSpectroscopy name={self.name}"

    # ================================================
    # properties
    # ================================================

    @property
    def GridType(self) -> ObservationType:                                    # Observation type
        return ObservationType.SPECTROSCOPIC.value

    @property
    def wave_cont(self) -> str | np.ndarray | None:                           # Wavelengths for continuum removal
        return self._wave_cont

    @property
    def res_cont(self) -> float | None:                                       # Resolutions for continuum removal
        return self._res_cont

    @property
    def remove_continuum(self) -> bool:                                       # Whether to remove continuum
        return self._remove_continuum

    @property
    def relevant_parameter_kinds(self) -> list[ParameterKind]:                # List of relevant parameter kinds the observation applies to
        return [
            ParameterKind.GRID,
            ParameterKind.RV,
            ParameterKind.VSINI,
            ParameterKind.LD,
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
    def from_parent(cls, parent_grid: ModelGrid, target_wavelength: np.ndarray, target_resolution: np.ndarray, remove_continuum: bool = False, wave_cont: np.ndarray | None = None, res_cont: float | None = None, name: str = 'unknown', logger: logging.Logger | None = None, log_level: str = 'INFO', display_unit: WavelengthUnit = WavelengthUnit.MICROMETER) -> 'SubGridSpectroscopy':
        '''
        Build spectroscopic subgrid from the parent grid, target_wavelength.

        Parameters
        ----------
        parent_grid (ModelGrid): Instance of ModelGrid
        target_wavelength (np.ndarray): Target wavelength to reach for the subgrid
        target_resolution (np.ndarray): Target resolution to reach for the subgrid
        remove_continuum        (bool): Whether to remove the continuum
        wave_cont         (str | None): Wavelengths used for the continuum
        res_cont        (float | None): Resolution used for the continuum
        name                     (str): Name of the subgrid
        logger (logging.Logger | None): Logger
        log_level                (str): Level of the Logger
        display_unit  (WavelengthUnit): Unit to display for the wavelength

        Returns
        -------
        SubGridSPectroscopy: Instance of SubGridSpectroscopy

        Usage: subgrid = SubGridSpectroscopy.from_parent(parent_grid, target_wavelength, target_resolution remove_continuum, wave_cont, res_cont, name, logger, log_level, display_unit)

        Authors: Allan Denis
        '''

        subgrid = cls(
            grid=parent_grid.grid,
            parent_grid=parent_grid,
            remove_continuum=remove_continuum,
            wave_cont=wave_cont,
            res_cont=res_cont,
            name=name,
            logger=logger,
            log_level=log_level,
            display_unit=display_unit,
        )
        subgrid._grid = subgrid._build_empty_adapted_grid(target_wavelength=target_wavelength, target_resolution=target_resolution)
        subgrid.adapt()

        return subgrid

    @classmethod
    def from_grid(cls, ds: xr.Dataset, parent_grid: ModelGrid, logger: logging.Logger | None = None, log_level: str = 'INFO', display_unit: WavelengthUnit = WavelengthUnit.MICROMETER) -> 'SubGridSpectroscopy':
        '''
        Retrieve spectorscopic subgrid from grid.

        Parameters
        ----------
        dx                (xr.Dataset): Grid data
        parent_grid        (ModelGrid): Instance of ModelGrid
        logger (logging.Logger | None): Logger
        log_level                (str): Level of the Logger
        display_unit  (WavelengthUnit): Unit to display for the wavelength

        Returns
        -------
        SubGridSpectroscopy: Instance of SubGridSpectroscopy

        Usage: subgrid = SubGridSpectroscopy.from_grid(grid, parent_grid, logger, log_level, display_unit)

        Authors: Allan Denis
        '''

        logger = logger or setup_logging(level=log_level, name='SubGridSpectroscopy')

        # Validation
        try:
            GridLoader._validate_model_grid_dataset(ds)
        except ForMoSAError as e:
            raise ForMoSAError(e, logger)

        if ds.attrs.get("grid_type") != ObservationType.SPECTROSCOPIC.value:
            raise ForMoSAError(f"Wrong grid type: {ds.attrs.get('grid_type')}. Expected {cls.GridType}")

        name = ds.attrs.get('name', 'unknown')
        remove_continuum, wave_cont, res_cont = bool(ds.attrs.get('remove_continuum')), ds.attrs.get('wave_cont'), ds.attrs.get('res_cont')

        logger.info('    Creating spectroscopic SubGrid from dataset')

        return cls(
            grid=ds,
            parent_grid=parent_grid,
            remove_continuum=remove_continuum,
            wave_cont=wave_cont,
            res_cont=res_cont,
            name=name,
            logger=logger,
            log_level=log_level,
            display_unit=display_unit,
        )

    # ======================================================
    # Methods
    # ======================================================

    def _validate_spectral(self) -> None:
        '''

        Authors: Allan Denis
        '''

        if (self.remove_continuum) and (self.wave_cont is None or self.res_cont is None):
            raise ForMoSAError(' If you want to remove the continuum, set values for wave_cont and res_cont' , self.logger)

    def adapt(self) -> None:
        '''
        Adapt the native grid to the target wavelength and resolution. Optionally remove the continuum.

        '''

        try:
            self.adapt_grid()
        except ForMoSAError as e:
            raise ForMoSAError(e, self.logger)

        self._grid.attrs['remove_continuum'] = int(self.remove_continuum)
        self._grid.attrs['wave_cont'] = self.wave_cont
        self._grid.attrs['res_cont'] = self.res_cont

    def _get_restriction_bounds(self) -> tuple[float, float]:
        '''
        Get restriction bounds for computing the restricted subgrid before the adaptation of the subgrid.

        Returns
        -------
        tuple[float, float]: Minimumn and maximum wavelengths of the restricted subgrid

        Authors: Allan Denis
        '''

        margin = 0.05   # 5% margin
        return self.wave[0] * (1 - margin), self.wave[-1] * (1 + margin)

    def _adapt_model(self, model_to_adapt: xr.DataArray) -> np.ndarray:
        '''
        Adapt a single model to the target wavelength and resolution.
        Optionally remove the continuum.

        Parameters
        ----------
        model_to_adapt  (xd.DataArray): Model to adapt

        Returns
        -------
        model_adapted  (xr.DataArray): Adapted model

        Authors: Simon Petrus, Matthieu Ravet, Paulina Palma-Bifani and Allan Denis
        '''

        try:
            if len(self.wave) > 0:
                model_adapted = us.resolution_decreasing(model_to_adapt.coords['wavelength'].values, model_to_adapt.values, model_to_adapt.attrs['res'], self.wave, self.res)
                if self.remove_continuum:
                    model_adapted -= us.continuum_estimate(self.wave, model_adapted, self.res, self.wave_cont, self.res_cont)

        except ForMoSAError as e:   # This line is necessary when we are in a Threapool to stop the execution of the code
            raise e

        return model_adapted

    def _apply_physics_effects(self, observed_model: ObservedModel, params: ObservedParameters) -> xr.DataArray:
        '''
        Apply the physics effects relevant to spectroscopy.

        Parameters
        ----------
        observed_model  (ObservedModel): Instance of class ObservedModel
        params (dict[Parameter, float]): Dictionary of parameters values

        Returns
        -------
        ObservedModel: Instance of class ObservedModel transformed

        Authors: Allan Denis
        '''

        # Parameters relevant to spectroscopy
        allowed_kinds = set(self.relevant_parameter_kinds)

        # Check provided parameters
        invalid = [p.kind for p in params.values.keys() if p.kind not in allowed_kinds]

        if invalid:
            raise ForMoSAError(f" Parameters {invalid} are not relevant for spectroscopic subgrid")

        return SpectralEffects._apply_physics(observed_model, params)

    def _apply_observational_effects(self, observed_model: ObservedModel, obs: Observation, bounds_lsq: tuple[float, float] = (-np.inf, np.inf)) -> ObservedModel:
        '''
        Apply the observational effects relevant to spectroscopy.

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

        return SpectralEffects._apply_observation(observed_model, obs, bounds_lsq)

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

        return SpectralEffects._compute_loglike(observed_model, obs, logL_type)




