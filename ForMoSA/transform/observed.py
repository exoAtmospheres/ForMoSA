import numpy as np
from typing import Mapping
from dataclasses import dataclass, replace

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.enums import ParameterKind
from ForMoSA.grid.model_grid import ModelGrid
from ForMoSA.parameter.parameter import Parameter

@dataclass
class ObservedParameters:
    '''
    Parameters drawn from the nested sampling.

    Authors: Allan Denis
    '''

    # ======================
    # Attributes
    # ======================

    values: Mapping[Parameter, float]

    # ======================
    # Post init checks
    # ======================

    def __post_init__(self):
        for key in self.values:
            if not isinstance(key, Parameter):
                raise ForMoSAError(f"All keys must be Parameter instances, got {type(key)}")

    # ======================
    # Properties
    # ======================

    @property
    def grid(self) -> "ObservedParameters":                                    # Grid parameters
        return ObservedParameters({p: v for p, v in self.values.items() if p.kind == ParameterKind.GRID})

    @property
    def physics(self) -> "ObservedParameters":                                 # Non grid parameters
        return ObservedParameters({p: v for p, v in self.values.items() if p.kind != ParameterKind.GRID})

    @property
    def global_params(self) -> "ObservedParameters":                                  # GLobal parameters
        return ObservedParameters({p: v for p, v in self.values.items() if p.scope == 'global'})

    @property
    def names(self) -> list[str]:                                              # Names of the parameters
        return [p.name for p in self.values]

    @property
    def kinds(self) -> list[ParameterKind]:                                    # Parameter kinds of the parameters
        return [p.kind for p in self.values]

    @property
    def has_grid(self) -> bool:                                                # Whether the parameter has a grid parameter kind
        return self.has_kind(ParameterKind.GRID)

    @property
    def has_physics(self) -> bool:                                             # Whether the parameter has a physics parameter kind
        return any(p.kind != ParameterKind.GRID for p in self.values)

    @property
    def values_by_kind(self) -> dict[ParameterKind, float]:                    # Parameter values by kind
        return {p.kind: v for p, v in self.values.items()}

    @property
    def values_by_name(self) -> dict[str, float]:                              # Parameter values by name
        return {p.name: v for p, v in self.values.items()}

    @property
    def params_by_kind(self) -> dict[ParameterKind, float]:                    # Parameters by kind
        return {p.kind: p for p in self.values}

    # ======================
    # Methods
    # ======================

    def has_name(self, name: str) -> bool:
        '''
        Check whether name is present in the parameter names.

        Parameters
        ----------
        name (str): Name to check

        Returns
        -------
        bool: Whether the name is present in the parameter names

        '''

        if not isinstance(name, str):
            raise ForMoSAError('Wrong type for name: {type(name)}. Expected a string')

        return any(p.name == name for p in self.values)

    def has_kind(self, kind: ParameterKind) -> bool:
        '''
        Check whether kind is present in the parameter kinds.

        Parameters
        ----------
        king (ParameterKind): Kind to check

        Returns
        -------
        bool: Whether the name is present in the parameter names

        '''

        if not isinstance(kind, ParameterKind):
            raise ForMoSAError('Wrong type for kind: {type(kind)}. Expected a ParameterKind')

        return any(p.kind == kind for p in self.values)

    def get_name(self, name: str) -> float:
        '''
        Get parameter value according to its name.

        Parameters
        ----------
        name (str): Name of the parameter

        Returns
        -------
        float: Value of the parameter

        Authors: Allan Denis
        '''

        if not self.has_name(name):
            raise ForMoSAError(f'Name ({name}) must be amongst the parameter names: {self.names}')

        for p, v in self.values.items():
            if p.name == name:
                return v

    def get_kind(self, kind: ParameterKind) -> float:
        '''
        Get parameter value according to its kind.

        Parameters
        ----------
        kind (ParameterKind): Kind of the parameter

        Returns
        -------
        float: Value of the parameter

        Authors: Allan Denis
        '''

        if not self.has_kind(kind):
            raise ForMoSAError(f'Kind ({kind}) must be amongst the parameter kinds: {self.kinds}')

        for p, v in self.values.items():
            if p.kind == kind:
                return v

    def require(self, *kinds: ParameterKind) -> None:
        '''
        Check that required parameters exist

        Parameters
        ----------
        *kind (ParameterKind): kinds of required parameters

        Authors: Allan Denis
        '''

        missing = [kind for kind in kinds if not self.has_kind(kind)]
        if missing:
            raise ForMoSAError(f"Missing required parameters: {', '.join(missing)}")

@dataclass
class ObservedModel:
    '''
    Model drawn from the nested sampling.

    Authors: Allan Denis
    '''

    # ======================
    # Attributes
    # ======================

    wave: np.ndarray                        # Wavelength array
    flux: np.ndarray                        # Planet signal / model flux
    res: np.ndarray | float                 # Spectral resolution
    component: np.ndarray | None = None     # Additive HC components (speckles, systematics, etc.)
    scaling: str = "analytic"               # Scaling method

    # ======================
    # Post-init checks
    # ======================

    def __post_init__(self):
        self.wave = np.asarray(self.wave, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        if self.component is None:
            self.component = np.zeros_like(self.flux)
        else:
            self.component = np.asarray(self.component, dtype=float)

        if self.wave.shape != self.flux.shape:
            raise ForMoSAError('wave and flux must have the same shape')

        if self.component.shape != self.flux.shape:
            raise ForMoSAError('component must have the same shape as flux')

    # ======================
    # Properties
    # ======================

    @property
    def total_flux(self) -> np.ndarray:                     # Total flux including HC components
        return self.flux + self.component

    @property
    def npts(self) -> int:                                  # Number of points
        return self.flux.size

    # ======================
    # Class methods
    # ======================

    @classmethod
    def from_grid_and_params(cls, grid: ModelGrid, params: ObservedParameters, interp_method: str = 'linear') -> 'ObservedModel':
        '''
        Build an instance of ObseredModel from a ModelGrid and an ObservedParameters objects.

        Parameters
        ----------
        grid            (ModelGrid): An instance of ModelGrid
        parmas (ObservedParameters): An instance of ObservedParameters
        interp_method         (str): Interpolation method

        Returns
        -------
        'ObservedModel': An instance of class ObservedModel

        Authors: Allan Denis
        '''

        # Initial checks
        if not isinstance(grid, ModelGrid):
            raise ForMoSAError(f'Wrong type for grid: {type(grid)}. Expected a ModelGrid')

        if not isinstance(params, ObservedParameters):
            raise ForMoSAError(f'Wrong type for params: {type(params)}. Expected an ObservedParameters')

        from ForMoSA.transform.apply_effects import ApplyPhysicsEffects

        grid_params = params.grid
        physics_params = params.physics

        model = grid._interpolate_between_gridpoints(grid_params.values_by_name, method=interp_method)
        observed_model = cls(model.coords['wavelength'], model.grid.values, model.attrs['res'])

        if np.any(np.isnan(observed_model.flux)):
            # NaNs produced because grid parameter is outside of the grid bounds, so we directly return the model
            return observed_model

        # ======================
        # RV
        # ======================

        if physics_params.has_kind(ParameterKind.RV):
            observed_model = ApplyPhysicsEffects._apply_rv(observed_model, physics_params.get_kind(ParameterKind.RV))

        # ======================
        # v.sini
        # ======================

        if physics_params.has_kind(ParameterKind.VSINI):
            vsini_param = physics_params.params_by_kind[ParameterKind.VSINI]

            observed_model = ApplyPhysicsEffects._apply_vsini(observed_model, physics_params.get_kind(ParameterKind.VSINI), physics_params.get_kind(ParameterKind.LD), vsini_param.vsini_function)

        # ======================
        # Reddening
        # ======================

        if physics_params.has_kind(ParameterKind.AV):
            observed_model = ApplyPhysicsEffects._apply_reddening(observed_model, physics_params.get_kind(ParameterKind.AV))

        # ======================
        # CPD
        # ======================

        if physics_params.has_kind(ParameterKind.BB_T):
            if not physics_params.has_kind(ParameterKind.BB_R):
                raise ForMoSAError(' Black Body radius is required when a Black Body temperature is given')

            if physics_params.has_kind(ParameterKind.DISTANCE):
                raise ForMoSAError(' Distance is required to add a CPD contribution')

            observed_model = ApplyPhysicsEffects._apply_cpd(observed_model, physics_params.get_kind(ParameterKind.BB_T), physics_params.get_kind(ParameterKind.BB_R), physics_params.get_kind(ParameterKind.DISTANCE))

        # ======================
        # Scaling (R, D, alpha)
        # ======================

        try:
            alpha = physics_params.get_kind(ParameterKind.ALPHA)
        except ForMoSAError:
            alpha = 1

        if physics_params.has_kind(ParameterKind.DISTANCE) and physics_params.has_kind(ParameterKind.RADIUS):
            observed_model = ApplyPhysicsEffects._apply_scaling(observed_model, alpha, physics_params.get_kind(ParameterKind.RADIUS), physics_params.get_kind(ParameterKind.DISTANCE))

        return observed_model

    # ======================
    # Methods
    # ======================

    def residuals(self, flux_obs: np.ndarray, component_only: bool = False) -> np.ndarray:
        '''
        Compute residuals between observation flux and the instance of ObservedModel.

        Parameters
        ----------
        flux_obs  (np.ndarray): Flux of the observations
        componant_only ( bool): Whether to use only the componant (without the flux) of the instance

        Returns
        -------
        np.ndarray: Residuals

        Authors: Allan Denis
        '''

        flux_obs = np.asarray(flux_obs, dtype=float)

        if flux_obs.size != self.npts:
            raise ForMoSAError(f'Flux of the observation must have the same number of points ({flux_obs.size}) than ObservedModel ({self.npts})')

        if component_only:
            return flux_obs - self.component
        else:
            return flux_obs - self.total_flux

    def std_residuals(self, flux_obs: np.ndarray) -> float:
        '''
        Compute the standard deviation of the residuals between observation flux and the instance of ObservedModel.

        Parameters
        ----------
        flux_obs  (np.ndarray): Flux of the observations

        Returns
        -------
        float: Standard deviation of the residuals

        Authors: Allan Denis
        '''

        if len(self.wave) == 1:
            return 1

        return np.std(self.residuals(flux_obs))

    def copy(self, **updates) -> "ObservedModel":
        """
        Return a modified copy of the ObservedModel.

        Parameters
        ----------
        **updates: Attributes to update

        Returns
        -------
        ObservedModel: Copy of ObservedModel

        Authors: Allan Denis
        """
        return replace(self, **updates)

    def _sort(self) -> None:
        '''
        Sort by increasing wavelength

        Authors: Allan Denis
        '''

        # Sort wave, flux, res and component
        isort = np.argsort(self.wave)
        self.wave, self.flux, self.res, self.component = self.wave[isort], self.flux[isort], self.res[isort], self.component[isort]