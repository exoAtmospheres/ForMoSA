import pytest
import numpy as np

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.enums import ParameterKind
from ForMoSA.parameter.prior import GaussianPrior
from ForMoSA.parameter.parameter import Parameter
from ForMoSA.transform.observed import ObservedModel, ObservedParameters

# ======================
# Fixtures
# ======================

@pytest.fixture
def wave():
    return np.array([1.0, 2.0, 3.0])

@pytest.fixture
def flux():
    return np.array([10.0, 20.0, 30.0])

@pytest.fixture
def component():
    return np.array([0.5, 0.5, 0.5])

@pytest.fixture
def parameters():

    p1 = Parameter("par1", GaussianPrior(1, 0.1), ParameterKind.GRID)
    p2 = Parameter("par2", GaussianPrior(1, 0.1), ParameterKind.R)
    return {p1: 1.0, p2: 2.0}

# ======================
# Tests ObservedModel
# ======================

def test_observed_model_init(wave, flux, component):
    model = ObservedModel(wave, flux, res=100, components=[component])
    np.testing.assert_array_equal(model.wave, wave)
    np.testing.assert_array_equal(model.flux, flux)
    np.testing.assert_array_equal(model.total_component, component)
    assert model.scaling == "analytic"

def test_observed_model_component_default(wave, flux):
    model = ObservedModel(wave, flux, res=100)
    np.testing.assert_array_equal(model.total_component, np.zeros_like(flux))

def test_observed_model_invalid_shapes(wave, flux):
    with pytest.raises(ForMoSAError):
        ObservedModel(wave[:-1], flux, res=100)
    with pytest.raises(ForMoSAError):
        ObservedModel(wave, flux, res=100, components=[np.array([1.0])])

def test_total_flux_property(wave, flux, component):
    model = ObservedModel(wave, flux, res=100, components=[component])
    expected = flux + component
    np.testing.assert_array_equal(model.total_flux, expected)

def test_npts_property(wave, flux):
    model = ObservedModel(wave, flux, res=100)
    assert model.npts == flux.size

def test_residuals_method(wave, flux, component):
    model = ObservedModel(wave, flux, res=100, components=[component])
    obs_flux = flux + component + 1.0
    res_all = model.residuals(obs_flux)
    res_component_only = model.residuals(obs_flux, component_only=True)
    np.testing.assert_array_equal(res_all, np.ones_like(flux))
    np.testing.assert_array_equal(res_component_only, np.ones_like(flux) + flux)

def test_residuals_invalid_size(wave, flux):
    model = ObservedModel(wave, flux, res=100)
    with pytest.raises(ForMoSAError):
        model.residuals(np.array([1.0, 2.0]))

def test_copy_method(wave, flux, component):
    model = ObservedModel(wave, flux, res=100, components=[component])
    copy_model = model.copy(flux=np.array([0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(copy_model.flux, np.zeros_like(flux))
    # Original unchanged
    np.testing.assert_array_equal(model.flux, flux)

# ======================
# Tests ObservedParameters
# ======================

def test_observed_parameters_init(parameters):
    obs_params = ObservedParameters(parameters)
    for key in parameters:
        assert key in obs_params.values

def test_observed_parameters_invalid_key():
    with pytest.raises(ForMoSAError):
        ObservedParameters({123: 1.0})

def test_names_and_kinds(parameters):
    obs_params = ObservedParameters(parameters)
    names = [p.name for p in parameters]
    kinds = [p.kind for p in parameters]
    assert obs_params.names == names
    assert obs_params.kinds == kinds

def test_has_name_and_kind(parameters):
    obs_params = ObservedParameters(parameters)
    param_keys = list(parameters.keys())
    assert obs_params.has_name(param_keys[0].name)
    assert obs_params.has_kind(param_keys[0].kind)
    assert not obs_params.has_name("nonexistent")

def test_get_method(parameters):
    obs_params = ObservedParameters(parameters)
    key = list(parameters.keys())[0]
    val = parameters[key]
    assert obs_params.get_name(key.name) == val
    with pytest.raises(ForMoSAError):
        obs_params.get_name("nonexistent")

def test_grid_and_physics_properties(parameters):
    obs_params = ObservedParameters(parameters)
    grid_params = {p: v for p, v in parameters.items() if p.kind == ParameterKind.GRID}
    physics_params = {p: v for p, v in parameters.items() if p.kind != ParameterKind.GRID}
    assert obs_params.grid.values == grid_params
    assert obs_params.physics.values == physics_params

def test_has_grid_and_physics(parameters):
    obs_params = ObservedParameters(parameters)
    assert obs_params.has_grid == any(p.kind == ParameterKind.GRID for p in parameters)
    assert obs_params.has_physics == any(p.kind != ParameterKind.GRID for p in parameters)
