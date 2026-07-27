import pytest
import numpy as np
import xarray as xr
import tempfile
from unittest.mock import patch, MagicMock

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.grid.model_grid import ModelGrid
from ForMoSA.filter.filter import PhotometryFilter
from ForMoSA.core.enums import WavelengthUnit, ObservationType
from ForMoSA.grid.subgrid_photometry import SubGridPhotometry
from ForMoSA.grid.subgrid_spectroscopy import SubGridSpectroscopy


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def minimal_grid_data():
    """Return minimal grid data, coords, and attributes."""
    data = np.random.rand(5, 3, 2)
    coords = {
        'wavelength': np.linspace(1.0, 2.0, 5),
        'param1': [0, 1, 2],
        'param2': [10, 20]
    }
    attrs = {
        'key': ['param1', 'param2'],
        'title': ['Param 1', 'Param 2'],
        'res': np.ones(5),
        'wave_unit': str(WavelengthUnit.MICROMETER.unit),
        'unit': ['unit1', 'unit2'],
        'par': ['par1', 'par2'],
    }
    return data, coords, attrs


@pytest.fixture
def model_grid(minimal_grid_data):
    """Return a minimal ModelGrid instance."""
    data, coords, attrs = minimal_grid_data
    return ModelGrid._from_attributes(data, coords, attrs)

@pytest.fixture
def minimal_spectro_dataset(model_grid):
    ds = model_grid.grid.copy()
    ds.attrs['grid_type'] = ObservationType.SPECTROSCOPIC.value
    ds.attrs['remove_continuum'] = True
    ds.attrs['wave_cont'] = np.array([1.0, 1.1])
    ds.attrs['res_cont'] = 100
    return ds

@pytest.fixture
def mock_filter():
    return PhotometryFilter('Keck', 'NIRC2', 'Lp')


# ==================================================
# Creation tests
# ==================================================

def test_creation_from_attributes(model_grid, minimal_grid_data):
    data, coords, attrs = minimal_grid_data
    assert isinstance(model_grid, ModelGrid)
    assert model_grid.grid['grid'].shape == (5, 3, 2)
    np.testing.assert_allclose(model_grid.grid.coords['wavelength'].values, coords['wavelength'])
    assert model_grid.keys == attrs['key']
    assert model_grid.titles == attrs['title']


def test_repr_and_grid_name(model_grid):
    repr_str = repr(model_grid)
    assert str(model_grid.grid_name) in repr_str
    assert "shape=" in repr_str


# ==================================================
# Unit conversion tests
# ==================================================

def test_wave_unit_conversion(model_grid):
    wave_default = model_grid.wave
    model_grid._set_unit(WavelengthUnit.NANOMETER)
    wave_nm = model_grid.wave
    np.testing.assert_allclose(wave_nm, wave_default * 1e3)


def test_invalid_unit_raises(model_grid):
    with pytest.raises(ForMoSAError):
        model_grid._set_unit("invalid_unit")


# ==================================================
# Accessing models
# ==================================================

def test_load_model_at_index(model_grid):
    idx = (1, 0)
    model = model_grid._load_model_at_specific_index(idx)
    assert isinstance(model, xr.DataArray)
    assert model.shape[0] == len(model_grid.wave)


def test_load_model_invalid_index(model_grid):
    with pytest.raises(ForMoSAError):
        model_grid._load_model_at_specific_index([0, 1])  # Not a tuple


# ==================================================
# Grid restriction
# ==================================================

def test_restricted_grid(model_grid):
    wmin, wmax = 1.2, 1.8
    restricted = model_grid._restricted_grid(f'{wmin}, {wmax}')
    assert isinstance(restricted, ModelGrid)
    assert restricted.grid.coords['wavelength'].min() >= wmin * 0.99
    assert restricted.grid.coords['wavelength'].max() <= wmax * 1.01
    np.testing.assert_allclose(
        restricted.grid.attrs['res'],
        model_grid.attrs['res'][
            (model_grid.grid.coords['wavelength'].values >= wmin * 0.99) &
            (model_grid.grid.coords['wavelength'].values <= wmax * 1.01)
        ]
    )


# ==================================================
# Interpolation tests
# ==================================================

def test_interpolate_between_gridpoints(model_grid):
    theta = {'param1': 1, 'param2': 15}
    model_grid._interpolate_between_gridpoints(theta)
    assert isinstance(model_grid.grid, xr.Dataset)


def test_interpolate_between_gridpoints_mismatch(model_grid):
    theta = {'params1': 10, 'param2': 15}  # Wrong param name
    with pytest.raises(ForMoSAError):
        model_grid._interpolate_between_gridpoints(theta)


# ==================================================
# Save/load tests
# ==================================================

def test_save_and_load_grid(model_grid):
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_grid.save_grid(tmp_dir)
        loaded_ds = model_grid._load_grid(tmp_dir)
    assert isinstance(loaded_ds, xr.Dataset)
    np.testing.assert_allclose(loaded_ds['grid'].values, model_grid.grid['grid'].values)


# ==================================================
# Nyquist and effective resolution
# ==================================================

def test_nyquist_and_effective_resolution(model_grid):
    nyquist = model_grid.nyquist
    eff_res = model_grid.effective_resolution
    assert len(nyquist) == len(model_grid.wave)
    assert len(eff_res) == len(model_grid.wave)
    assert np.all(eff_res <= model_grid.res)


# ==================================================
# Keys, titles, limits, key_values
# ==================================================

def test_keys_titles_lims_and_key_values(model_grid):
    assert set(model_grid.keys) == {'param1', 'param2'}
    assert set(model_grid.titles) == {'Param 1', 'Param 2'}
    kv = model_grid.key_values
    for k in model_grid.keys:
        assert k in kv
        assert np.all(np.isin(kv[k], model_grid.grid[k].values))
    lims = model_grid.lims_params_grid
    for k, (mn, mx) in lims.items():
        assert mn <= mx



# ==================================================
# Creation from parent
# ==================================================

def test_spectroscopy_from_parent(model_grid):
    target_wave = np.linspace(1.0, 2.0, 5)
    target_res = np.ones(5) * 0.5

    subgrid = SubGridSpectroscopy.from_parent(
        parent_grid=model_grid,
        target_wavelength=target_wave,
        target_resolution=target_res,
        remove_continuum=True,
        wave_cont=np.array([1.0, 1.1]),
        res_cont=50,
        name="spec_sub"
    )
    assert isinstance(subgrid, SubGridSpectroscopy)
    assert subgrid.remove_continuum is True
    assert (subgrid.wave_cont == np.array([1.0, 1.1])).all()

def test_photometry_from_parent(model_grid, mock_filter):
    model_grid.grid.coords['wavelength'] = np.linspace(1, 10, 5)
    subgrid = SubGridPhotometry.from_parent(model_grid, Filter=np.array([mock_filter]), name="phot_sub")
    assert isinstance(subgrid, SubGridPhotometry)
    assert subgrid.Filter[0].name == "Keck/NIRC2.Lp"


# ==================================================
# Creation from grid
# ==================================================

def test_spectroscopy_from_grid(minimal_spectro_dataset, model_grid):
    with patch("ForMoSA.grid.grid_loader.GridLoader._validate_model_grid_dataset"):
        subgrid = SubGridSpectroscopy.from_grid(minimal_spectro_dataset, parent_grid=model_grid)
    assert isinstance(subgrid, SubGridSpectroscopy)
    assert subgrid.remove_continuum is True

def test_spectroscopy_invalid_continuum_raises(model_grid):
    with pytest.raises(ForMoSAError):
        SubGridSpectroscopy(
            grid=model_grid.grid,
            parent_grid=model_grid,
            remove_continuum=True,
            wave_cont=None,
            res_cont=None
        )

def test_spectroscopy_adapt_sets_attributes(model_grid):
    target_wave = np.linspace(1.0, 2.0, 5)
    target_res = np.ones(5) * 0.5
    subgrid = SubGridSpectroscopy.from_parent(
        parent_grid=model_grid,
        target_wavelength=target_wave,
        target_resolution=target_res,
        remove_continuum=False
    )
    subgrid.adapt()
    assert 'remove_continuum' in subgrid._grid.attrs
    assert 'wave_cont' in subgrid._grid.attrs
    assert 'res_cont' in subgrid._grid.attrs

def test_photometry_from_grid(model_grid, mock_filter):
    ds = model_grid.grid.copy()
    ds.attrs['grid_type'] = ObservationType.PHOTOMETRIC.value
    ds.attrs['filter_name'] = "mock"
    with patch("ForMoSA.grid.grid_loader.GridLoader._validate_model_grid_dataset"):
        with patch("ForMoSA.filter.filter.PhotometryFilter._from_filter_name", return_value=mock_filter):
            subgrid = SubGridPhotometry.from_grid(ds, parent_grid=model_grid)
    assert isinstance(subgrid, SubGridPhotometry)
    assert subgrid.Filter[0].name == "Keck/NIRC2.Lp"

def test_photometry_invalid_filter_type_raises(model_grid):
    from types import SimpleNamespace
    invalid_filter = SimpleNamespace()
    with pytest.raises(ForMoSAError):
        SubGridPhotometry(model_grid.grid, model_grid, Filter=np.array([invalid_filter], dtype=object))
