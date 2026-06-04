import pytest
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock
import xarray as xr

from ForMoSA.grid.model_grid import ModelGrid
from ForMoSA.grid.subgrid_set import SubGridSet
from ForMoSA.filter.filter import PhotometryFilter
from ForMoSA.grid.subgrid_spectroscopy import SubGridSpectroscopy
from ForMoSA.grid.subgrid_photometry import SubGridPhotometry
from ForMoSA.core.enums import WavelengthUnit, ObservationType
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.grid.subgrid_base import SubGrid


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def mock_model_grid():
    data = np.random.rand(30, 3, 2)
    coords = {
        'wavelength': np.linspace(1.0, 10.0, 30),
        'param1': [0, 1, 2],
        'param2': [10, 20]
    }
    attrs = {
        'key': ['param1', 'param2'],
        'title': ['Param 1', 'Param 2'],
        'res': np.ones(30),
        'wave_unit': str(WavelengthUnit.MICROMETER.unit),
        'unit': ['unit1', 'unit2'],
        'par': ['par1', 'par2'],
    }
    return ModelGrid._from_attributes(data, coords, attrs)


@pytest.fixture
def mock_spectro_subgrid(mock_model_grid):
    target_wavelength = np.linspace(2, 4, 5)
    target_resolution = np.ones(5) * 0.5
    wave_cont, res_cont = target_wavelength, 0.3
    return SubGridSpectroscopy.from_parent(
        parent_grid=mock_model_grid,
        target_wavelength=target_wavelength,
        target_resolution=target_resolution,
        remove_continuum=False,
        wave_cont=wave_cont,
        res_cont=res_cont
    )


@pytest.fixture
def mock_photo_subgrid(mock_model_grid):
    filt = PhotometryFilter('Keck', 'NIRC2', 'lp')
    return SubGridPhotometry.from_parent(mock_model_grid, np.array([filt]))


# ==================================================
# Tests
# ==================================================

def test_init_valid(mock_model_grid):
    sgset = SubGridSet(parent_grid=mock_model_grid)
    assert sgset.parent_grid is mock_model_grid
    assert sgset.is_empty


def test_init_invalid_parent_grid():
    with pytest.raises(ForMoSAError):
        SubGridSet(parent_grid="not a grid")


def test_add_subgrid(mock_model_grid, mock_spectro_subgrid):
    sgset = SubGridSet(parent_grid=mock_model_grid)
    sgset.add_subgrid(mock_spectro_subgrid)
    assert sgset.n_subgrids == 1
    assert not sgset.is_empty


def test_add_subgrid_invalid_type(mock_model_grid):
    sgset = SubGridSet(parent_grid=mock_model_grid)
    with pytest.raises(ForMoSAError):
        sgset.add_subgrid("not a subgrid")


def test_has_spectroscopy_and_photometry(mock_model_grid, mock_spectro_subgrid, mock_photo_subgrid):
    sgset = SubGridSet(parent_grid=mock_model_grid)
    sgset.add_subgrid(mock_spectro_subgrid)
    sgset.add_subgrid(mock_photo_subgrid)
    assert sgset.has_spectroscopy
    assert sgset.has_photometry


def test_subgrid_filters(mock_model_grid, mock_spectro_subgrid, mock_photo_subgrid):
    sgset = SubGridSet(parent_grid=mock_model_grid)
    sgset.add_subgrid(mock_spectro_subgrid)
    sgset.add_subgrid(mock_photo_subgrid)
    assert sgset.spectroscopic_subgrids == [mock_spectro_subgrid]
    assert sgset.photometric_subgrids == [mock_photo_subgrid]


def test_wavelength_range_empty(mock_model_grid):
    sgset = SubGridSet(parent_grid=mock_model_grid)
    assert sgset.wavelength_range == (0.0, 0.0)


def test_wavelength_range_non_empty(mock_model_grid, mock_spectro_subgrid, mock_photo_subgrid):
    sgset = SubGridSet(parent_grid=mock_model_grid)
    sgset.add_subgrid(mock_spectro_subgrid)
    sgset.add_subgrid(mock_photo_subgrid)
    assert sgset.wavelength_range == (2.0, 4.0)


def test_save_subgrids(mock_model_grid, mock_spectro_subgrid, mock_photo_subgrid):
    sgset = SubGridSet(parent_grid=mock_model_grid)
    sgset.add_subgrid(mock_spectro_subgrid)
    sgset.add_subgrid(mock_photo_subgrid)
    with tempfile.TemporaryDirectory() as tmp_dir:
        sgset.save_all(tmp_dir)
        reloaded = SubGridSet.from_path(tmp_dir, parent_grid=mock_model_grid)
        assert reloaded.has_spectroscopy
        assert reloaded.has_photometry
        assert reloaded.n_subgrids == 2

def test_add_subgrid_from_file(mock_model_grid):
    sgset = SubGridSet(parent_grid=mock_model_grid)
    fake_path = "fake_file.nc"

    with patch("ForMoSA.grid.subgrid_set.SubGrid.from_file", return_value=MagicMock()) as mock_loader:
        sgset.add_subgrid(fake_path)

        mock_loader.assert_called_once_with(fake_path, sgset.parent_grid, logger=sgset.logger)
        assert sgset.n_subgrids == 1

def test_add_subgrid_from_dataset(mock_model_grid):
    sgset = SubGridSet(parent_grid=mock_model_grid)
    dataset_mock = xr.Dataset(attrs={"grid_type": ObservationType.SPECTROSCOPIC.value})

    with patch("ForMoSA.grid.subgrid_set.SubGrid.from_dataset", return_value=MagicMock()) as mock_create:
        sgset.add_subgrid(dataset_mock)

        mock_create.assert_called_once_with(dataset_mock, sgset.parent_grid, logger=sgset.logger)
        assert sgset.n_subgrids == 1

def test_from_dataset_missing_grid_type(mock_model_grid):
    with pytest.raises(ForMoSAError):
        SubGrid.from_dataset(xr.Dataset(), mock_model_grid)  # no grid_type attr

def test_from_dataset_dispatches_by_type(mock_model_grid):
    dataset_s = xr.Dataset(attrs={"grid_type": ObservationType.SPECTROSCOPIC.value})
    dataset_p = xr.Dataset(attrs={"grid_type": ObservationType.PHOTOMETRIC.value})

    with patch("ForMoSA.grid.subgrid_spectroscopy.SubGridSpectroscopy.from_grid", return_value=MagicMock()) as mock_s:
        subgrid_s = SubGrid.from_dataset(dataset_s, mock_model_grid)
        mock_s.assert_called_once()
        assert subgrid_s is mock_s.return_value

    with patch("ForMoSA.grid.subgrid_photometry.SubGridPhotometry.from_grid", return_value=MagicMock()) as mock_p:
        subgrid_p = SubGrid.from_dataset(dataset_p, mock_model_grid)
        mock_p.assert_called_once()
        assert subgrid_p is mock_p.return_value

def test_from_dataset_invalid_type(mock_model_grid):
    dataset_mock = xr.Dataset(attrs={"grid_type": "unknown_type"})

    with pytest.raises(ForMoSAError):
        SubGrid.from_dataset(dataset_mock, mock_model_grid)

def test_add_subgrid_no_args(mock_model_grid):
    sgset = SubGridSet(parent_grid=mock_model_grid)
    with pytest.raises(ForMoSAError):
        sgset.add_subgrid()
