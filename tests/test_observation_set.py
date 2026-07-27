import pytest
import numpy as np
import tempfile

from ForMoSA.core.errors import ForMoSAError
from ForMoSA.core.enums import WavelengthUnit
from ForMoSA.observation.observation_spectroscopy import SpectralObservation
from ForMoSA.observation.observation_photometry import PhotometryObservation
from ForMoSA.observation.observation_set import ObservationSet

# ======================
# Fixtures
# ======================

@pytest.fixture
def obs_set():
    """Return a fresh ObservationSet instance for each test."""
    return ObservationSet()


# ======================
# Tests adding observations
# ======================

def test_add_spectral_observation_from_dict(obs_set):
    """Add a spectral observation using a dictionary."""
    data = {
        "wave": [400, 500, 600],
        "flux": [1.0, 1.1, 1.2],
        "res": [10, 10, 10],
        "err": [1, 1, 1],
        "facility": ["test", "test", "test"],
        "instrument": ["test", "test", "test"],
        "native_unit": WavelengthUnit.MICROMETER
    }

    obs_set.add_observation(data)

    assert len(obs_set) == 1
    assert isinstance(obs_set[0], SpectralObservation)
    np.testing.assert_array_equal(obs_set[0]._wave, np.array([400, 500, 600]))


def test_add_spectral_observation_from_attributes(obs_set):
    """Add a spectral observation using direct attributes."""
    obs_set.add_observation(
        wave=[400, 500, 600],
        flux=[1.0, 1.1, 1.2],
        res=[10, 10, 10],
        err=[1, 1, 1],
        facility=['Test', 'Test', 'Test'],
        instrument=['Test', 'Test', 'Test'],
        native_unit=WavelengthUnit.MICROMETER
    )

    assert len(obs_set) == 1
    assert isinstance(obs_set[0], SpectralObservation)
    np.testing.assert_array_equal(obs_set[0]._wave, np.array([400, 500, 600]))


def test_add_photometric_observation_from_dict(obs_set):
    """Add a photometric observation using a dictionary."""
    data = {
        "filter_id": "Lp",
        "flux": [1.0],
        "wave": [3.7],
        "err": [0.2],
        "native_unit": WavelengthUnit.MICROMETER,
        "facility": 'Keck',
        "instrument": 'NIRC2'
    }

    obs_set.add_observation(data)

    assert len(obs_set) == 1
    assert isinstance(obs_set[0], PhotometryObservation)
    assert obs_set[0].filter_id == "Lp"


# ======================
# Tests invalid data
# ======================

def test_invalid_data_raises_error(obs_set):
    """Adding observation with missing keys should raise an error."""
    invalid_data = {"some_key": "value"}
    with pytest.raises(ForMoSAError):
        obs_set.add_observation(invalid_data)


def test_invalid_attributes_raises_error(obs_set):
    """Adding observation with incomplete attributes should raise an error."""
    with pytest.raises(ForMoSAError):
        obs_set.add_observation(flux=[1.0, 1.1])


# ======================
# Test multiple observations
# ======================

def test_multiple_observations(obs_set):
    """Add multiple observations of different types."""
    # Add spectral observation
    obs_set.add_observation(
        wave=[0.4, 0.5, 0.6],
        flux=[1.0, 1.1, 1.2],
        res=[10, 10, 10],
        err=[0.1, 0.1, 0.1],
        facility=['Test', 'Test', 'Test'],
        instrument=['Test', 'Test', 'Test'],
        native_unit=WavelengthUnit.MICROMETER
    )

    # Add photometric observation
    data = {
        "filter_id": "Lp",
        "flux": [1.0],
        "wave": [3.7],
        "err": [0.2],
        "native_unit": WavelengthUnit.MICROMETER,
        "facility": 'Keck',
        "instrument": 'NIRC2'
    }
    obs_set.add_observation(data)

    assert len(obs_set) == 2
    assert isinstance(obs_set[0], SpectralObservation)
    assert isinstance(obs_set[1], PhotometryObservation)


# ======================
# Test saving and loading
# ======================

def test_save_and_load_observations(obs_set):
    """Save and reload observations using a temporary directory."""
    # Add spectral observation
    obs_set.add_observation(
        wave=[0.4, 0.5, 0.6],
        flux=[1.0, 1.1, 1.2],
        res=[10, 10, 10],
        err=[0.1, 0.1, 0.1],
        facility=['Test', 'Test', 'Test'],
        instrument=['Test', 'Test', 'Test'],
        native_unit=WavelengthUnit.MICROMETER
    )

    # Add photometric observation
    data = {
        "filter_id": "Lp",
        "flux": [1.0],
        "wave": [3.7],
        "err": [0.2],
        "native_unit": WavelengthUnit.MICROMETER,
        "facility": 'Keck',
        "instrument": 'NIRC2'
    }
    obs_set.add_observation(data)

    with tempfile.TemporaryDirectory() as tmp_dir:
        obs_set.save_all(tmp_dir)
        reloaded = ObservationSet.from_npz(tmp_dir)

        assert len(reloaded) == 2
        assert isinstance(reloaded[0], SpectralObservation)
        assert isinstance(reloaded[1], PhotometryObservation)


# ======================
# Test adapting observations
# ======================

def test_adapt_observations(obs_set):
    """Test adapting observations to new grids or windows."""
    # Add spectral observation
    obs_set.add_observation(
        wave=np.linspace(1, 2, 100),
        flux=np.ones(100),
        err=np.ones(100)*0.1,
        res=np.ones(100)*10,
        facility=np.full(100, 'Test'),
        instrument=np.full(100, 'Test'),
        native_unit=WavelengthUnit.MICROMETER
    )

    # Add photometric observation
    data = {
        "filter_id": "Lp",
        "flux": [1.0],
        "wave": [3.7],
        "err": [0.2],
        "native_unit": WavelengthUnit.MICROMETER,
        "facility": 'Keck',
        "instrument": 'NIRC2'
    }
    obs_set.add_observation(data)

    obs_set.adapt_all(
        [2, 0],
        ['1.1, 1.2 / 1.9, 2', '0'],
        [1, 0]
    )

    assert isinstance(obs_set.observations[0], SpectralObservation)
    assert isinstance(obs_set.observations[1], PhotometryObservation)
    assert obs_set.observations[0].wave_cont == '1.1, 1.2 / 1.9, 2'
