import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from ForMoSA.core.enums import ObservationType, WavelengthUnit
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.observation.observation_base import Observation
from ForMoSA.observation.observation_photometry import PhotometryObservation
from ForMoSA.observation.observation_spectroscopy import SpectralObservation

# ======================
# Fixtures
# ======================

@pytest.fixture
def spectral_data():
    wave = np.linspace(1.0, 2.0, 100)
    flux = np.random.randn(100)
    res = np.full(100, 1000.0)
    err = np.ones(100)
    return wave, flux, res, err

@pytest.fixture
def photometric_data():
    wave = np.array([3.7])
    flux = np.array([1.0])
    err = np.array([0.1])
    return wave, flux, err

@pytest.fixture
def target_resolution(spectral_data):
    wave, _, _, _ = spectral_data
    return np.full_like(wave, 30.0)

@pytest.fixture
def sample_data():
    return {
        "wave": np.array([1.0, 2.0, 3.0]),
        "flux": np.array([10.0, 20.0, 30.0]),
        "err": np.array([0.1, 0.2, 0.3]),
        "facility": "TEST",
        "instrument": "INST",
        "native_unit": WavelengthUnit.MICROMETER
    }

@pytest.fixture
def fake_path(tmp_path):
    return tmp_path / "fake_obs.npz"

# ======================
# Tests Observation base
# ======================

def test_observation_is_abstract():
    """Observation base class must not be instantiable."""
    with pytest.raises(TypeError):
        Observation()

# ======================
# Tests SpectralObservation
# ======================

def test_spectral_observation_creation(spectral_data):
    wave, flux, res, err = spectral_data
    obs = SpectralObservation(
        wave=wave,
        flux=flux,
        res=res,
        facility="TEST",
        instrument="TEST",
        err=err,
        native_unit=WavelengthUnit.MICROMETER,
    )
    assert obs.ObsType == ObservationType.SPECTROSCOPIC.obstype
    assert len(obs.wave) == len(obs.flux)
    assert len(obs.wave) == len(obs.res)

def test_spectral_observation_inconsistent_lengths(spectral_data):
    wave, flux, res, err = spectral_data
    flux_short = flux[:-10]
    err_short = err[:-10]
    with pytest.raises(ForMoSAError):
        SpectralObservation(
            wave=wave,
            flux=flux_short,
            res=res,
            facility="TEST",
            instrument="TEST",
            err=err_short,
            native_unit=WavelengthUnit.MICROMETER,
        )

def test_spectral_observation_positive_resolution(spectral_data):
    wave, flux, _, err = spectral_data
    res_negative = np.full(100, -100.0)
    with pytest.raises(ForMoSAError):
        SpectralObservation(
            wave=wave,
            flux=flux,
            res=res_negative,
            facility="TEST",
            instrument="TEST",
            err=err,
            native_unit=WavelengthUnit.MICROMETER,
        )

def test_spectral_observation_no_nan(spectral_data):
    wave, flux, res, err = spectral_data
    flux_with_nan = flux.copy()
    flux_with_nan[10] = np.nan
    obs = SpectralObservation(
        wave=wave,
        flux=flux_with_nan,
        res=res,
        facility="TEST",
        instrument="TEST",
        err=err,
        native_unit=WavelengthUnit.MICROMETER,
    )
    assert not np.isnan(obs.flux).any()

def test_restricted_spectral_observation(spectral_data):
    wave, flux, res, err = spectral_data
    obs = SpectralObservation(
        wave=wave,
        flux=flux,
        res=res,
        facility="TEST",
        instrument="TEST",
        err=err,
        native_unit=WavelengthUnit.MICROMETER,
    )
    windows = '1.1, 1.9'
    obs_restricted = obs._restricted_observation(windows)
    assert isinstance(obs_restricted, Observation)
    assert obs.n_points > obs_restricted.n_points
    assert min(obs.wave) < min(obs_restricted.wave) < max(obs_restricted.wave) < max(obs.wave)

# ======================
# Tests PhotometryObservation
# ======================

def test_photometric_observation_creation(photometric_data):
    wave, flux, err = photometric_data
    obs = PhotometryObservation(
        wave=wave,
        flux=flux,
        err=err,
        facility="Keck",
        instrument="NIRC2",
        filter_id="Lp",
        native_unit=WavelengthUnit.MICROMETER,
    )
    assert obs.ObsType == ObservationType.PHOTOMETRIC.obstype
    assert len(obs.wave) == len(obs.flux)

def test_photometric_observation_positive_errors(photometric_data):
    wave, flux, _ = photometric_data
    err_negative = np.array([-0.1])
    with pytest.raises(ForMoSAError):
        PhotometryObservation(
            wave=wave,
            flux=flux,
            err=err_negative,
            facility="Keck",
            instrument="NIRC2",
            filter_id="Lp",
            native_unit=WavelengthUnit.MICROMETER,
        )

# ======================
# Test Observation metadata
# ======================

def test_observation_metadata(spectral_data):
    wave, flux, res, err = spectral_data
    obs = SpectralObservation(
        wave=wave,
        flux=flux,
        res=res,
        facility="JWST",
        instrument="NIRSpec",
        err=err,
        native_unit=WavelengthUnit.MICROMETER,
    )
    assert obs.facility == "JWST"
    assert obs.instrument == "NIRSpec"

def test_adapt_to_resolution_basic(spectral_data, target_resolution):
    wave, flux, res, err = spectral_data

    obs = SpectralObservation(
        wave=wave,
        flux=flux,
        res=res,
        err=err,
        facility="TEST",
        instrument="TEST",
        native_unit=WavelengthUnit.MICROMETER,
    )

    adapted = obs._adapt_to_resolution(target_resolution)

    assert isinstance(adapted, Observation)
    assert adapted is not obs

    assert np.all(adapted.res <= obs.res)
    assert np.all(adapted.res == target_resolution)

    assert not np.allclose(adapted.flux, obs.flux)

    assert np.allclose(adapted.wave, obs.wave)

def test_adapt_to_resolution_is_capped(spectral_data):
    wave, flux, res, err = spectral_data
    target_res = np.full_like(res, 1e6)

    obs = SpectralObservation(
        wave=wave,
        flux=flux,
        res=res,
        err=err,
        facility="TEST",
        instrument="TEST",
        native_unit=WavelengthUnit.MICROMETER,
    )

    adapted = obs._adapt_to_resolution(target_res)

    assert np.all(adapted.res == obs.res)

def test_adapt_to_resolution_does_not_modify_original(spectral_data, target_resolution):
    wave, flux, res, err = spectral_data

    obs = SpectralObservation(
        wave=wave,
        flux=flux.copy(),
        res=res.copy(),
        err=err,
        facility="TEST",
        instrument="TEST",
        native_unit=WavelengthUnit.MICROMETER,
    )

    _ = obs._adapt_to_resolution(target_resolution)

    assert np.all(obs.res == res)
    assert np.all(obs.flux == flux)

def test_adapt_to_resolution_with_continuum(spectral_data, target_resolution):
    wave, flux, res, err = spectral_data

    obs = SpectralObservation(
        wave=wave,
        flux=flux,
        res=res,
        err=err,
        facility="TEST",
        instrument="TEST",
        native_unit=WavelengthUnit.MICROMETER,
    )

    adapted = obs._adapt_to_resolution(
        target_resolution,
        wave_cont="1.1,1.9",
        res_cont=50.0,
    )

    assert adapted.flux_cont is not None
    assert adapted.wave_cont == "1.1,1.9"
    assert adapted.res_cont == 50.0

    assert not np.allclose(adapted.flux, obs.flux)

def test_adapt_to_resolution_continuum_without_wave_cont(spectral_data, target_resolution):
    wave, flux, res, err = spectral_data

    obs = SpectralObservation(
        wave=wave,
        flux=flux,
        res=res,
        err=err,
        facility="TEST",
        instrument="TEST",
        native_unit=WavelengthUnit.MICROMETER,
    )

    adapted = obs._adapt_to_resolution(
        target_resolution,
        res_cont=100.0,
    )

    assert adapted.flux_cont is not None
    assert adapted.wave_cont is not None

def test_adapt_to_resolution_high_contrast(spectral_data, target_resolution):
    wave, flux, res, err = spectral_data
    star_flux = np.random.randn(len(wave))

    obs = SpectralObservation(
        wave=wave,
        flux=flux,
        res=res,
        err=err,
        star_flux=star_flux,
        facility="TEST",
        instrument="TEST",
        native_unit=WavelengthUnit.MICROMETER,
    )

    adapted = obs._adapt_to_resolution(target_resolution)

    assert adapted.hc_mode is True
    assert adapted.star_flux.shape == (len(wave), 1)
    assert not np.allclose(adapted.star_flux, star_flux)


def test_from_dict_calls_loader(sample_data):
    with patch("ForMoSA.observation.observation_loader.ObservationLoader") as mock_loader:
        mock_instance = MagicMock()
        mock_loader._from_data.return_value = mock_instance

        from ForMoSA.observation.observation_base import Observation
        obs = Observation._from_dict(sample_data, log_level="DEBUG")

        mock_loader._from_data.assert_called_once_with(sample_data, logger=None, log_level="DEBUG")
        assert obs is mock_instance

def test_from_attributes_calls_loader(sample_data):
    with patch("ForMoSA.observation.observation_loader.ObservationLoader") as mock_loader:
        mock_instance = MagicMock()
        mock_loader._from_attributes.return_value = mock_instance

        from ForMoSA.observation.observation_base import Observation
        obs = Observation._from_attributes(**sample_data)

        mock_loader._from_attributes.assert_called_once_with(**sample_data)
        assert obs is mock_instance

def test_from_file_calls_loader(fake_path):
    with patch("ForMoSA.observation.observation_loader.ObservationLoader") as mock_loader:
        mock_instance = MagicMock()
        mock_loader._from_fits.return_value = mock_instance

        from ForMoSA.observation.observation_base import Observation
        obs = Observation._from_file(fake_path, log_level="INFO")

        mock_loader._from_fits.assert_called_once_with(fake_path, logger=None, log_level="INFO")
        assert obs is mock_instance


