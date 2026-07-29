import numpy as np
import pytest

from ForMoSA import Analysis
from ForMoSA.core.enums import WavelengthUnit
from ForMoSA.grid.model_grid import ModelGrid
from ForMoSA.observation.observation_spectroscopy import SpectralObservation
from ForMoSA.config.global_config import (
    ConfigPath, ConfigAdapt, ConfigInversion, ConfigParameters, Config_NS, ConfigNestle,
)

# End-to-end smoke test for the Analysis.adapt -> nested_sampling -> plot pipeline.
# Uses a small, deterministic, in-memory model grid (not a real physics grid --
# those are large external files not shipped in this repo) and a noiseless synthetic
# observation generated from that same grid at a known injected parameter, so
# recovery can be checked against a known truth. par2 is held fixed at its true
# value so the free-parameter recovery isn't confounded by a shape degeneracy
# between par1 and par2 over the test's narrow wavelength window.
#
# nestle is used as the sampler since it's pure Python and needs no external
# binary (unlike PyMultiNest/UltraNest), keeping this fast and dependency-light.

PAR1_TRUE = 1.3
PAR2_TRUE = 0.12
AMPLITUDE_TRUE = 2.5  # arbitrary; must be absorbed by the pipeline's analytic scaling


def _model_flux(wave, par1, par2):
    return np.exp(-par1 * wave) * (1.0 + par2 * wave)


@pytest.fixture(scope="module")
def fitted_analysis(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("analysis_smoke")

    # ---- synthetic model grid ----
    wave = np.linspace(1.0, 2.0, 30)
    par1_grid = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    par2_grid = np.array([0.0, 0.1, 0.2])

    data = np.zeros((len(wave), len(par1_grid), len(par2_grid)))
    for i, p1 in enumerate(par1_grid):
        for j, p2 in enumerate(par2_grid):
            data[:, i, j] = _model_flux(wave, p1, p2)

    coords = {'wavelength': wave, 'par1': par1_grid, 'par2': par2_grid}
    attrs = {
        'key': ['par1', 'par2'],
        'title': ['Par1', 'Par2'],
        'res': np.full(len(wave), 5000.0),
        'wave_unit': str(WavelengthUnit.MICROMETER.unit),
        'unit': ['unit1', 'unit2'],
        'par': ['par1', 'par2'],
    }

    grid = ModelGrid._from_attributes(data, coords, attrs)
    grid_path = tmp_dir / "model_grid.nc"
    grid.grid.to_netcdf(grid_path, format="NETCDF4", engine="netcdf4", mode="w")

    # ---- synthetic noiseless observation, injected at a known truth ----
    wave_obs = np.linspace(1.05, 1.95, 25)
    flux_obs = AMPLITUDE_TRUE * _model_flux(wave_obs, PAR1_TRUE, PAR2_TRUE)
    err_obs = 0.02 * np.abs(flux_obs)
    res_obs = np.full_like(wave_obs, 2000.0)

    obs = SpectralObservation(
        wave=wave_obs, flux=flux_obs, err=err_obs, res=res_obs,
        facility=np.full(wave_obs.shape, "TEST"), instrument=np.full(wave_obs.shape, "TEST"),
        native_unit=WavelengthUnit.MICROMETER, name="synthetic_target",
    )
    obs_dir = tmp_dir / "obs"
    obs.save_observation(obs_dir, file_format="fits")
    obs_files = list(obs_dir.glob("*.fits"))

    # ---- run the pipeline ----
    result_path = tmp_dir / "results"
    config_path = ConfigPath(
        observation_path=[str(f) for f in obs_files],
        adapt_store_path=str(tmp_dir / "adapt"),
        result_path=str(result_path),
        model_path=str(grid_path),
    )

    analysis = Analysis(config_path)

    config_adapt = ConfigAdapt(backend="sequential", n_jobs=1)
    config_inversion = ConfigInversion(ns_algo="nestle", npoints="40", wav_fit=["1.05, 1.95"])
    analysis.adapt(config_adapt, config_inversion)

    config_params = ConfigParameters(par1=["uniform", "0.5", "2.5"], par2=["constant", str(PAR2_TRUE)])
    config_ns = Config_NS(nestle=ConfigNestle(method="single", maxiter="3000", dlogz="0.5"))
    analysis.nested_sampling(config_params, config_adapt, config_inversion, config_ns)

    return analysis, result_path


def test_analysis_adapt_and_fit_complete(fitted_analysis):
    analysis, _ = fitted_analysis
    assert analysis.adapted is True
    assert analysis.fitted is True
    assert analysis.ns is not None
    assert analysis.ns.results is not None
    assert np.isfinite(analysis.ns.results.best_logL)


def test_analysis_recovers_injected_parameter(fitted_analysis):
    """40 live points / dlogz=0.5 won't converge tightly, and a coarse 0.5-spaced
    par1 grid introduces some interpolation bias -- rtol=0.15 comfortably covers
    both while still catching a badly broken pipeline (wrong sign, wrong order of
    magnitude, wrong parameter recovered)."""
    analysis, _ = fitted_analysis
    median = analysis.ns.results.median_parameters
    assert np.isclose(median["Par1"], PAR1_TRUE, rtol=0.15)


def test_analysis_plot_produces_all_figures(fitted_analysis):
    analysis, result_path = fitted_analysis
    analysis.plot(analysis.ns.results, save=True)

    for filename in ("corner.pdf", "chains.pdf", "radar.pdf", "best_fit.pdf"):
        path = result_path / filename
        assert path.exists()
        assert path.stat().st_size > 0
