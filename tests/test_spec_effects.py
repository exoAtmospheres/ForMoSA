import numpy as np
import pytest
import extinction
import astropy.constants as const

from ForMoSA.utils.spec import reddening_fct, vsini_fct
from ForMoSA.core.enums import VsiniFunction

# reddening_fct and vsini_fct both delegate to external, already-validated libraries
# (extinction.fm07 and PyAstronomy's rotBroad/fastRotBroad respectively). These tests
# check that ForMoSA wires them correctly (units, dispatch, edge cases, qualitative
# physical behavior), not that the underlying curves/kernels are themselves correct.


def _equivalent_width(wav, flx):
    """Trapezoidal integral of (1 - flux), without depending on a specific numpy
    trapz/trapezoid function name across numpy versions."""
    return np.sum(0.5 * ((1 - flx[1:]) + (1 - flx[:-1])) * np.diff(wav))


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def flat_spectrum():
    wav = np.linspace(0.5, 2.5, 50)  # microns
    flx = np.ones_like(wav)
    return wav, flx


@pytest.fixture
def gaussian_line():
    """A single absorption line on a flat continuum -- deterministic, no randomness."""
    wav = np.linspace(2.0, 2.05, 400)
    flx = 1.0 - 0.5 * np.exp(-(wav - 2.025) ** 2 / (2 * 0.0005 ** 2))
    res = np.full_like(wav, 50000.0)
    return wav, flx, res


# ==========================================================
# reddening_fct (interstellar extinction via extinction.fm07)
# ==========================================================

def test_reddening_zero_av_leaves_flux_unchanged(flat_spectrum):
    wav, flx = flat_spectrum
    flx_rd = reddening_fct(wav, flx, 0.0)
    np.testing.assert_array_equal(flx_rd, flx)


def test_reddening_empty_flux_passthrough():
    flx_rd = reddening_fct(np.array([]), np.array([]), 1.0)
    assert len(flx_rd) == 0


def test_reddening_matches_independent_unit_conversion(flat_spectrum):
    """
    Reconstructs the wavelength unit conversion (micron -> Angstrom) and the
    magnitude-to-flux conversion independently here, rather than delegating to
    reddening_fct itself, to verify ForMoSA wires extinction.fm07 correctly.
    """
    wav, flx = flat_spectrum
    av = 1.3
    expected_mag = extinction.fm07(wav * 10000, av, unit='aa')
    expected_flx = flx * 10 ** (-0.4 * expected_mag)

    assert np.allclose(reddening_fct(wav, flx, av), expected_flx)


def test_reddening_attenuates_more_at_bluer_wavelength(flat_spectrum):
    wav, flx = flat_spectrum
    flx_rd = reddening_fct(wav, flx, 1.0)
    assert flx_rd[0] < flx_rd[-1]


def test_reddening_scales_linearly_with_av(flat_spectrum):
    """The extinction curve shape is fixed by wavelength; only its amplitude
    scales with Av, so A(2*Av) = 2*A(Av) at every wavelength."""
    wav, flx = flat_spectrum
    dered_at_av1 = extinction.fm07(wav * 10000, 1.0, unit='aa')
    expected_at_av2 = flx * 10 ** (-0.4 * 2 * dered_at_av1)

    assert np.allclose(reddening_fct(wav, flx, 2.0), expected_at_av2)


# ==========================================================
# vsini_fct dispatcher (rotational broadening)
# ==========================================================

def test_vsini_zero_leaves_spectrum_unchanged(gaussian_line):
    wav, flx, res = gaussian_line
    flx_out, res_out = vsini_fct(
        wav, flx.copy(), res.copy(), ld_picked=0.6, vsini_picked=0.0,
        vsini_type=VsiniFunction.RotBroad.value,
    )
    np.testing.assert_array_equal(flx_out, flx)
    np.testing.assert_array_equal(res_out, res)


def test_vsini_empty_flux_passthrough():
    flx_out, res_out = vsini_fct(
        np.array([]), np.array([]), np.array([]),
        ld_picked=0.6, vsini_picked=20.0, vsini_type=VsiniFunction.RotBroad.value,
    )
    assert len(flx_out) == 0 and len(res_out) == 0


def test_vsini_updates_resolution_to_c_over_vsini(gaussian_line):
    wav, flx, res = gaussian_line
    vsini = 15.0
    _, res_out = vsini_fct(
        wav, flx.copy(), res.copy(), ld_picked=0.6, vsini_picked=vsini,
        vsini_type=VsiniFunction.RotBroad.value,
    )
    expected_res = const.c.to('km/s').value / vsini
    assert np.allclose(res_out, expected_res)


def test_vsini_unknown_type_raises(gaussian_line):
    wav, flx, res = gaussian_line
    with pytest.raises(ValueError):
        vsini_fct(wav, flx.copy(), res.copy(), ld_picked=0.6, vsini_picked=10.0, vsini_type="NotARealMethod")


@pytest.mark.parametrize("vsini_type", [
    VsiniFunction.RotBroad.value,
    VsiniFunction.FastRotBroad.value,
    VsiniFunction.Accurate.value,
    VsiniFunction.AccurateFast.value,
])
def test_vsini_broadening_shallows_the_line(gaussian_line, vsini_type):
    """Wiring check: every vsini_type must actually perform broadening -- the
    line should get shallower (flux minimum rises toward the continuum) as
    vsini increases, for all four dispatch targets."""
    wav, flx, res = gaussian_line
    depth_low = 1 - vsini_fct(wav, flx.copy(), res.copy(), 0.6, 10.0, vsini_type)[0].min()
    depth_high = 1 - vsini_fct(wav, flx.copy(), res.copy(), 0.6, 100.0, vsini_type)[0].min()
    assert depth_high < depth_low


def test_vsini_rot_broad_conserves_equivalent_width(gaussian_line):
    """RotBroad is the reference (non-approximate) implementation: rotational
    broadening redistributes flux across the line but should conserve
    equivalent width."""
    wav, flx, res = gaussian_line
    ew_in = _equivalent_width(wav, flx)
    flx_out, _ = vsini_fct(wav, flx.copy(), res.copy(), 0.6, 80.0, VsiniFunction.RotBroad.value)
    ew_out = _equivalent_width(wav, flx_out)
    assert np.isclose(ew_out, ew_in, rtol=1e-3)
