import numpy as np
import pytest

from ForMoSA.utils.spec import trapezoidal_integral, integrate_filter, resolution_decreasing

# These functions implement the actual model-grid adaptation math:
# - resolution_decreasing is called by SubGridSpectroscopy._adapt_model to resample a
#   model spectrum onto an observation's wavelength/resolution grid.
# - integrate_filter (via trapezoidal_integral) mirrors SubGridPhotometry._adapt_model's
#   filter-curve integration (that call site now shares trapezoidal_integral too).


def _equivalent_width(wav, flx):
    return np.sum(0.5 * ((1 - flx[1:]) + (1 - flx[:-1])) * np.diff(wav))


def _fwhm_subpixel(wav, flx):
    """FWHM of an absorption dip, with linear interpolation across the half-max
    crossing pixels so the measurement isn't limited to the wavelength grid's
    own sampling step."""
    depth = 1 - flx
    half = depth.max() / 2
    idx = np.where(depth >= half)[0]
    i0, i1 = idx[0], idx[-1]

    def _cross(ia, ib):
        da, db = depth[ia], depth[ib]
        t = (half - da) / (db - da)
        return wav[ia] + t * (wav[ib] - wav[ia])

    left = _cross(i0 - 1, i0) if i0 > 0 else wav[i0]
    right = _cross(i1, i1 + 1) if i1 < len(wav) - 1 else wav[i1]
    return right - left


# ==========================================================
# trapezoidal_integral
# ==========================================================

def test_trapezoidal_integral_of_constant():
    # integral of f(x)=1 over [0,4] = 4; trapezoid rule is exact for constants
    assert np.isclose(trapezoidal_integral(np.array([1.0, 1.0, 1.0]), np.array([0.0, 2.0, 4.0])), 4.0)


def test_trapezoidal_integral_of_linear_is_exact():
    # integral of f(x)=x over [0,4] = 8; trapezoid rule is exact for linear functions too
    assert np.isclose(trapezoidal_integral(np.array([0.0, 4.0]), np.array([0.0, 4.0])), 8.0)


def test_trapezoidal_integral_hand_computed():
    # x=[0,1,2], y=[0,2,0]: (0+2)/2*1 + (2+0)/2*1 = 2
    assert np.isclose(trapezoidal_integral(np.array([0.0, 2.0, 0.0]), np.array([0.0, 1.0, 2.0])), 2.0)


# ==========================================================
# integrate_filter (filter-weighted flux average)
# ==========================================================

def test_integrate_filter_constant_flux_returns_that_constant():
    """A constant flux, weighted-averaged by any transmission curve, must equal
    that same constant regardless of the filter's shape."""
    wav_filt = np.array([1.0, 1.3, 1.5, 1.8, 2.0])
    trans_filt = np.array([0.0, 0.6, 1.0, 0.3, 0.0])
    wav_input = np.linspace(0.5, 2.5, 200)
    flux_const = np.full_like(wav_input, 3.7)

    assert np.isclose(integrate_filter(wav_filt, trans_filt, wav_input, flux_const), 3.7)


def test_integrate_filter_symmetric_filter_and_linear_ramp():
    """Under a symmetric (triangular) weighting, the weighted average of a
    linear function equals the function's value at the filter's center."""
    wav_filt = np.array([1.0, 1.5, 2.0])
    trans_filt = np.array([0.0, 1.0, 0.0])
    wav_input = np.linspace(0.5, 2.5, 200)
    flux_ramp = 2.0 + 5.0 * wav_input

    expected = 2.0 + 5.0 * 1.5  # ramp value at the filter's center wavelength
    assert np.isclose(integrate_filter(wav_filt, trans_filt, wav_input, flux_ramp), expected)


def test_integrate_filter_zero_transmission_returns_nan():
    wav_filt = np.array([1.0, 1.5, 2.0])
    trans_filt = np.array([0.0, 0.0, 0.0])
    wav_input = np.linspace(0.5, 2.5, 50)
    flux = np.ones_like(wav_input)

    assert np.isnan(integrate_filter(wav_filt, trans_filt, wav_input, flux))


# ==========================================================
# resolution_decreasing (spectroscopic model-grid adaptation)
# ==========================================================

def test_resolution_decreasing_empty_input_returns_empty():
    result = resolution_decreasing(np.array([]), np.array([]), np.array([]), np.array([1.0]), np.array([100.0]))
    assert len(result) == 0


def test_resolution_decreasing_conserves_equivalent_width():
    """Convolution redistributes flux across a line but must conserve its
    equivalent width."""
    wav = np.linspace(2.0, 2.05, 3000)
    wav_center = 2.025
    sigma_line = 3e-5
    flx = 1.0 - 0.5 * np.exp(-(wav - wav_center) ** 2 / (2 * sigma_line ** 2))

    fwhm_line = 2.355 * sigma_line
    res_in = np.full_like(wav, wav_center / fwhm_line)
    res_out = np.full_like(wav, 15000.0)

    out = resolution_decreasing(wav, flx, res_in, wav, res_out)

    ew_in = _equivalent_width(wav, flx)
    ew_out = _equivalent_width(wav, out)
    assert np.isclose(ew_out, ew_in, rtol=1e-4)


def test_resolution_decreasing_output_width_matches_target_resolution():
    """
    If res_input is set self-consistently with the line's own intrinsic width
    (fwhm_line = 2.355*sigma_line), the quadrature-sum formula
    fwhm_conv = sqrt(fwhm_out^2 - fwhm_in^2) makes the line's own width cancel
    out exactly: the output FWHM should equal wav/res_output, independent of
    the input line's width, as long as res_input >= res_output.
    """
    wav = np.linspace(2.0, 2.05, 3000)
    wav_center = 2.025
    sigma_line = 3e-5
    flx = 1.0 - 0.5 * np.exp(-(wav - wav_center) ** 2 / (2 * sigma_line ** 2))

    fwhm_line = 2.355 * sigma_line
    res_in = np.full_like(wav, wav_center / fwhm_line)

    for res_out_val in (20000.0, 15000.0, 10000.0):
        res_out = np.full_like(wav, res_out_val)
        out = resolution_decreasing(wav, flx, res_in, wav, res_out)

        fwhm_measured = _fwhm_subpixel(wav, out)
        fwhm_predicted = wav_center / res_out_val
        assert np.isclose(fwhm_measured, fwhm_predicted, rtol=0.05)


def test_resolution_decreasing_no_op_when_resolutions_match():
    """Degrading to the same resolution the input already has should leave the
    spectrum effectively unchanged (fwhm_conv = 0)."""
    wav = np.linspace(2.0, 2.05, 500)
    flx = 1.0 - 0.5 * np.exp(-(wav - 2.025) ** 2 / (2 * 3e-4 ** 2))
    res = np.full_like(wav, 5000.0)

    out = resolution_decreasing(wav, flx, res, wav, res)

    assert np.allclose(out, flx, atol=1e-3)
