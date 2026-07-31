import numpy as np

from ForMoSA.utils.logL_functions import (
    logL_chi2,
    logL_chi2_covariance,
    logL_chi2_noisescaling,
    logL_chi2_noisescaling_covariance,
    logL_CCF_Brogi,
    logL_CCF_Zucker,
    logL_CCF_custom,
)

# All expected values below are derived independently from the textbook Gaussian
# log-likelihood, not by calling the functions under test, so a mismatch means the
# implementation is wrong rather than the test being miscalibrated.


# ==========================================================
# logL_chi2
# ==========================================================

def test_logL_chi2_basic():
    delta_flx = np.array([1.0, -1.0, 2.0])
    err = np.array([1.0, 1.0, 2.0])
    # chi2 = (1/1)^2 + (-1/1)^2 + (2/2)^2 = 3
    assert np.isclose(logL_chi2(delta_flx, err), -1.5)


def test_logL_chi2_ignores_nan():
    delta_flx = np.array([1.0, np.nan, 2.0])
    err = np.array([1.0, 1.0, 2.0])
    # nansum drops the masked point: chi2 = 1^2 + 1^2 = 2
    assert np.isclose(logL_chi2(delta_flx, err), -1.0)


def test_logL_chi2_full_matches_gaussian_normalization():
    """
    full=True is documented as adding "the usual constant terms", i.e. the standard
    Gaussian log-likelihood normalization for independent, heteroscedastic noise:
        logL = -N/2 * ln(2*pi) - 1/2 * sum(ln(err_i^2)) - chi2/2
    This is a different quantity from -1/2 * ln(sum(err_i^2)), which is what the
    current implementation computes (see np.log(np.dot(err, err)) in the source).
    For N=1 the two forms coincide, so use N=3 to distinguish them.
    """
    delta_flx = np.array([1.0, -1.0, 2.0])
    err = np.array([1.0, 1.0, 2.0])
    N = len(delta_flx)

    correct = -N / 2 * np.log(2 * np.pi) - 0.5 * np.sum(np.log(err ** 2)) - 1.5

    assert np.isclose(logL_chi2(delta_flx, err, full=True), correct)


# ==========================================================
# logL_chi2_covariance
# ==========================================================

def test_logL_chi2_covariance_diagonal_matches_logL_chi2():
    """A diagonal covariance matrix must reduce exactly to the plain chi2 case."""
    delta_flx = np.array([1.0, -1.0, 2.0])
    err = np.array([1.0, 1.0, 2.0])
    cov = np.diag(err ** 2)
    inv_cov = np.diag(1 / err ** 2)

    assert np.isclose(
        logL_chi2_covariance(delta_flx, cov, inv_cov),
        logL_chi2(delta_flx, err),
    )


def test_logL_chi2_covariance_full_matches_gaussian_normalization():
    """
    Unlike logL_chi2, this implementation uses np.linalg.slogdet(cov) for the
    normalization constant, which is the mathematically correct log-determinant
    term -- so it should already agree with the textbook formula exactly.
    """
    delta_flx = np.array([1.0, -1.0, 2.0])
    err = np.array([1.0, 1.0, 2.0])
    cov = np.diag(err ** 2)
    inv_cov = np.diag(1 / err ** 2)
    N = len(delta_flx)

    _, logdet = np.linalg.slogdet(cov)
    chi2 = delta_flx @ inv_cov @ delta_flx
    correct = -N / 2 * np.log(2 * np.pi) - 0.5 * logdet - chi2 / 2

    assert np.isclose(logL_chi2_covariance(delta_flx, cov, inv_cov, full=True), correct)


def test_logL_chi2_covariance_correlated_noise():
    """Non-diagonal covariance: exercises the spectrally-correlated-noise path."""
    delta_flx = np.array([1.0, 1.0])
    cov = np.array([[2.0, 1.0], [1.0, 2.0]])
    inv_cov = np.linalg.inv(cov)

    # chi2 = delta^T . inv_cov . delta = 2/3
    assert np.isclose(
        logL_chi2_covariance(delta_flx, cov, inv_cov),
        -1 / 3,
    )


# ==========================================================
# logL_chi2_noisescaling
# ==========================================================

def test_logL_chi2_noisescaling_basic():
    delta_flx = np.array([1.0, -1.0, 2.0])
    err = np.array([1.0, 1.0, 2.0])
    # chi2 = 3, N = 3 => chi2/N = 1 => log(1) = 0
    assert np.isclose(logL_chi2_noisescaling(delta_flx, err), 0.0)


def test_logL_chi2_noisescaling_scales_with_residual_amplitude():
    err = np.array([1.0, 1.0, 2.0])
    delta_flx = np.array([2.0, -2.0, 4.0])
    N = 3
    chi2 = np.sum((delta_flx / err) ** 2)  # = 12
    expected = -N / 2 * np.log(chi2 / N)
    assert np.isclose(logL_chi2_noisescaling(delta_flx, err), expected)


def test_logL_chi2_noisescaling_full_matches_gaussian_normalization():
    """Same normalization-constant question as test_logL_chi2_full_*, carried
    through the noise-scaling-marginalized profile likelihood."""
    err = np.array([1.0, 1.0, 2.0])
    delta_flx = np.array([2.0, -2.0, 4.0])
    N = 3
    chi2 = np.sum((delta_flx / err) ** 2)
    base = -N / 2 * np.log(chi2 / N)
    correct = base - N / 2 - N / 2 * np.log(2 * np.pi) - 0.5 * np.sum(np.log(err ** 2))

    assert np.isclose(logL_chi2_noisescaling(delta_flx, err, full=True), correct)


# ==========================================================
# logL_chi2_noisescaling_covariance
# ==========================================================

def test_logL_chi2_noisescaling_covariance_diagonal_matches_noisescaling():
    delta_flx = np.array([1.0, -1.0, 2.0])
    err = np.array([1.0, 1.0, 2.0])
    cov = np.diag(err ** 2)
    inv_cov = np.diag(1 / err ** 2)

    assert np.isclose(
        logL_chi2_noisescaling_covariance(delta_flx, cov, inv_cov),
        logL_chi2_noisescaling(delta_flx, err),
    )


# ==========================================================
# logL_CCF_Brogi (Brogi et al. 2019)
# ==========================================================

def test_logL_CCF_Brogi_known_value():
    """
    Note: this function mean-subtracts and normalizes its inputs *in place*, so
    fresh arrays are passed here to avoid aliasing between test cases.
    """
    flx_obs = np.array([0.0, 1.0, 2.0])
    flx_mod = np.array([2.0, 1.0, 0.0])

    # Derived analytically: after centering+normalizing, Sf2 = Sg2 = 1/3, R = -1/3,
    # so logL = -N/2 * ln(4/3).
    expected = -1.5 * np.log(4 / 3)

    assert np.isclose(logL_CCF_Brogi(flx_obs, flx_mod), expected)


def test_logL_CCF_Brogi_mutates_inputs_in_place():
    """Documents a real side effect: callers must not reuse the same array
    across multiple calls (e.g. looping over an RV grid) without copying."""
    flx_obs = np.array([0.0, 1.0, 2.0])
    flx_mod = np.array([2.0, 1.0, 0.0])

    logL_CCF_Brogi(flx_obs, flx_mod)

    assert not np.array_equal(flx_obs, np.array([0.0, 1.0, 2.0]))
    assert not np.array_equal(flx_mod, np.array([2.0, 1.0, 0.0]))


# ==========================================================
# logL_CCF_Zucker (Zucker 2003)
# ==========================================================

def test_logL_CCF_Zucker_known_value():
    flx_obs = np.array([1.0, 0.0])
    flx_mod = np.array([1.0, 1.0])

    # C2 = R^2 / (Sf2 * Sg2) is the squared cosine between the raw vectors here
    # (no centering/normalization in this variant): cos^2(45 deg) = 0.5.
    expected = -1.0 * np.log(0.5)  # = ln(2)

    assert np.isclose(logL_CCF_Zucker(flx_obs, flx_mod), expected)


def test_logL_CCF_Zucker_orthogonal_gives_zero():
    """Uncorrelated vectors (C2 = 0) should give logL = -N/2 * ln(1) = 0 exactly."""
    flx_obs = np.array([1.0, 0.0])
    flx_mod = np.array([0.0, 1.0])

    assert np.isclose(logL_CCF_Zucker(flx_obs, flx_mod), 0.0)


# ==========================================================
# logL_CCF_custom
# ==========================================================

def test_logL_CCF_custom_known_value():
    flx_obs = np.array([1.0, 0.0])
    flx_mod = np.array([1.0, 1.0])
    err_obs = np.array([1.0, 1.0])

    # Sf2=0.5, Sg2=1.0, R=0.5, sigma2_weight=1 => logL = -1 * 0.5 = -0.5
    assert np.isclose(logL_CCF_custom(flx_obs, flx_mod, err_obs), -0.5)


def test_logL_CCF_custom_weights_by_inverse_variance():
    """Same flux vectors as above but heteroscedastic errors: sigma2_weight is the
    harmonic-mean-like combination 1 / mean(1/err^2), not a plain average."""
    flx_obs = np.array([1.0, 0.0])
    flx_mod = np.array([1.0, 1.0])
    err_obs = np.array([1.0, 2.0])

    sigma2_weight = 1 / np.mean(1 / err_obs ** 2)  # = 1.6
    assert np.isclose(sigma2_weight, 1.6)
    assert np.isclose(logL_CCF_custom(flx_obs, flx_mod, err_obs), -0.3125)
