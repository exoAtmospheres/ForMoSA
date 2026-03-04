import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import extinction
import astropy.units as u
import astropy.constants as const
from PyAstronomy.pyasl import rotBroad, fastRotBroad
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from ForMoSA.core.enums import VsiniFunction
from scipy import optimize


def calc_ck(flx_mod: np.ndarray, flx_obs: np.ndarray, err_obs: np.ndarray, r_picked: float, d_picked: float, analytic: str = 'no', bounds: tuple[float, float] = (-float('inf'), float('inf'))) -> tuple[np.ndarray, float]:
    '''
    Calculation of the dilution factor Ck and re-normalization of the synthetic spectrum.

    Parameters
    ----------
    flx_mod : np.ndarray
        Flux of the model spectrum
    flx_obs : np.ndarray
        Flux of the data
    err_obs : np.ndarray
        Error of the data
    r_picked : float
        Radius (Rjup)
    d_picked : float
        Distance (pc)
    analytic : str
        If 'yes', compute Ck by linear least-squares fitting, else use alpha*(R/d)^2
    bounds : tuple[float, float]
        Bounds for the Least Squares

    Returns
    -------
    flx_mod_ck : np.ndarray
        Scaled model flux
    ck : float
        Scaling coefficient

    Authors
    -------
    Simon Petrus and Allan Denis
'''

    # Fixed analytical scaling
    if analytic == 'no':
        r_picked *= u.Rjup
        d_picked *= u.pc
        ck = (r_picked.to(u.m).value/d_picked.to(u.m).value)**2
        flx_mod_ck = flx_mod * ck
    else:
        # Fit mode using fit_linear_model
        A = [flx_mod]
        b = flx_obs

        # Solve via fit_linear_model
        result = fit_linear_model(components=A, flx_obs=b, err_obs=err_obs, bounds=bounds)

        flx_mod_ck = result['model']
        ck = result['coeffs'][0]

    return flx_mod_ck, ck


# ----------------------------------------------------------------------------------------------------------------------

def convolve_and_sample(wv_channels: list, sigmas_wvs: list, model_wvs: np.ndarray, model_fluxes: np.ndarray, num_sigma: int=3, force_int: bool=True) -> np.ndarray: # num_sigma = 3 is a good compromise between sampling enough the gaussian and fast interpolation
    """
    Simulate the observations of a model. Convolves the model with a variable Gaussian LSF, sampled at each desired
    spectral channel.

    Parameters
    ----------
        wv_channels : list(floats)
            the wavelengths values desired
        sigmas_wvs : list(floats)
            the LSF gaussian standard deviation of each wv_channels [IN UNITS OF model_wvs]
        model_wvs : array
            the wavelengths of the model
        model_fluxes : array
            the fluxes of the model
        num_sigma : int
            number of +/- sigmas to evaluate the LSF to.
        force_int : bolean
            False by default. If True, will force interpolation onto wv_channels when the kernel is singular
    Returns
    -------
        array
            the fluxes in each of the wavelength channels

    Authors
    -------
    Jason Wang
"""

    model_in_range = np.where((model_wvs >= np.min(wv_channels)) & (model_wvs < np.max(wv_channels)))
    dwv_model = np.abs(model_wvs[model_in_range] - np.roll(model_wvs[model_in_range], 1))
    dwv_model[0] = dwv_model[1]
    filter_size = int(np.ceil(np.max((2 * num_sigma * sigmas_wvs) / np.min(dwv_model))))
    filter_coords = np.linspace(-num_sigma, num_sigma, filter_size)
    filter_coords = np.tile(filter_coords, [wv_channels.shape[0], 1])  # shape of (N_output, filter_size)
    filter_wv_coords = filter_coords * sigmas_wvs[:, None] + wv_channels[:, None]  # model wavelengths we want

    lsf = np.exp(-filter_coords ** 2 / 2) / np.sqrt(2 * np.pi)
    left_fill = model_fluxes[model_in_range][0]
    right_fill = model_fluxes[model_in_range][-1]
    model_interp = interp1d(model_wvs, model_fluxes, kind='cubic', bounds_error=False, fill_value=(left_fill,right_fill))

    if np.sum(lsf) != 0:
        filter_model = model_interp(filter_wv_coords)
        output_model = np.nansum(filter_model * lsf, axis=1) / np.sum(lsf, axis=1)
    else:
        if force_int == True:
            output_model = model_interp(wv_channels)
        else:
            output_model = model_fluxes

    return output_model


# ----------------------------------------------------------------------------------------------------------------------


def resolution_decreasing(wav_input: np.ndarray, flx_input: np.ndarray, res_input: np.ndarray, wav_output: np.ndarray, res_output: np.ndarray) -> np.ndarray:
    """
    Decrease the resolution of a spectrum by convolving with a Gaussian kernel.

    Parameters
    ----------
        wav_input : array
            Wavelength grid of the input
        flx_input : array
            Flux of the input
        res_input : array
            Spectral resolution at wav_input
        wav_output : array
            Desired output wavelength grid
        res_output : array
            Spectral resolution at wav_output

    Returns
    -------
        array
            Flux at lower resolution, resampled to wav_output

    Authors
    -------
    Simon Petrus
"""

    if len(flx_input) == 0 or len(wav_input) == 0:
        return np.array([])

    # Interpolation to common resolution
    res_in_interp = interp1d(wav_input, res_input, kind='linear', bounds_error=False)
    res_out_interp = interp1d(wav_output, res_output, kind='linear', bounds_error=False)
    res_in = res_in_interp(wav_output)
    res_out = res_out_interp(wav_output)

    # FWHM and convolution width
    fwhm_in = wav_output / res_in
    fwhm_out = wav_output / res_out

    fwhm_conv = np.sqrt(np.maximum(fwhm_out**2 - fwhm_in**2, 0.0))  # avoid sqrt of negative
    sigma_conv = fwhm_conv / 2.355  # convert FWHM to sigma

    # Apply convolution
    flx_output = convolve_and_sample(wav_output, sigma_conv, wav_input, flx_input, force_int=True)
    return flx_output


# ----------------------------------------------------------------------------------------------------------------------


def continuum_estimate(wav_input: np.ndarray, flx_input: np.ndarray, res_input: np.ndarray, wav_cont_bounds: str, res_cont: float) -> np.ndarray:
    """
    Decrease the resolution of a spectrum (data or model). The function calculates the FWHM as a function of the
    wavelengths of the custom spectral resolution (estimated for the continuum). It then calculates a sigma to decrease
    the resolution of the spectrum to this custom FWHM for each wavelength using a gaussian filter and resample it on
    the wavelength grid of the data.

    Parameters
    ----------
        wav_input : np.ndarray
            Wavelength grid of the spectrum for which you want to estimate the continuum
        flx_input : np.ndarray
            Flux of the spectrum for which you want to estimate the continuum
        res_input : np.ndarray
            Spectral resolution of the spectrum for which you want to estimate the continuum
        wav_cont_bounds : str
            Wavelength bounds where you want to estimate the continuum
        res_cont : float
            Approximate resolution of the continuum
    Returns
    -------
        np.ndarray
            Estimated continuum of the spectrum re-sampled on the data wavelength grid

    Authors
    -------
    Simon Petrus, Matthieu Ravet

"""

    # Initialize
    flx_cont = np.asarray([])
    wav_cont = np.asarray([])
    # Redifine a spectrum only composed by the wavelength ranges used to estimate the continuum
    for wav_cont_cut in wav_cont_bounds.split('/'):
        wav_cont_cut = wav_cont_cut.split(',')
        ind_cont_cut = np.where((float(wav_cont_cut[0]) <= wav_input) & (wav_input <= float(wav_cont_cut[1])))

        # To limit the computing time, the convolution is not as a function of the wavelength but calculated
        # from the median wavelength. We just want an estimate of the continuum here.
        wav_median = np.median(wav_input[ind_cont_cut])
        dwav_median = np.median(np.abs(wav_input[ind_cont_cut] - np.roll(wav_input[ind_cont_cut], 1))) # Estimated the median wavelength separation instead of taking wav_median - (wav_median+1) that could be on a border

        fwhm = wav_median / np.median(res_input)
        fwhm_continuum = wav_median / float(res_cont)

        fwhm_conv = np.sqrt(fwhm_continuum**2 - fwhm**2)
        sigma = fwhm_conv / (dwav_median * 2.355)
        cont = gaussian_filter(flx_input[ind_cont_cut], sigma)

        # Concatenate everything
        wav_cont = np.concatenate((wav_cont, wav_input[ind_cont_cut]))
        flx_cont = np.concatenate((flx_cont, cont))

    # Reinterpolate onto the original wavelength grid
    continuum_interp = interp1d(wav_cont, flx_cont, kind='linear', fill_value = 'extrapolate')
    continuum = continuum_interp(wav_input)

    return continuum


# ----------------------------------------------------------------------------------------------------------------------



def doppler_fct(wav_mod_spectro: np.ndarray, flx_mod_spectro: np.ndarray, res_mod_spectro: np.ndarray, rv_picked: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Application of a Doppler shifting to the interpolated synthetic spectrum using the function pyasl.dopplerShift.
    The side effects of the Doppler shifting are taking into account by using a model interpolated on a larger wavelength grid as the wavelength grid of the data.
    After the Doppler shifting, the model is then cut to the wavelength of the data.

    Parameters
    ----------
        wav_mod_spectro : array
            Wavelength grid of the model
        flx_mod_spectro : array
            Flux of the interpolated synthetic spectrum
        res_mod_spectro : array
            Resolution of the odel
        rv_picked : float
            Radial velocity randomly picked by the nested sampling (in km.s-1)
    Returns
    -------
        array
            Wavelength grid after Doppler shifting
        array
            New flux of the interpolated synthetic spectrum
        array
            New resolution of the interpolated synthetic spectrum

    Authors
    -------
    Simon Petrus, Allan Denis and Matthieu Ravet
"""

    if len(flx_mod_spectro) != 0:
        new_wav = wav_mod_spectro * ((rv_picked / const.c.to(u.km/u.s).value) + 1)
        rv_interp = interp1d(new_wav, flx_mod_spectro, bounds_error=False)
        flx_post_doppler = rv_interp(wav_mod_spectro)

        # Remove the nans caused by the RV correction
        # Note: this step is not problematic as the wavelength range of the model is slightly larger than the wavelength range of the data
        # so we do not lose any data in the model within the wavelength range of the data
        nans = np.where(~np.isnan(flx_post_doppler))[0]
        wav_post_doppler, flx_post_doppler = wav_mod_spectro[nans], flx_post_doppler[nans]
        res_post_doppler = res_mod_spectro[nans]
    else:
        wav_post_doppler, flx_post_doppler, res_post_doppler = wav_mod_spectro, flx_mod_spectro, res_mod_spectro

    return wav_post_doppler, flx_post_doppler, res_post_doppler



# ----------------------------------------------------------------------------------------------------------------------



def reddening_fct(wav: np.ndarray, flx: np.ndarray, av_picked: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Application of a sythetic interstellar extinction to the interpolated synthetic spectrum using the function
    extinction.fm07.

    Parameters
    ----------
        wav : array
            Wavelength grid of the model
        flx : array
            Flux of the interpolated synthetic spectrum
        av_picked : float
            Extinction randomly picked by the nested sampling (in mag)
    Returns
    -------
        array
            New flux of the interpolated synthetic spectrum

    Authors
    -------
    Simon Petrus
"""

    if len(flx) != 0:
        dered_merge = extinction.fm07(wav * 10000, av_picked, unit='aa')
        flx_rd = flx * 10**(-0.4*dered_merge)
    else:
        flx_rd = flx

    return flx_rd



# ----------------------------------------------------------------------------------------------------------------------



def vsini_fct(wav_mod_spectro: np.ndarray, flx_mod_spectro: np.ndarray, res_mod_obs_spectro: np.ndarray, ld_picked: float, vsini_picked: float, vsini_type: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Application of a rotational velocity (line broadening) to the interpolated synthetic spectrum

    Parameters
    ----------
        wav_mod_spectro : array
            Wavelength grid of the model
        flx_mod_spectro : array
            Flux of tge interpolated synthetic spectrum (spectroscopy)
        res_mod_obs_spectro : array
            Resolution of the model as a function of the wavelength grid of the data
        ld_picked : float
            Limb darkening randomly picked by the nested sampling
        vsini_picked : float
            v.sin(i) randomly picked by the nested samplin (in km.s-1)
        vsini_type : str
            Vsin(i) function to use
    Returns
    -------
        array
            New flux of the broadened synthetic spectrum (spectroscopy)
        array
            New resolution of the broadened synthetic spectrum (photometry)

    Authors
    -------
    Allan Denis
"""
    if len(flx_mod_spectro) != 0:
        if vsini_picked != 0:
            if vsini_type == VsiniFunction.RotBroad.value:
                flx_mod_spectro_broad = vsini_fct_rot_broad(wav_mod_spectro, flx_mod_spectro, ld_picked, vsini_picked)
            elif vsini_type == VsiniFunction.FastRotBroad.value:
                flx_mod_spectro_broad = vsini_fct_fast_rot_broad(wav_mod_spectro, flx_mod_spectro, ld_picked, vsini_picked)
            elif vsini_type == VsiniFunction.Accurate.value:
                flx_mod_spectro_broad = vsini_fct_accurate(wav_mod_spectro, flx_mod_spectro, ld_picked, vsini_picked)
            elif vsini_type == VsiniFunction.AccurateFast.value:
                flx_mod_spectro_broad = vsini_fct_accurate_fast_rot_broad(wav_mod_spectro, flx_mod_spectro, ld_picked, vsini_picked)
            else:
                raise ValueError(f'Unknow rotational broadening method {vsini_type}')

            # Because of the v.sini correction, the resolution of the model has been downgraded, so we update it
            res_mod_obs_spectro_broad = const.c.to('km/s').value / vsini_picked * np.ones(len(res_mod_obs_spectro))

        else: # vsini_picked is 0
            res_mod_obs_spectro_broad, flx_mod_spectro_broad = res_mod_obs_spectro, flx_mod_spectro
    else:
        flx_mod_spectro_broad, res_mod_obs_spectro_broad = flx_mod_spectro, res_mod_obs_spectro


    return flx_mod_spectro_broad, res_mod_obs_spectro_broad



# ----------------------------------------------------------------------------------------------------------------------



def vsini_fct_rot_broad(wav_mod_spectro: np.ndarray, flx_mod_spectro: np.ndarray, ld_picked: np.ndarray, vsini_picked: float) -> np.ndarray:
    """
    Application of a rotation velocity (line broadening) to the interpolated synthetic spectrum using the function
    extinction.fm07.

    Parameters
    ----------
        wav_mod_spectro : array
            Wavelength grid of the model
        flx_mod_spectro : array
            Flux of the interpolated synthetic spectrum
        ld_picked : float
            Limd darkening randomly picked by the nested sampling
        vsini_picked : float
            v.sin(i) randomly picked by the nested sampling (in km.s-1)
    Returns
    -------
        array
            New flux of the interpolated synthetic spectrum

    Authors
    -------
    Simon Petrus
"""
    # Correct irregulatities in the wavelength grid
    wav_interval = wav_mod_spectro[1:] - wav_mod_spectro[:-1]
    wav_to_vsini = np.arange(min(wav_mod_spectro), max(wav_mod_spectro), min(wav_interval) * 2/3)
    vsini_interp = interp1d(wav_mod_spectro, flx_mod_spectro, fill_value="extrapolate")
    flx_to_vsini = vsini_interp(wav_to_vsini)
    # Apply the v.sin(i)
    new_flx = rotBroad(wav_to_vsini, flx_to_vsini, ld_picked, vsini_picked)
    vsini_interp = interp1d(wav_to_vsini, new_flx, fill_value="extrapolate")
    flx_mod_spectro_broad = vsini_interp(wav_mod_spectro)

    return flx_mod_spectro_broad



# ----------------------------------------------------------------------------------------------------------------------



def vsini_fct_fast_rot_broad(wav_mod_spectro: np.ndarray, flx_mod_spectro: np.ndarray, ld_picked: np.ndarray, vsini_picked: float) -> np.ndarray:
    """
    Application of a rotation velocity (line broadening) to the interpolated synthetic spectrum using the function
    extinction.fm07.

    Parameters
    ----------
        wav_mod_spectro : array
            Wavelength grid of the model
        flx_mod_spectro : array
            Flux of the interpolated synthetic spectrum
        ld_picked : float
            Limd darkening randomly picked by the nested sampling
        vsini_picked : float
            v.sin(i) randomly picked by the nested sampling (in km.s-1)
    Returns
    -------
        array
            New flux of the interpolated synthetic spectrum

    Authors
    -------
    Simon Petrus
"""
    # Correct irregulatities in the wavelength grid
    wav_interval = wav_mod_spectro[1:] - wav_mod_spectro[:-1]
    wav_to_vsini = np.arange(min(wav_mod_spectro), max(wav_mod_spectro), min(wav_interval) * 2/3)
    vsini_interp = interp1d(wav_mod_spectro, flx_mod_spectro, fill_value="extrapolate")
    flx_to_vsini = vsini_interp(wav_to_vsini)
    # Apply the v.sin(i)
    new_flx = fastRotBroad(wav_to_vsini, flx_to_vsini, ld_picked, vsini_picked)
    vsini_interp = interp1d(wav_to_vsini, new_flx, fill_value="extrapolate")
    flx_mod_spectro_broad = vsini_interp(wav_mod_spectro)

    return flx_mod_spectro_broad



# ----------------------------------------------------------------------------------------------------------------------



def vsini_fct_accurate(wav_mod_spectro: np.ndarray, flx_mod_spectro: np.ndarray, ld_picked: np.ndarray, vsini_picked: np.ndarray, nr: int=50, ntheta: int=100, dif: float=0.0) -> np.ndarray:
    '''
    A routine to quickly rotationally broaden a spectrum in linear time.
    Adapted from Carvalho & Johns-Krull 2023 https://ui.adsabs.harvard.edu/abs/2023RNAAS...7...91C/abstract

    Parameters
    ----------
        wav_mod_spectro : array
            Wavelength grid of the model
        flx_mod_spectro : array
            Flux of the interpolated synthetic spectrum
        ld_picked : float
            Limd darkening randomly picked by the nested sampling
        vsini_picked : float
            v.sin(i) randomly picked by the nested sampling (in km.s-1)
        nr : int
            (default = 10) The number of radial bins on the projected disk
        ntheta : int
            (default = 100) The number of azimuthal bins in the largest radial annulus
                                            note: the number of bins at each r is int(r*ntheta) where r < 1
        dif : float
            (default = 0) The differential rotation coefficient, applied according to the law Omeg(th)/Omeg(eq) = (1 - dif/2 - (dif/2) cos(2 th)).
                                            Dif = .675 nicely reproduces the law proposed by Smith, 1994, A&A, Vol. 287, p. 523-534, to unify WTTS and CTTS.
                                            Dif = .23 is similar to observed solar differential rotation. Note: the th in the above expression is the stellar co-latitude, not the same as the integration variable used below.
                                            This is a disk integration routine.
    Returns
    -------
        array
            New flux of the interpolated synthetic spectrum

    Authors
    -------
    Allan Denis
'''

    ns = np.copy(flx_mod_spectro)*0.0
    tarea = 0.0
    dr = 1./nr
    for j in range(0, nr):
        r = dr/2.0 + j*dr
        area = ((r + dr/2.0)**2 - (r - dr/2.0)**2)/int(ntheta*r) * (1.0 - ld_picked + ld_picked * np.cos(np.arcsin(r)))
        for k in range(0,int(ntheta*r)):
            th = np.pi/int(ntheta*r) + k * 2.0*np.pi/int(ntheta*r)
            if dif != 0:
                vl = vsini_picked * r * np.sin(th) * (1.0 - dif/2.0 - dif/2.0*np.cos(2.0*np.arccos(r*np.cos(th))))
                ns += area * np.interp(wav_mod_spectro + wav_mod_spectro*vl/const.c.to(u.km/u.s).value, wav_mod_spectro, flx_mod_spectro)
                tarea += area
            else:
                vl = r * vsini_picked * np.sin(th)
                ns += area * np.interp(wav_mod_spectro + wav_mod_spectro*vl/const.c.to(u.km/u.s).value, wav_mod_spectro, flx_mod_spectro)
                tarea += area

    flx_mod_spectro_broad = ns / tarea
    return flx_mod_spectro_broad



# ----------------------------------------------------------------------------------------------------------------------



def vsini_fct_accurate_fast_rot_broad(wav_mod_spectro: np.ndarray, flx_mod_spectro: np.ndarray, ld_picked: float, vsini_picked: float) -> np.ndarray:
    """
    Application of a rotation velocity (line broadening) to the interpolated synthetic spectrum using the Carvalho & Johns-Krull (2023) approach

    Parameters
    ----------
        wav_mod_spectro : array
            Wavelength grid of the model
        flx_mod_spectro : array
            Flux of the interpolated synthetic spectrum
        ld_picked : float
            Limd darkening randomly picked by the nested sampling
        vsini_picked : float
            v.sin(i) randomly picked by the nested sampling (in km.s-1)
    Returns
    -------
        array
            New flux of the interpolated synthetic spectrum

    Authors
    -------
    Simon Petrus, Arthur Vigan and Allan Denis
"""
    # Correct irregulatities in the wavelength grid
    wav_interval = wav_mod_spectro[1:] - wav_mod_spectro[:-1]
    wav_to_vsini = np.arange(min(wav_mod_spectro), max(wav_mod_spectro), min(wav_interval) * 2/3)
    vsini_interp = interp1d(wav_mod_spectro, flx_mod_spectro, fill_value="extrapolate")
    flx_to_vsini = vsini_interp(wav_to_vsini)
    # Apply the v.sin(i)
    new_flx = vsini_fct_accurate(wav_to_vsini, flx_to_vsini, ld_picked, vsini_picked, nr=10, ntheta=100, dif=0.0)
    vsini_interp = interp1d(wav_to_vsini, new_flx, fill_value="extrapolate")
    flx_mod_spectro_broad = vsini_interp(wav_mod_spectro)

    return flx_mod_spectro_broad



# ----------------------------------------------------------------------------------------------------------------------



def bb_cpd_fct(wav: np.ndarray, flx: np.ndarray, distance: np.ndarray, bb_t_picked: np.ndarray, bb_r_picked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Function to add the effect of a cpd (circum planetary disc) to the models.

    Parameters
    ----------
        wav : array
            Wavelength grid of the model
        flx : array
            Flux of the interpolated synthetic spectrum
        distance : array
            Distance from the observation in pc units
        bb_t_picked : float
            Temperature value randomly picked by the nested sampling in K units
        bb_r_picked   float): Radius randomly picked by the nested sampling in units of planetary radius
    Returns
    -------
        array
            New flux of the interpolated synthetic spectrum

    Authors
    -------
    Paulina Palma-Bifani
'''

    if len(flx) > 0:
        bb_t_picked *= u.K
        bb_r_picked *= u.Rjup
        distance *= u.pc

        a = 2.0 * const.h * const.c**2
        b = const.h * const.c / (wav * u.um * const.k_B * bb_t_picked)
        intensity = a / ((wav * u.um)**5 * (np.exp(b) - 1.0))
        flx_bb = (np.pi * bb_r_picked**2 / distance**2 * intensity).to(u.W / u.m**2 / u.micron)

        # add to model flux of the atmosphere
        flx_bb = flx + flx_bb.value
    else:
        flx_bb = flx

    return flx_bb


# ----------------------------------------------------------------------------------------------------------------------


def fit_linear_model(components: list[np.ndarray], flx_obs: np.ndarray, err_obs: np.ndarray | None = None, bounds: tuple | None = None, fixed_coeffs: dict[int, float] | None = None) -> dict:
    """
    Generic weighted linear model fitter.

    Solve:
        flx_obs = sum_i c_i * components

    Parameters
    ----------
    components : list[np.ndarray]
        Model components M_i
    flx_obs : np.ndarray
        Observation
    err_obs : np.ndarray | None
        Error
    bounds : tuple(float, float) | None
        Bounds for the least sqaures
    fixed_coeffs : dict(int: float)
        Dictionary of fixed coeff {index: value}

    Returns
    -------
    dict with keys:
        coeffs
        reconstructed
        model

    Authors
    -------
    Allan Denis
"""

    n = len(components)
    flx_obs = np.asarray(flx_obs)

    if err_obs is None:
        weights = np.ones_like(flx_obs)
    else:
        weights = 1.0 / err_obs**2

    fixed_coeffs = fixed_coeffs or {}
    free_idx = [i for i in range(n) if i not in fixed_coeffs]

    # ======================
    # Build weighted matrix
    # ======================
    A = np.column_stack(components)
    A_w = A * weights[:, None]
    b_w = flx_obs * weights

    # ======================
    # Remove fixed components
    # ======================
    if fixed_coeffs:
        for i, val in fixed_coeffs.items():
            b_w -= A_w[:, i] * val

    # ======================
    # Solve for free coeffs
    # ======================
    coeffs = np.zeros(n)

    if free_idx:
        A_free = A_w[:, free_idx]

        if bounds is not None:
            low, high = bounds
            if np.ndim(low):
                low = np.asarray(low)[free_idx]
                high = np.asarray(high)[free_idx]
            res = optimize.lsq_linear(A_free, b_w, bounds=(low, high))
            coeffs_free = res.x
        else:
            coeffs_free, *_ = np.linalg.lstsq(A_free, b_w, rcond=None)

        for i, idx in enumerate(free_idx):
            coeffs[idx] = coeffs_free[i]

    # ======================
    # Insert fixed coeffs
    # ======================
    for i, val in fixed_coeffs.items():
        coeffs[i] = val

    # ======================
    # Reconstruction
    # ======================
    reconstructed = np.array([coeffs[i] * components[i] for i in range(n)])

    model = np.sum(reconstructed, axis=0)

    return {"coeffs": coeffs, "reconstructed": reconstructed, "model": model}

# ----------------------------------------------------------------------------------------------------------------------


def build_linear_components(flx_mod: np.ndarray | None = None, transm: np.ndarray | None = None, flx_cont_mod: np.ndarray | None = None, star_flx_obs: np.ndarray | None = None, flx_cont_obs: np.ndarray | None = None, star_flx_cont_obs: np.ndarray | None = None, system_obs: np.ndarray | None = None, analytic: str = "yes",  ck_value: float | None = None, bounds: tuple | None = None):
    '''
    Build linear components, fixed coefficients and bounds
    for fit_linear_model().

    Parameters
    ----------
    flx_mod : np.ndarray | None
        Model flux
    transm : np.ndarray | None
        Transmission function
    flx_cont_mod : np.ndarray | None
        Model continuum flux
    star_flx_obs : np.ndarray | None
        Stellar flux
    flx_cont_obs : np.ndarray | None
        Observed continuum flux
    star_flx_cont_obs : np.ndarray | None
        Observation star flux continuum
    system_obs : np.ndarray | None
        Systematics model
    analytic : str
        'yes' to fit for the model, 'no' to apply a constant scaling law
    ck_value : float | None
        Fixed scaling coefficient if scaling_mode is "fixed"
    bounds : tuple | None
        Bounds for the optimization

    Returns
    -------
    components : list[np.ndarray]
        Model components
    fixed_coeffs : dict[int, float]
        Fixed coefficients
    labels : list[str]
        Labels for each component

    Authors
    -------
    Allan Denis
'''

    components = []
    fixed_coeffs = {}
    labels = []

    # ======================
    # Astrophysical model
    # ======================
    if flx_mod is not None:
        comp = flx_mod
        if transm is not None:
            comp = transm * comp

        if flx_cont_mod is not None and star_flx_obs is not None:
            speckles = star_flx_obs / star_flx_cont_obs[:, None]
            mid = speckles.shape[1] // 2
            comp = comp - flx_cont_mod * speckles[:, mid]

        components.append(comp)
        labels.append("model")

        if analytic == "no":
            if ck_value is None:
                raise ValueError("ck_value must be provided when analytic='no'")
            fixed_coeffs[0] = ck_value

    # ======================
    # Stellar speckles
    # ======================
    if star_flx_obs is not None:
        speckles = star_flx_obs / star_flx_cont_obs[:, None]
        for i in range(speckles.shape[1]):
            components.append(speckles[:, i] * flx_cont_obs)
            labels.append(f"speckle_{i}")

    # ======================
    # Systematics
    # ======================
    if system_obs is not None and system_obs.size > 0:
        for i in range(system_obs.shape[1]):
            components.append(system_obs[:, i])
            labels.append(f"systematic_{i}")

    return components, fixed_coeffs, bounds, labels


# ----------------------------------------------------------------------------------------------------------------------


def compute_ccf(wav_mod_spectro: np.ndarray, flx_mod_spectro: np.ndarray, wav_obs_spectro: np.ndarray, flx_obs_spectro: np.ndarray, err_obs_spectro: np.ndarray, res_mod_spectro: np.ndarray, res_obs_spectro: np.ndarray, res_cont: float, wav_fit: str | np.ndarray,  star_flx_obs_spectro: np.ndarray = np.array([]), transm_obs_spectro: np.ndarray = np.array([]), system_obs_spectro: np.ndarray = np.array([]), rv_grid: np.ndarray = np.linspace(-300, 300, 600), rv_sini_map: bool = False, normalize: bool = True):
    '''
    Function to compute the ccf between a template and data

    Parameters
    ----------
        wav_mod_spectro : np.ndarray
            Wavelength grid of the template
        flx_mod_spectro : np.ndarray
            Flux of the template
        wav_obs_spectro : np.ndarray
            Wavelength grid of the data
        flx_obs_spectro : np.ndarray
            Flux of the data
        err_obs_spectro : np.ndarray
            Error of the data
        res_mod_spectro : np.ndarray
            Resolution of the template
        res_obs_spectro : np.ndarray
            Resolution of the data
        res_cont : float
            Resolution of the continuum
        wav_fit : str | np.ndarray
            Wavelengths used for fitting
        star_flx_obs_spectro : np.ndarray
            Star flux of the data
        transm_obs_spectro : float | np.ndarray
            Transmission
        system_obs_spectro : np.ndarray
            Systematics
        rv_grid : np.ndarray
            Grid of RV for the CCF function
        rv_vsini_map : bool
            Whether to use this function to compute a rv / vsini map
        normalize : bool
            Whether to normalize ccf

    Returns
    -------
        ccf_norm : np.ndarray
            CCF function normalized by the SNR
        acf_norm : np.ndarray
            ACF function rescaled and shifted at the peak of the RV
        star_ccf_norm : np.ndarray
            CCF function with star data
        rv_peak : float
            Peak of the RV

    Authors
    -------
    Arthur Vigan and Allan Denis
'''

    if not rv_sini_map and np.max(np.abs(rv_grid)) < 100:
        print(f'Your grid is {rv_grid}. \nPlease, choose an RV grid going beyond [-100, 100]')
        return None, None, None, None, None

    ind_for_fitting = np.array([], dtype=int)
    for wav_fit_cut in wav_fit.split('/'):
        wmin, wmax = map(float, wav_fit_cut.split(','))
        indices = np.where((wav_obs_spectro >= wmin) & (wav_obs_spectro <= wmax))[0]
        ind_for_fitting = np.concatenate((ind_for_fitting, indices))
    ind_for_fitting = np.sort(ind_for_fitting)

    wav_obs_spectro, flx_obs_spectro, err_obs_spectro, res_obs_spectro = wav_obs_spectro[ind_for_fitting], flx_obs_spectro[ind_for_fitting], err_obs_spectro[ind_for_fitting], res_obs_spectro[ind_for_fitting]

    # Estimate continuum of observed flux
    flx_cont_obs_spectro = continuum_estimate(wav_obs_spectro, flx_obs_spectro, res_obs_spectro, wav_fit_cut, res_cont)

    speckles = 1
    star_flx_cont_obs_spectro = np.array([])
    if star_flx_obs_spectro.size > 0:
        star_flx_obs_spectro = star_flx_obs_spectro[ind_for_fitting]
        mid_idx = star_flx_obs_spectro.shape[1] // 2
        star_flx_mid = star_flx_obs_spectro[:, mid_idx]
        star_flx_cont_obs_spectro = continuum_estimate(wav_obs_spectro, star_flx_mid, res_obs_spectro, wav_fit, res_cont)
        speckles = star_flx_obs_spectro / star_flx_cont_obs_spectro[:, None]
    if len(transm_obs_spectro) > 0:
        transm_obs_spectro = transm_obs_spectro[ind_for_fitting]
    else:
        transm_obs_spectro = 1
    if len(system_obs_spectro) > 0:
        system_obs_spectro = system_obs_spectro[ind_for_fitting]

    # Preprocess model template at RV = 0
    flx_mod_spectro_no_rv = resolution_decreasing(wav_mod_spectro, flx_mod_spectro, res_mod_spectro, wav_obs_spectro, res_obs_spectro)
    flx_cont_mod_spectro_no_rv = continuum_estimate(wav_obs_spectro, flx_mod_spectro_no_rv * transm_obs_spectro, res_obs_spectro, wav_fit, res_cont)

    mid_speckles = speckles[:, speckles.shape[1] // 2] if isinstance(speckles, np.ndarray) else speckles
    flx_mod_spectro_no_rv_hf = transm_obs_spectro * flx_mod_spectro_no_rv - flx_cont_mod_spectro_no_rv * mid_speckles

    if star_flx_obs_spectro.size > 0:
        flx_obs_spectro_hf = flx_obs_spectro - flx_cont_obs_spectro * mid_speckles
        star_flx_obs_spectro_hf = star_flx_obs_spectro[:, mid_idx] - star_flx_cont_obs_spectro
    else:
        flx_obs_spectro_hf = flx_obs_spectro - flx_cont_obs_spectro
        star_flx_obs_spectro_hf = np.array([])

    # Compute CCF (parallel)
    try:
        with ThreadPool(processes=mp.cpu_count()) as pool:
            pbar = tqdm(total=len(rv_grid), leave=False)
            results = []

            def update(*_):
                pbar.update()

            for irv in rv_grid:
                task = pool.apply_async(compute_ccf_single_rv, args=(irv, wav_mod_spectro, flx_mod_spectro, flx_mod_spectro_no_rv_hf, res_mod_spectro, wav_obs_spectro, flx_obs_spectro, flx_cont_obs_spectro, flx_obs_spectro_hf, res_obs_spectro, wav_fit, res_cont, err_obs_spectro, transm_obs_spectro, star_flx_obs_spectro, star_flx_cont_obs_spectro, star_flx_obs_spectro_hf, system_obs_spectro, speckles, 0.6), callback=update)
                results.append(task)

            pool.close()
            pool.join()

            ccf, acf, ccf_star, logL = map(np.zeros, [len(rv_grid)] * 4)
            for i, task in enumerate(results):
                ccf[i], acf[i], ccf_star[i], logL[i] = task.get()
    except Exception as e:
        print(f'Parallel computation of CCF produced the following error: {e}. Trying serial computation.')
        ccf, acf, ccf_star, logL = [], [], [], []
        for irv in tqdm(rv_grid):
            res = compute_ccf_single_rv(irv, wav_mod_spectro, flx_mod_spectro, flx_mod_spectro_no_rv_hf, res_mod_spectro, wav_obs_spectro, flx_obs_spectro, flx_cont_obs_spectro, flx_obs_spectro_hf, res_obs_spectro, wav_fit, res_cont, err_obs_spectro, transm_obs_spectro, star_flx_obs_spectro, star_flx_cont_obs_spectro, star_flx_obs_spectro_hf, system_obs_spectro, speckles, 0.6)
            ccf.append(res[0])
            acf.append(res[1])
            ccf_star.append(res[2])
            logL.append(res[3])

        ccf, acf, ccf_star, logL = map(np.array, [ccf, acf, ccf_star, logL])

    if not(rv_sini_map):
        # Normalize CCF
        mask_far = np.abs(rv_grid) > 100
        mask_valid = np.abs(rv_grid) <= 150
        imax = np.argmax(ccf)
        rv_peak = rv_grid[imax]

        if normalize:

            for arr in [ccf, acf, ccf_star]:
                arr -= np.mean(arr[mask_far])

            # SNR normalization
            acf_scale = np.max(ccf[mask_valid]) / np.max(acf)
            sigma = np.sqrt(np.abs(np.nanvar(ccf[np.abs(rv_grid - rv_peak) > 100]) - np.nanvar(acf[mask_far] * acf_scale)))
            max_peak = ccf[imax] / sigma

            ccf = ccf / sigma
            ccf_star = ccf_star / np.std(ccf_star)
            acf = acf * (max_peak / np.max(acf))

            print(f'Maximum peak for rv = {rv_peak:.1f} km/s (SNR = {max_peak:.1f})')

        return ccf, acf, ccf_star, rv_peak, logL, ccf

    return logL



# ----------------------------------------------------------------------------------------------------------------------


def compute_ccf_single_rv(rv: float, wav_mod_spectro: np.ndarray, flx_mod_spectro: np.ndarray, flx_mod_spectro_no_rv_hf: np.ndarray, res_mod_spectro: np.ndarray, wav_obs_spectro: np.ndarray, flx_obs_spectro: np.ndarray, flx_cont_obs_spectro: np.ndarray, flx_obs_spectro_hf: np.ndarray, res_obs_spectro: np.ndarray, wav_cont: str, res_cont: np.ndarray, err_obs_spectro: np.ndarray, transm_obs_spectro: int | np.ndarray = 1, star_flx_obs_spectro: np.ndarray = np.array([]), star_flx_cont_obs_spectro: (int | np.ndarray) = 1, star_flx_obs_spectro_hf: np.ndarray = np.array([]), system_obs_spectro: np.ndarray = np.array([]), speckles: (int | np.ndarray) = 1, ld: float = 0.6) -> tuple[float, float, float]:
    '''
    Function to compute the correlation between template and data for a specific rv value

    Parameters
    ----------
        rv : float
            rv value
        vsini : float
            vsini value
        wav_mod_spectro : np.ndarray
            Wavelength grid of the template
        flx_mod_spectro : np.ndarray
            Flux of the template
        flx_mod_spectro_no_rv_hf : np.ndarray
            High frequency content of the flux of the model at 0 rv
        res_mod_spectro : np.ndarray
            Resolution of the template interpolated onto the wavelength grid of the data
        wav_obs_spectro : np.ndarray
            Wavelength grid of the data
        flx_obs_spectro : np.ndarray
            Fhe flux of the data
        flx_cont_obs_spectro : np.ndarray
            Continuum of the flux of the data
        res_obs_spectro : np.ndarray
            Resolution of the data
        wav_cont : str
            Wavelength used for the continuum
        res_cont : float
            Resolution of the continuum
        err_obs_spectro : np.ndarray
            Error of the flux
        transm_obs_spectro : int | np.ndarray
            Transmission
        star_flx_obs_spectro : np.ndarray
            Flux of the star
        star_flx_cont_obs_spectro : int | np.ndarray
            Continuum of the flux of the star
        system_obs_spectro : np.ndarray
            Systematics
        speckles : int | np.ndarray
            Speckles

    Returns
    -------
        ccf : float
            Correlation between the template and the data
        acf : float
            Autocorrelation between the template and iself
        ccf_star : float
            Correlation between the template and the star data

    Authors
    -------
    Arthur Vigan and Allan Denis
'''

    # Doppler shift + resolution match
    wav_shifted, flx_shifted, res_shifted = doppler_fct(wav_mod_spectro, flx_mod_spectro, res_mod_spectro, rv)
    flx_shifted = resolution_decreasing(wav_shifted, flx_shifted, res_shifted, wav_obs_spectro, res_obs_spectro)

    # Continuum estimation
    flx_cont = continuum_estimate(wav_obs_spectro, flx_shifted * transm_obs_spectro, res_obs_spectro, wav_cont, res_cont)

    mid_idx = speckles.shape[1] // 2 if isinstance(speckles, np.ndarray) else 0
    speckles_mid = speckles[:, mid_idx] if isinstance(speckles, np.ndarray) else speckles
    flx_rv_hf = transm_obs_spectro * flx_shifted - flx_cont * speckles_mid

    # Estimate model signal
    components, fixed_coeffs, bounds, labels = build_linear_components(
                                            flx_shifted,
                                            transm_obs_spectro,
                                            flx_cont,
                                            star_flx_obs_spectro,
                                            flx_cont_obs_spectro,
                                            star_flx_cont_obs_spectro,
                                            system_obs_spectro,
                                            analytic='yes'
                                            )

    result = fit_linear_model(
        components=components,
        flx_obs=flx_obs_spectro,
        err_obs=err_obs_spectro,
        bounds=bounds,
        fixed_coeffs=fixed_coeffs
        )

    ccf, best_model_hf = result['coeffs'][0], result['reconstructed'][0]

    # ACF
    acf = np.sum(flx_rv_hf * flx_mod_spectro_no_rv_hf) / (np.sqrt(np.sum(flx_rv_hf ** 2)) * np.sqrt(np.sum(flx_mod_spectro_no_rv_hf ** 2)))

    # CCF with star if available
    if star_flx_obs_spectro_hf.size > 0:
        ccf_star = np.sum(flx_rv_hf * star_flx_obs_spectro_hf) / (np.sqrt(np.sum(flx_rv_hf ** 2)) * np.sqrt(np.sum(star_flx_obs_spectro_hf ** 2)))
    else:
        ccf_star = 1

    # log-likelihood estimation
    residuals = flx_obs_spectro_hf - best_model_hf
    ki2 = np.sum((residuals / err_obs_spectro) ** 2)
    logL = -0.5 * ki2 - 0.5 * np.nansum(np.log(2 * np.pi * err_obs_spectro ** 2))

    return ccf, acf, ccf_star, logL
