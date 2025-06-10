import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import extinction
import astropy.units as u
import astropy.constants as const
from PyAstronomy.pyasl import rotBroad, fastRotBroad
import ForMoSA.utils as utils
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------

def convolve_and_sample(wv_channels: list, sigmas_wvs: list, model_wvs: np.ndarray, model_fluxes: np.ndarray, num_sigma: int=3, force_int: bool=True) -> np.ndarray: # num_sigma = 3 is a good compromise between sampling enough the gaussian and fast interpolation
    """
    Simulate the observations of a model. Convolves the model with a variable Gaussian LSF, sampled at each desired
    spectral channel.

    Args:
        wv_channels (list(floats)): the wavelengths values desired
        sigmas_wvs  (list(floats)): the LSF gaussian standard deviation of each wv_channels [IN UNITS OF model_wvs]
        model_wvs          (array): the wavelengths of the model
        model_fluxes       (array): the fluxes of the model
        num_sigma            (int): number of +/- sigmas to evaluate the LSF to.
        force_int         (bolean): False by default. If True, will force interpolation onto wv_channels when the kernel is singular
    Returns:
        - output_model     (array): the fluxes in each of the wavelength channels

    Author: Jason Wang
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
    Decrease the resolution of a spectrum. The function calculates the FWHM as a function of the
    wavelengths for the input and output fluxes and estimates the highest one
    for each wavelength (the lowest spectral resolution). It then calculates a sigma to decrease the resolution of the
    spectrum to this lowest FWHM for each wavelength and resample it on the wavelength grid of the data using the
    function 'convolve_and_sample'.

    Args:
        wav_input        (array): Wavelength grid of the input
        flx_input        (array): Flux of the input
        res_input        (array): Spectral resolution of the input as a function of wav_output
        wav_output       (array): Wavelength grid of the output
        res_output       (array): Spectral resolution of the output as a function of the wavelength grid of the input
    Returns:
        - flx_output     (array): Flux of the spectrum with a decreased spectral resolution, re-sampled on the data wavelength grid

    Author: Simon Petrus
    """
    # Little nuggets to speed up in case of missing input
    if len(flx_input) == 0:
        flx_output = flx_input
    else:

        # Estimate of the FWHM of the input as a function of the wavelength
        fwhm = wav_output / res_input
        sigma_conv = fwhm / 2.355
        flx_output = convolve_and_sample(wav_output, sigma_conv, wav_input, flx_input, force_int=True)

    return flx_output

# ----------------------------------------------------------------------------------------------------------------------


def continuum_estimate(wav_input: np.ndarray, flx_input: np.ndarray, res_input: np.ndarray, wav_cont_bounds: str | np.ndarray, res_cont: float) -> np.ndarray:
    """
    Decrease the resolution of a spectrum (data or model). The function calculates the FWHM as a function of the
    wavelengths of the custom spectral resolution (estimated for the continuum). It then calculates a sigma to decrease
    the resolution of the spectrum to this custom FWHM for each wavelength using a gaussian filter and resample it on
    the wavelength grid of the data.

    Args:
        wav_input              (np.ndarray): Wavelength grid of the spectrum for which you want to estimate the continuum
        flx_input              (np.ndarray): Flux of the spectrum for which you want to estimate the continuum
        res_input              (np.ndarray): Spectral resolution of the spectrum for which you want to estimate the continuum
        wav_cont_bounds  (str | np.ndarray): Wavelength bounds where you want to estimate the continuum
        res_cont                    (float): Approximate resolution of the continuum
    Returns:
        - continuum    (np.ndarray): Estimated continuum of the spectrum re-sampled on the data wavelength grid

    Author: Simon Petrus, Matthieu Ravet

    """

    # Initialize
    flx_cont = np.asarray([])
    wav_cont = np.asarray([])
    # Redifine a spectrum only composed by the wavelength ranges used to estimate the continuum
    for _, wav_cont_cut in enumerate(wav_cont_bounds.split('/')):
        wav_cont_cut = wav_cont_cut.split(',')
        ind_cont_cut = np.where((float(wav_cont_cut[0]) <= wav_input) & (wav_input <= float(wav_cont_cut[1])))

        # To limit the computing time, the convolution is not as a function of the wavelength but calculated
        # from the median wavelength. We just want an estimate of the continuum here.
        wav_median = np.median(wav_input[ind_cont_cut])
        dwav_median = np.median(np.abs(wav_input[ind_cont_cut] - np.roll(wav_input[ind_cont_cut], 1))) # Estimated the median wavelength separation instead of taking wav_median - (wav_median+1) that could be on a border

        fwhm = wav_median / np.median(res_input)
        fwhm_continuum = wav_median / res_cont

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



def calc_ck(obs_dict_spectro: dict, obs_dict_photo: dict, flx_mod_spectro: np.ndarray, flx_mod_photo: np.ndarray, r_picked: float, d_picked: float, alpha: float=1, analytic: str='no') -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Calculation of the dilution factor Ck and re-normalization of the interpolated synthetic spectrum (from the radius
    and distance or analytically).

    Args:
        obs_dict_spectro       (dict): Dictionay containing all the observationnal entries (spectroscopy)
        obs_dict_photo         (dict): Dictionay containing all the observationnal entries (photometry)
        flx_mod_spectro        (array): Flux of the interpolated synthetic spectrum (spectroscopy)
        flx_mod_photo          (array): Flux of the interpolated synthetic spectrum (photometry)
        r_picked               (float): Radius randomly picked by the nested sampling (in RJup)
        d_picked               (float): Distance randomly picked by the nested sampling (in pc)
        alpha                  (float): Manual scaling factor (set to 1 by default) such that ck = alpha * (r/d)²
        analytic                 (str): = 'yes' if Ck needs to be calculated analytically by the formula from Cushing et al. (2008)
    Returns:
        - flx_mod_spectro_ck   (array): Re-normalysed model spectrum
        - flx_mod_photo_ck     (array): Re-normalysed model photometry
        - ck_spectro           (float): Scaling coefficient for spectroscopy
        - ck_photo             (float): Scaling coefficient for photometry

    Author: Simon Petrus
    """
    # Calculation of the dilution factor ck as a function of the radius and distance
    if analytic == 'no':
        r_picked *= u.Rjup
        d_picked *= u.pc
        ck = alpha * (r_picked.to(u.m).value/d_picked.to(u.m).value)**2
        ck_spectro, ck_photo = ck, ck
    # Calculation of the dilution factor ck analytically
    else:
        if len(obs_dict_spectro['wav']) != 0:
            ck_top_merge = np.sum((flx_mod_spectro * obs_dict_spectro['flx']) / (obs_dict_spectro['err'] * obs_dict_spectro['err']))
            ck_bot_merge = np.sum((flx_mod_spectro / obs_dict_spectro['err'])**2)
            ck_spectro = ck_top_merge / ck_bot_merge
        else:
            ck_top_merge = 0
            ck_bot_merge = 0
            ck_spectro = 1
        if len(obs_dict_photo['wav']) != 0:
            ck_top_phot = np.sum((flx_mod_photo * obs_dict_photo['flx']) / (obs_dict_photo['err'] * obs_dict_photo['err']))
            ck_bot_phot = np.sum((flx_mod_photo / obs_dict_photo['err'])**2)
            ck_photo = ck_top_phot / ck_bot_phot
        else:
            ck_top_phot = 0
            ck_bot_phot = 0
            ck_photo = 1

    # Re-normalization of the interpolated synthetic spectra with ck
    if len(obs_dict_spectro['wav']) != 0:
        flx_mod_spectro_ck = flx_mod_spectro * ck_spectro
    else:
        flx_mod_spectro_ck = flx_mod_spectro
    if len(obs_dict_photo['wav']) != 0:
        flx_mod_photo_ck = flx_mod_photo * ck_photo
    else:
        flx_mod_photo_ck = flx_mod_photo

    return flx_mod_spectro_ck, flx_mod_photo_ck, ck_spectro, ck_photo


# ----------------------------------------------------------------------------------------------------------------------



def doppler_fct(wav_mod_spectro: np.ndarray, flx_mod_spectro: np.ndarray, rv_picked: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Application of a Doppler shifting to the interpolated synthetic spectrum using the function pyasl.dopplerShift.
    The side effects of the Doppler shifting are taking into account by using a model interpolated on a larger wavelength grid as the wavelength grid of the data.
    After the Doppler shifting, the model is then cut to the wavelength of the data.

    Args:
        wav_mod_spectro      (array): Wavelength grid of the model
        flx_mod_spectro      (array): Flux of the interpolated synthetic spectrum
        rv_picked            (float): Radial velocity randomly picked by the nested sampling (in km.s-1)
    Returns:
        - wav_post_doppler   (array): Wavelength grid after Doppler shifting
        - flx_post_doppler   (array): New flux of the interpolated synthetic spectrum

    Author: Simon Petrus, Allan Denis and Matthieu Ravet
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
    else:
        wav_post_doppler, flx_post_doppler = wav_mod_spectro, flx_mod_spectro

    return wav_post_doppler, flx_post_doppler



# ----------------------------------------------------------------------------------------------------------------------



def reddening_fct(wav_mod_spectro: np.ndarray, wav_obs_photo: np.ndarray, flx_mod_spectro: np.ndarray, flx_mod_photo: np.ndarray, av_picked: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Application of a sythetic interstellar extinction to the interpolated synthetic spectrum using the function
    extinction.fm07.

    Args:
        wav_mod_spectro         (array): Wavelength grid of the model (spectroscopy)
        wav_obs_photo           (array): Wavelength of the data/model (photometry)
        flx_mod_spectro         (array): Flux of the interpolated synthetic spectrum (spectroscopy)
        flx_mod_photo           (array): Flux of the interpolated synthetic spectrum (photometry)
        av_picked               (float): Extinction randomly picked by the nested sampling (in mag)
    Returns:
        - flx_mod_spectro_rd    (array): New flux of the interpolated synthetic spectrum (spectroscopy)
        - flx_mod_photo_rd      (array): New flux of the interpolated synthetic spectrum (photometry)

    Author: Simon Petrus
    """
    if len(flx_mod_spectro) != 0:
        dered_merge = extinction.fm07(wav_mod_spectro * 10000, av_picked, unit='aa')
        flx_mod_spectro_rd = flx_mod_spectro * 10**(-0.4*dered_merge)
    else:
        flx_mod_spectro_rd = flx_mod_spectro
    if len(flx_mod_photo) != 0:
        dered_phot = extinction.fm07(wav_obs_photo * 10000, av_picked, unit='aa')
        flx_mod_photo_rd = flx_mod_photo * 10**(-0.4*dered_phot)
    else:
        flx_mod_photo_rd = flx_mod_photo

    return flx_mod_spectro_rd, flx_mod_photo_rd



# ----------------------------------------------------------------------------------------------------------------------



def vsini_fct(wav_mod_spectro: np.ndarray, flx_mod_spectro: np.ndarray, res_mod_obs_spectro: np.ndarray, ld_picked: float, vsini_picked: float, vsini_type: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Application of a rotational velocity (line broadening) to the interpolated synthetic spectrum

    Args:
        wav_mod_spectro          (array): Wavelength grid of the model
        flx_mod_spectro          (array): Flux of tge interpolated synthetic spectrum (spectroscopy)
        res_mod_obs_spectro      (array): Resolution of the model as a function of the wavelength grid of the data
        ld_picked                (float): Limb darkening randomly picked by the nested sampling
        vsini_picked             (float): v.sin(i) randomly picked by the nested samplin (in km.s-1)
        vsini_type                 (str): Vsin(i) function to use
    Returns:
        - flx_mod_spectro_broad  (array): New flux of the broadened synthetic spectrum (spectroscopy)
        - res_mod_obs_broad      (array): New resolution of the broadened synthetic spectrum (photometry)

    Author: Allan Denis
    """
    if len(flx_mod_spectro) != 0:
        if vsini_picked != 0:
            if vsini_type == 'RotBroad':
                flx_mod_spectro_broad = vsini_fct_rot_broad(wav_mod_spectro, flx_mod_spectro, ld_picked, vsini_picked)
            elif vsini_type == 'FastRotBroad':
                flx_mod_spectro_broad = vsini_fct_fast_rot_broad(wav_mod_spectro, flx_mod_spectro, ld_picked, vsini_picked)
            elif vsini_type == 'Accurate':
                flx_mod_spectro_broad = vsini_fct_accurate(wav_mod_spectro, flx_mod_spectro, ld_picked, vsini_picked)
            elif vsini_type == 'AccurateFastRotBroad':
                flx_mod_spectro_broad = vsini_fct_accurate_fast_rot_broad(wav_mod_spectro, flx_mod_spectro, ld_picked, vsini_picked)
            else:
                raise ValueError(f'Unknow rotational broadening method {vsini_type}')

        # Because of the v.sini correction, the resolution of the model has been downgraded, so we update it
        if vsini_picked != 0:
            res_mod_obs_spectro_broad = const.c.to('km/s').value / vsini_picked * np.ones(len(res_mod_obs_spectro))
    else:
        flx_mod_spectro_broad, res_mod_obs_spectro_broad = flx_mod_spectro, res_mod_obs_spectro

    return flx_mod_spectro_broad, res_mod_obs_spectro_broad



# ----------------------------------------------------------------------------------------------------------------------



def vsini_fct_rot_broad(wav_mod_spectro: np.ndarray, flx_mod_spectro: np.ndarray, ld_picked: np.ndarray, vsini_picked: float) -> np.ndarray:
    """
    Application of a rotation velocity (line broadening) to the interpolated synthetic spectrum using the function
    extinction.fm07.

    Args:
        wav_mod_spectro            (array): Wavelength grid of the model
        flx_mod_spectro            (array): Flux of the interpolated synthetic spectrum
        ld_picked                  (float): Limd darkening randomly picked by the nested sampling
        vsini_picked               (float): v.sin(i) randomly picked by the nested sampling (in km.s-1)
    Returns:
        - flx_mod_spectro_broad    (array): New flux of the interpolated synthetic spectrum

    Author: Simon Petrus
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

    Args:
        wav_mod_spectro            (array): Wavelength grid of the model
        flx_mod_spectro            (array): Flux of the interpolated synthetic spectrum
        ld_picked                  (float): Limd darkening randomly picked by the nested sampling
        vsini_picked               (float): v.sin(i) randomly picked by the nested sampling (in km.s-1)
    Returns:
        - flx_mod_spectro_broad    (array): New flux of the interpolated synthetic spectrum

    Author: Simon Petrus
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

    Args:
        wav_mod_spectro            (array): Wavelength grid of the model
        flx_mod_spectro            (array): Flux of the interpolated synthetic spectrum
        ld_picked                  (float): Limd darkening randomly picked by the nested sampling
        vsini_picked               (float): v.sin(i) randomly picked by the nested sampling (in km.s-1)
        nr                           (int): (default = 10) The number of radial bins on the projected disk
        ntheta                       (int): (default = 100) The number of azimuthal bins in the largest radial annulus
                                            note: the number of bins at each r is int(r*ntheta) where r < 1
        dif                        (float): (default = 0) The differential rotation coefficient, applied according to the law Omeg(th)/Omeg(eq) = (1 - dif/2 - (dif/2) cos(2 th)).
                                            Dif = .675 nicely reproduces the law proposed by Smith, 1994, A&A, Vol. 287, p. 523-534, to unify WTTS and CTTS.
                                            Dif = .23 is similar to observed solar differential rotation. Note: the th in the above expression is the stellar co-latitude, not the same as the integration variable used below.
                                            This is a disk integration routine.
    Returns:
        - flx_mod_spectro_broad    (array): New flux of the interpolated synthetic spectrum

    Author: Allan Denis
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

    Args:
        wav_mod_spectro           (array): Wavelength grid of the model
        flx_mod_spectro           (array): Flux of the interpolated synthetic spectrum
        ld_picked                 (float): Limd darkening randomly picked by the nested sampling
        vsini_picked              (float): v.sin(i) randomly picked by the nested sampling (in km.s-1)
    Returns:
        - flx_mod_spectro_broad   (array): New flux of the interpolated synthetic spectrum

    Author: Simon Petrus, Arthur Vigan and Allan Denis
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



def bb_cpd_fct(wav_mod_spectro: np.ndarray, wav_obs_photo: np.ndarray, flx_mod_spectro: np.ndarray, flx_mod_photo: np.ndarray, distance: np.ndarray, bb_t_picked: np.ndarray, bb_r_picked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Function to add the effect of a cpd (circum planetary disc) to the models.

    Args:
        wav_mod_spectro        (array): Wavelength grid of the model (spectroscopy)
        wav_obs_photo          (array): Wavelength of the data/model (photometry)
        flx_mod_spectro        (array): Flux of the interpolated synthetic spectrum (spectroscopy)
        flx_mod_photo          (array): Flux of the interpolated synthetic spectrum (photometry)
        distance               (array): Distance from the observation in pc units
        bb_t_picked            (float): Temperature value randomly picked by the nested sampling in K units
        bb_r_picked            (float): Radius randomly picked by the nested sampling in units of planetary radius
    Returns:
        - flx_mod_spectro_bb   (array): New flux of the interpolated synthetic spectrum (spectroscopy)
        - flx_mod_photo_bb     (array): New flux of the interpolated synthetic spectrum (photometry)

    Author: Paulina Palma-Bifani
    '''

    bb_t_picked *= u.K
    bb_r_picked *= u.Rjup
    distance *= u.pc

    def planck(wav, T):
        a = 2.0*const.h*const.c**2
        b = const.h*const.c/(wav*const.k_B*T)
        intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
        return intensity

    bb_intensity    = planck(wav_mod_spectro*u.um, bb_t_picked)
    bb_intensity_f    = planck(wav_obs_photo*u.um, bb_t_picked)

    flux_bb_lambda   = ( np.pi*bb_r_picked**2/(distance**2) * bb_intensity ).to(u.W/u.m**2/u.micron)
    flux_bb_lambda_f = ( np.pi*bb_r_picked**2/(distance**2) * bb_intensity_f ).to(u.W/u.m**2/u.micron)

    # add to model flux of the atmosphere
    flx_mod_spectro_bb = flx_mod_spectro + flux_bb_lambda.value
    flx_mod_photo_bb = flx_mod_photo + flux_bb_lambda_f.value

    return flx_mod_spectro_bb, flx_mod_photo_bb


# ----------------------------------------------------------------------------------------------------------------------


def compute_ccf(rv_grid: np.ndarray, wav_mod_spectro: np.ndarray, flx_mod_spectro: np.ndarray, wav_obs_spectro: np.ndarray, flx_obs_spectro: np.ndarray, err_obs_spectro: np.ndarray, res_mod_obs_spectro: np.ndarray, res_obs_spectro: np.ndarray, res_cont: float, wav_cont_cut: str | np.ndarray,  star_flx_obs_spectro: np.ndarray = np.array([]), transm_obs_spectro: float | np.ndarray = 1, system_obs_spectro: np.ndarray = np.array([])):
    '''
    Function to compute the ccf between a template and data

    Args:
        rv_grid                     (np.ndarray): Grid of RV for the CCF function
        wav_mod_spectro             (np.ndarray): Wavelength grid of the template
        flx_mod_spectro             (np.ndarray): Flux of the template
        wav_obs_spectro             (np.ndarray): Wavelength grid of the data
        flx_obs_spectro             (np.ndarray): Flux of the data
        err_obs_spectro             (np.ndarray): Error of the data
        res_mod_obs_spectro         (np.ndarray): Resolution of the template interpolated onto the wavelength grid of the data
        res_obs_spectro             (np.ndarray): Resolution of the data
        res_cont                         (float): Resolution of the continuum
        wav_cont_cut          (str | np.ndarray): Wavelengths used for the continuum estimation
        star_flx_obs_spectro        (np.ndarray): Star flux of the data
        transm_obs_spectro  (float | np.ndarray): Transmission
        system_obs_spectro          (np.ndarray): Systematics

    Returns:
        ccf (np.ndarray):

    Authors: Allan Denis
    '''

    # Continuums estimation
    flx_cont_obs_spectro = continuum_estimate(wav_obs_spectro, flx_obs_spectro, res_obs_spectro, wav_cont_cut, res_cont)
    star_flx_cont_obs_spectro = continuum_estimate(wav_obs_spectro, star_flx_obs_spectro, res_obs_spectro, wav_cont_cut, res_cont)
    # Initialization of ccf_list and acf_list
    # Todel at rv = 0 for autocorrelation
    flx_mod_spectro_no_rv = resolution_decreasing(wav_mod_spectro, flx_mod_spectro, res_mod_obs_spectro, wav_obs_spectro, res_obs_spectro)
    # Continuum of template at rv = 0 for autocorrelation
    flx_cont_mod_spectro_no_rv = continuum_estimate(wav_obs_spectro, flx_mod_spectro_no_rv, res_obs_spectro, wav_cont_cut, res_cont)
    flx_mod_spectro_no_rv = transm_obs_spectro * (flx_mod_spectro_no_rv - flx_cont_mod_spectro_no_rv)

    # compute CCF with pool of workers
    with ThreadPool(processes=mp.cpu_count()) as pool:
        pbar = tqdm(total=len(rv_grid), leave=False)

        def update(*a):
            pbar.update()

        tasks = []
        # Loop in rv
        for irv in rv_grid:
            tasks.append(pool.apply_async(compute_ccf_single_rv, args=(irv, wav_mod_spectro, flx_mod_spectro, flx_mod_spectro_no_rv, res_mod_obs_spectro, wav_obs_spectro, flx_obs_spectro, err_obs_spectro, flx_cont_obs_spectro, res_obs_spectro, res_cont, wav_cont_cut, star_flx_obs_spectro, star_flx_cont_obs_spectro, transm_obs_spectro, system_obs_spectro)))

        pool.close()
        pool.join()

        # extract results
        ccf = np.zeros(len(rv_grid))
        acf = np.zeros(len(rv_grid))
        for irv, task in enumerate(tasks):
            res = task.get()
            ccf[irv] = res[0]
            acf[irv] = res[1]

    return ccf, acf


# ----------------------------------------------------------------------------------------------------------------------


def compute_ccf_single_rv(rv: float, wav_mod_spectro: np.ndarray, flx_mod_spectro: np.ndarray, flx_mod_spectro_no_rv_hf: np.ndarray, res_mod_obs_spectro: np.ndarray, wav_obs_spectro: np.ndarray, flx_obs_spectro: np.ndarray, err_obs_spectro: np.ndarray, flx_cont_obs_spectro: np.ndarray, res_obs_spectro: np.ndarray, res_cont: np.ndarray,  wav_cont_cut: np.ndarray, star_flx_obs_spectro: np.ndarray, star_flx_cont_obs_spectro: np.ndarray, transm_obs_spectro: np.ndarray, system_obs_spectro: np.ndarray) -> tuple[float, float]:
    '''
    Function to compute the correlation between template and data for a specific rv value


    Args:
        rv                               (float): rv value
        wav_mod_spectro             (np.ndarray): Wavelength grid of the template
        flx_mod_spectro             (np.ndarray): Flux of the template
        flx_mod_spectro_no_rv_hf    (np.ndarray): High frequency content of the flux of the model at 0 rv
        res_mod_obs_spectro         (np.ndarray): Resolution of the template interpolated onto the wavelength grid of the data
        wav_obs_spectro             (np.ndarray): Wavelength grid of the data
        flx_obs_spectro             (np.ndarray): Flux of the data
        err_obs_spectro             (np.ndarray): Error of the data
        flx_cont_obs_spectro        (np.ndarray): Continuum of the flux of the data
        res_obs_spectro             (np.ndarray): Resolution of the data
        res_cont                         (float): Resolution of the continuum
        wav_cont_cut                (np.ndarray): Wavelengths used for the continuum estimation
        star_flx_obs_spectro        (np.ndarray): Star flux of the data
        star_flx_cont_obs_spectro   (np.ndarray): Continuum of the flux of the star
        transm_obs_spectro          (np.ndarray): Transmission
        system_obs_spectro          (np.ndarray): Systematics

    Returns:
        ccf   (float): Correlation between the template and the data
        acf   (float): Autocorrelation between the template and iself

    Authors: Allan Denis
    '''

    # Doppler shifting
    wav_mod_spectro_doppler, flx_mod_spectro_doppler = doppler_fct(wav_mod_spectro, flx_mod_spectro, rv)
    # Resolution decreasing
    flx_mod_spectro_doppler = resolution_decreasing(wav_mod_spectro_doppler, flx_mod_spectro_doppler, res_mod_obs_spectro, wav_obs_spectro, res_obs_spectro)
    # Continuum estimation
    flx_cont_mod_spectro_doppler = continuum_estimate(wav_obs_spectro, flx_mod_spectro_doppler, res_obs_spectro, wav_cont_cut, res_cont)
    # CCF estimation
    ccf, flx_mod_spectro_ccf, speckles = utils.hc.hc_model(flx_obs_spectro, flx_cont_obs_spectro, transm_obs_spectro, star_flx_obs_spectro, star_flx_cont_obs_spectro, flx_mod_spectro_doppler, flx_cont_mod_spectro_doppler, err_obs_spectro, bounds=(0,'inf'))
    # ACF estimation
    flx_mod_spectro_rv_hf = transm_obs_spectro * (flx_mod_spectro_doppler - flx_cont_mod_spectro_doppler)
    acf = np.sum(flx_mod_spectro_rv_hf * flx_mod_spectro_no_rv_hf) / (np.sqrt(np.sum(flx_mod_spectro_rv_hf**2)) * np.sqrt(np.sum(flx_mod_spectro_no_rv_hf**2)))

    return ccf, acf