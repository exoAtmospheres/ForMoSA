import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import extinction
import astropy.units as u
import astropy.constants as const
from PyAstronomy.pyasl import rotBroad, fastRotBroad

# ----------------------------------------------------------------------------------------------------------------------


def convolve_and_sample(wv_channels, sigmas_wvs, model_wvs, model_fluxes, num_sigma=3, force_int=True): # num_sigma = 3 is a good compromise between sampling enough the gaussian and fast interpolation
    """
    Simulate the observations of a model. Convolves the model with a variable Gaussian LSF, sampled at each desired
    spectral channel.

    Args:
        wv_channels (list(floats)): the wavelengths values desired
        sigmas_wvs  (list(floats)): the LSF gaussian standard deviation of each wv_channels [IN UNITS OF model_wvs] 
        model_wvs          (array): the wavelengths of the model 
        model_fluxes       (array): the fluxes of the model 
        num_sigma          (float): number of +/- sigmas to evaluate the LSF to.
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


def resolution_decreasing(wav_input, flx_input, res_input, wav_output, res_output):
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
        fwhm_input = wav_output / res_input
        # Estimate of the FWHM of the output as a function of the wavelength
        fwhm_output = wav_output / res_output

        # Estimate of the sigma for the convolution as a function of the wavelength and decrease the resolution
        max_fwhm = np.nanmax([fwhm_input, fwhm_output], axis=0)
        min_fwhm = np.nanmin([fwhm_input, fwhm_output], axis=0)
        fwhm_conv = np.sqrt(max_fwhm ** 2 - min_fwhm ** 2)
        sigma_conv = fwhm_conv / 2.355
        flx_output = convolve_and_sample(wav_output, sigma_conv, wav_input, flx_input, force_int=True)

    return flx_output

# ----------------------------------------------------------------------------------------------------------------------


def continuum_estimate(wav_input, flx_input, res_input, wav_cont_bounds, res_cont):
    """
    Decrease the resolution of a spectrum (data or model). The function calculates the FWHM as a function of the
    wavelengths of the custom spectral resolution (estimated for the continuum). It then calculates a sigma to decrease
    the resolution of the spectrum to this custom FWHM for each wavelength using a gaussian filter and resample it on
    the wavelength grid of the data.

    Args:
        wav_input            (array): Wavelength grid of the spectrum for which you want to estimate the continuum
        flx_input            (array): Flux of the spectrum for which you want to estimate the continuum
        res_input            (array): Spectral resolution of the spectrum for which you want to estimate the continuum
        wav_cont_bounds      (array): Wavelength bounds where you want to estimate the continuum
        res_cont               (int): Approximate resolution of the continuum
    Returns:
        - continuum    (array): Estimated continuum of the spectrum re-sampled on the data wavelength grid

    Author: Simon Petrus, Matthieu Ravet

    """

    # Initialize
    flx_cont = np.asarray([])
    wav_cont = np.asarray([])
    # Redifined a spectrum only composed by the wavelength ranges used to estimate the continuum
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
        
        # Concatenate everything
        wav_cont = np.concatenate((wav_cont, wav_input[ind_cont_cut]))
        flx_cont = np.concatenate((flx_cont, cont))
        
    # Reinterpolate onto the original wavelength grid
    continuum_interp = interp1d(wav_cont, flx_cont, kind='linear', fill_value = 'extrapolate')
    continuum = continuum_interp(wav_input)

    return continuum



# ----------------------------------------------------------------------------------------------------------------------



def calc_flx_scale(obs_dict, flx_mod_spectro, flx_mod_photo, r_picked, d_picked, alpha=1, mode='physical', use_cov=False):
    """
    Calculation of the flux scaling factor (from the radius and distance or analytically).

    Args:
        obs_dict                (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)
        flx_mod_spectro        (array): Flux of the interpolated synthetic spectrum (spectroscopy)
        flx_mod_photo          (array): Flux of the interpolated synthetic spectrum (photometry)
        r_picked               (float): Radius randomly picked by the nested sampling (in RJup)
        d_picked               (float): Distance randomly picked by the nested sampling (in pc)
        alpha                  (float): Manual scaling factor (set to 1 by default) such that ck = alpha * (r/d)²
        mode                     (str): = 'physical' if the scaling needs to be calulated with r and d
                                        = 'analytic' if the scaling needs to be calculated analytically by the formula from Cushing et al. (2008)
        use_cov                 (bool): True or False if you want to use or not the full covariance matrix (formula from De Regt et al. (2025))
    Returns:
        - scale_spectro        (float): Flux scaling factor of the spectroscopy
        - scale_photo          (float): Flus scaling factor of the photometry

    Author: Simon Petrus 
    """
    # Calculation of the dilution factor as a function of the radius and distance
    if mode == 'physical':
        r_picked *= u.Rjup
        d_picked *= u.pc
        scale_spectro  = alpha * (r_picked.to(u.m).value/d_picked.to(u.m).value)**2
        scale_photo = scale_spectro
    # Calculation of the dilution factor analytically
    elif mode == 'analytic':
        if len(obs_dict['wav_spectro']) != 0:
            if use_cov == True:
                inv_cov_m = obs_dict['inv_cov'] @ flx_mod_spectro
                inv_cov_d = obs_dict['inv_cov'] @ obs_dict['flx_spectro']
                scale_spectro = (flx_mod_spectro @ inv_cov_d) / (flx_mod_spectro @ inv_cov_m)
            else:
                scale_spectro = np.sum((flx_mod_spectro * obs_dict['flx_spectro']) / (obs_dict['err_spectro'] * obs_dict['err_spectro'])) / np.sum((flx_mod_spectro / obs_dict['err_spectro'])**2)
        else:
            scale_spectro = 1
        if len(obs_dict['wav_photo']) != 0:
            scale_photo = np.sum((flx_mod_photo * obs_dict['flx_photo']) / (obs_dict['err_photo'] * obs_dict['err_photo'])) / np.sum((flx_mod_photo / obs_dict['err_photo'])**2)
        else:
            scale_photo = 1
    else:
        raise Exception(f'Mode {mode} unrecognize. It needs to be either "analytic" or "physical".')

    return scale_spectro*flx_mod_spectro, scale_photo*flx_mod_photo, scale_spectro, scale_photo



# ----------------------------------------------------------------------------------------------------------------------



def doppler_fct(wav_mod_spectro, flx_mod_spectro, rv_picked):
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



def reddening_fct(wav_mod_spectro, wav_obs_photo, flx_mod_spectro, flx_mod_photo, av_picked):
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



def vsini_fct(wav_mod_spectro, flx_mod_spectro, res_mod_obs_spectro, ld_picked, vsini_picked, vsini_type):
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



def vsini_fct_rot_broad(wav_mod_spectro, flx_mod_spectro, ld_picked, vsini_picked):
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



def vsini_fct_fast_rot_broad(wav_mod_spectro, flx_mod_spectro, ld_picked, vsini_picked):
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



def vsini_fct_accurate(wav_mod_spectro, flx_mod_spectro, ld_picked, vsini_picked, nr=50, ntheta=100, dif=0.0):
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



def vsini_fct_accurate_fast_rot_broad(wav_mod_spectro, flx_mod_spectro, ld_picked, vsini_picked):
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



def bb_cpd_fct(wav_mod_spectro, wav_obs_photo, flx_mod_spectro, flx_mod_photo, distance, bb_t_picked, bb_r_picked):
    '''
    Function to add the effect of a cpd (circum planetary disc) to the models.

    Args:
        wav_mod_spectro        (array): Wavelength grid of the model (spectroscopy)
        wav_obs_photo          (array): Wavelength of the data/model (photometry)
        flx_mod_spectro        (array): Flux of the interpolated synthetic spectrum (spectroscopy)
        flx_mod_photo          (array): Flux of the interpolated synthetic spectrum (photometry)
        distance               (array): Distance from the observation in pc units
        bb_temp                (float): Temperature value randomly picked by the nested sampling in K units
        bb_rad                 (float): Radius randomly picked by the nested sampling in units of planetary radius
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