from __future__ import division
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from spectres import spectres
# ----------------------------------------------------------------------------------------------------------------------


def decoupe(second):
    """
    Re-arranged a number of seconds in the hours-minutes-seconds format.

    Args:
        second (float): number of second
    Returns:
        - float     : hours
        - float     : minutes
        - float     : seconds

    Author: Simon Petrus
    """

    hour = second / 3600
    second %= 3600
    minute = second / 60
    second %= 60

    return hour, minute, second

# ----------------------------------------------------------------------------------------------------------------------


def find_nearest(array, value):
    """
    Return the indice of the closest values from a desire value in an array.

    Args:
        array (array): Array to explore
        value (float): Desire value
    Returns:
        - idx (int)          : Indice of the closest values from the desire value

    Author: Simon Petrus
    """
    idx = (np.abs(array - value)).argmin()

    return idx

# ----------------------------------------------------------------------------------------------------------------------


def extract_observation(global_params, wav_mod_nativ, res_mod_nativ, cont='no', obs_name='', indobs=0):
    """
    Take back the extracted data spectrum from the function 'adapt_observation_range' and decrease its spectral
    resolution.

    Args:
        global_params     (object): Class containing each parameter
        wav_mod_nativ      (array): Wavelength grid of the model
        res_mod_nativ (array(int)): Spectral resolution of the model
        cont                 (str): Boolean string. If the function is used to estimate the continuum cont='yes'
        obs_name             (str): Name of the current observation looping
        indobs               (int): Index of the current observation looping

    Returns:
        - obs_spectro (array)       : List containing the sub-spectra defined by the parameter "wav_for_adapt" with decreased resolution  [wav, flx, err, reso]
        - obs_photo (array)         : List containing the photometry (0 replace the spectral resolution here).  [wav_phot, flx_phot, err_phot, 0]
        - obs_photo_ins (array)     : List containing different filters used for the data (1 per photometric point). [filter_phot_1, filter_phot_2, ..., filter_phot_n]
        - obs_opt (array)           : List containing the optional sub-arrays defined by the parameter "wav_for_adapt". [cov, tran, star, system] 

    Author: Simon Petrus, Matthieu Ravet
    """

    # Extract the wavelengths, flux, errors, spectral resolution, and instrument/filter names from the observation file.
    obs_spectro, obs_photo, obs_photo_ins, obs_opt = adapt_observation_range(global_params, obs_name=obs_name, indobs=indobs)

    # Reduce the spectral resolution for each sub-spectrum.
    for range_ind, rangee in enumerate(global_params.wav_for_adapt[indobs].split('/')):
        rangee = rangee.split(',')
        mask_spectro_cut = (float(rangee[0]) <= obs_spectro[0]) & (obs_spectro[0] <= float(rangee[1]))
        if len(obs_spectro[0][mask_spectro_cut]) != 0:
            # Interpolate the resolution of the model onto the wavelength of the data to properly decrease the resolution if necessary
            mask_mod_obs = (wav_mod_nativ <= obs_spectro[0][mask_spectro_cut][-1]) & (wav_mod_nativ > obs_spectro[0][mask_spectro_cut][0])
            wav_mod_obs = wav_mod_nativ[mask_mod_obs]
            res_mod_obs = res_mod_nativ[mask_mod_obs]
            interp_mod_to_obs = interp1d(wav_mod_obs, res_mod_obs, fill_value='extrapolate')
            res_mod_obs = interp_mod_to_obs(obs_spectro[0][mask_spectro_cut])
            # If we want to decrease the resolution of the data: (if by_sample, the data don't need to be adapted)
            if global_params.adapt_method[indobs] == 'by_reso':
                obs_spectro[1][mask_spectro_cut] = resolution_decreasing(global_params,
                                                                         obs_spectro[0][mask_spectro_cut],
                                                                         obs_spectro[1][mask_spectro_cut],
                                                                         obs_spectro[3][mask_spectro_cut],
                                                                         wav_mod_nativ,
                                                                         [], 
                                                                         res_mod_obs,
                                                                         'obs', indobs=indobs)
            # If we want to estimate and substract the continuum of the data:
            if cont == 'yes':
                obs_spectro[1][mask_spectro_cut] -= continuum_estimate(global_params,
                                                                       obs_spectro[0][mask_spectro_cut],
                                                                       obs_spectro[1][mask_spectro_cut],
                                                                       obs_spectro[3][mask_spectro_cut], indobs=indobs)
                
    return obs_spectro, obs_photo, obs_photo_ins, obs_opt

# ----------------------------------------------------------------------------------------------------------------------


def adapt_observation_range(global_params, obs_name='', indobs=0):
    """
    Extract the information from the observation file, including the wavelengths (um - vacuum), flux (W.m-2.um.1), errors (W.m-2.um.1), covariance (W.m-2.um.1)**2, spectral resolution, 
    instrument/filter name, transmission (Atmo+inst) and star flux (W.m-2.um.1). The wavelength range is define by the parameter "wav_for_adapt".

    Args:
        global_params  (object): Class containing each parameter
        obs_name          (str): Name of the current observation looping
        indobs            (int): Index of the current observation looping

    Returns:
        - obs_spectro (array)     : List containing the sub-spectra defined by the parameter "wav_for_adapt" with decreased resolution  [wav, flx, err, reso]
        - obs_photo (array)       : List containing the photometry (0 replace the spectral resolution here).  [wav_phot, flx_phot, err_phot, 0]
        - obs_photo_ins (array)   : List containing different filters used for the data (1 per photometric point). [filter_phot_1, filter_phot_2, ..., filter_phot_n]
        - obs_opt (array)         : List containing the optional sub-arrays defined by the parameter "wav_for_adapt". [cov, tran, star, system] 

    Author: Simon Petrus, Matthieu Ravet and Allan Denis
    """
    # Extraction
    with fits.open(global_params.observation_path) as hdul:

        # Check the format of the file and extract data accordingly
        wav = hdul[1].data['WAV']
        flx = hdul[1].data['FLX']
        res = hdul[1].data['RES']
        ins = hdul[1].data['INS']
        try: # Check for spectral covariances
            err = hdul[1].data['ERR']
            cov = np.asarray([]) # Create an empty covariance matrix if not already present in the data (to not slow the inversion)
        except:
            cov = hdul[1].data['COV']
            err = np.sqrt(np.diag(np.abs(cov)))
        try: # Check for transmission
            transm = hdul[1].data['TRANSM']
        except:
            transm = np.asarray([])
        try: # Check for star flux
            star_flx = hdul[1].data['STAR_FLX1'][:,np.newaxis]
            is_star = True
        except:
            star_flx = np.asarray([])   
            is_star = False
        if is_star:
            i = 2
            while True: # In case there is multiple star flux (usually shifted to account for the PSF)
                try:
                    star_flx = np.concatenate((star_flx, hdul[1].data['STAR_FLX' + str(i)][:,np.newaxis]),axis=1)
                    i += 1
                except:
                    break
        try:
            is_system = True
            system = hdul[1].data['SYSTEMATICS1'][:,np.newaxis]
        except:
            is_system = False
            system = np.asarray([])
        if is_system:
            i = 2
            while True: # In case there is multiple systematics
                try:
                    system = np.concatenate((system, hdul[1].data['SYSTEMATICS' + str(i)][:,np.newaxis]),axis=1)
                    i += 1
                except:
                    break

        # Only take the covariance if you use the chi2_covariance likelihood function (will need to be change when new likelihood functions using the
        # covariance matrix will come)
        if global_params.logL_type[indobs] != 'chi2_covariance':
            cov = np.asarray([])

        # Filter the NaN and inf values
        nan_mod_ind = (~np.isnan(flx)) & (~np.isnan(err)) & (np.isfinite(flx)) & (np.isfinite(err))
        if len(cov) != 0:
            nan_mod_ind = (nan_mod_ind) & np.all(~np.isnan(cov), axis=0) & np.all(~np.isnan(cov), axis=1) & np.all(np.isfinite(cov), axis=0) & np.all(np.isfinite(cov), axis=1)
        if len(transm) != 0:
            nan_mod_ind = (nan_mod_ind) & (~np.isnan(transm)) & (np.isfinite(transm))
        if len(star_flx) != 0:
            for i in range(len(star_flx[0])):
                nan_mod_ind = (nan_mod_ind) & (~np.isnan(star_flx.T[i])) & (np.isfinite(star_flx.T[i]))
        if len(system) != 0:
            for i in range(len(system[0])):
                nan_mod_ind = (nan_mod_ind) & (~np.isnan(system.T[i])) & (np.isfinite(system.T[i])) 
        wav = wav[nan_mod_ind]
        flx = flx[nan_mod_ind]
        res = res[nan_mod_ind]
        ins = ins[nan_mod_ind]
        err = err[nan_mod_ind]
        if len(cov) != 0:
            cov = np.transpose(np.transpose(cov[nan_mod_ind])[nan_mod_ind])
        if len(transm) != 0 and len(star_flx) != 0:
            transm = transm[nan_mod_ind]
        if len(star_flx) != 0:
            star_flx = np.delete(star_flx, np.where(~nan_mod_ind), axis=0)
        if len(system) != 0:
            system = np.delete(system, np.where(~nan_mod_ind), axis=0)
            
        # - - - - - - - - - 

        # Separate photometry and spectroscopy + cuts
        mask_photo = (res == 0.0)

        # Photometry part
        obs_photo = np.asarray([wav[mask_photo],
                                flx[mask_photo],
                                err[mask_photo],
                                res[mask_photo]])
        obs_photo_ins = np.asarray(ins[mask_photo])

        # Spectroscopy part
        wav_spectro = wav[~mask_photo]
        flx_spectro = flx[~mask_photo]
        err_spectro = err[~mask_photo]
        res_spectro = res[~mask_photo]
        mask_spectro = np.zeros(len(wav_spectro), dtype=bool)
        for range_ind, rangee in enumerate(global_params.wav_for_adapt[indobs].split('/')):
            rangee = rangee.split(',')
            mask_spectro += (float(rangee[0]) <= wav_spectro) & (wav_spectro <= float(rangee[1]))
        obs_spectro = np.asarray([wav_spectro[mask_spectro],
                                  flx_spectro[mask_spectro],
                                  err_spectro[mask_spectro],
                                  res_spectro[mask_spectro]])

        # Optional arrays
        if len(cov) != 0: # Check if the covariance exists
            cov_spectro = cov[np.ix_(~mask_photo,~mask_photo)]
            inv_cov_spectro = np.linalg.inv(cov_spectro[np.ix_(mask_spectro,mask_spectro)]) # Save only the inverse covariance to speed up the inversion
        else:
            inv_cov_spectro = np.asarray([])
        if len(transm) != 0:
            transm_spectro = transm[~mask_photo][mask_spectro]
        else:
            transm_spectro = np.asarray([])
        if len(star_flx) != 0:
            star_flx_spectro = star_flx[~mask_photo][mask_spectro]
        else:
            star_flx_spectro = np.asarray([])
        if len(system) != 0:
            system_spectro = system[~mask_photo][mask_spectro]
        else:
            system_spectro = np.asarray([])
        obs_opt = np.asarray([inv_cov_spectro,
                            transm_spectro,
                            star_flx_spectro,
                            system_spectro], dtype=object)
        
        return obs_spectro, obs_photo, obs_photo_ins, obs_opt   


# ----------------------------------------------------------------------------------------------------------------------


def adapt_model(global_params, wav_mod_nativ, flx_mod_nativ, res_mod_obs, wav_obs_spectro, res_obs_spectro, obs_photo_ins, obs_name='', indobs=0):
    """
    Extracts a synthetic spectrum from a grid and decreases its spectral resolution. The photometry points are
    calculated too. Then each sub-spectrum are merged.

    Args:
        global_params  (object): Class containing each parameter used in ForMoSA
        wav_mod_nativ   (array): Wavelength grid of the model
        flx_mod_nativ   (array): Flux of the model
        res_mod_obs     (array): Spectral resolution of the model interpolated at wav_obs_spectro
        wav_obs_spectro (array): Wavelength grid of the spectroscopic data
        res_obs_spectro (array): Spectral resolution grid of the spectroscopic data
        obs_photo_ins   (array): List containing different filters used for the data (1 per photometric point). [filter_phot_1, filter_phot_2, ..., filter_phot_n]
        wav_obs
        obs_name          (str): Name of the current observation looping
        indobs            (int): Index of the current observation looping
    Returns:
        - mod_spectro   (array): Flux of the spectrum with a decreased spectral resolution, re-sampled on the data wavelength grid
        - mod_photo     (array): List containing the photometry ('0' replace the spectral resolution here).

    Author: Simon Petrus, Matthieu Ravet
    """
    # Estimate and subtract the continuum (if needed)
    if global_params.continuum_sub[indobs] != 'NA':
        mod_spectro, mod_photo = extract_model(global_params,
                                               wav_mod_nativ,
                                               flx_mod_nativ,
                                               res_mod_obs,
                                               wav_obs_spectro,
                                               res_obs_spectro,
                                               obs_photo_ins,
                                               cont='yes', obs_name=obs_name, indobs=indobs)
    else:
        mod_spectro, mod_photo = extract_model(global_params,
                                               wav_mod_nativ,
                                               flx_mod_nativ,
                                               res_mod_obs,
                                               wav_obs_spectro,
                                               res_obs_spectro,
                                               obs_photo_ins,
                                               obs_name=obs_name, indobs=indobs)

    return mod_spectro, mod_photo

# ----------------------------------------------------------------------------------------------------------------------


def extract_model(global_params, wav_mod_nativ, flx_mod_nativ, res_mod_obs, wav_obs_spectro, res_obs_spectro, obs_photo_ins, cont='no', obs_name='', indobs=0):
    """
    Extracts a synthetic spectrum from a grid and decreases its spectral resolution. The photometry points are
    calculated too.

    Args:
        global_params  (object): Class containing each parameter used in ForMoSA
        wav_mod_nativ   (array): Wavelength grid of the model
        flx_mod_nativ   (array): Flux of the model
        res_obs_mod     (array): Spectral resolution of the model interpolated at wav_obs_spectro
        wav_obs_spectro (array): Wavelength grid of the spectroscopic data
        res_obs_spectro (array): Spectral resolution grid of the spectroscopic data
        cont              (str): Boolean string. If the function is used to estimate the continuum cont='yes'
        obs_name          (str): Name of the current observation looping
        indobs            (int): Index of the current observation looping
    Returns:
        - mod_spectro   (array): List containing the sub-spectra defined by the parameter "wav_for_adapt".
        - mod           (array): List containing the photometry ('0' replace the spectral resolution here).

    Author: Simon Petrus, Matthieu Ravet
    """
    # Create final models
    mod_spectro, mod_photo = np.empty(len(wav_obs_spectro), dtype=float), np.empty(len(obs_photo_ins), dtype=float)
    # Reduce the spectral resolution for each sub-spectrum.
    for range_ind, rangee in enumerate(global_params.wav_for_adapt[indobs].split('/')):
        rangee = rangee.split(',')
        mask_spectro_cut = (float(rangee[0]) <= wav_obs_spectro) & (wav_obs_spectro <= float(rangee[1]))
        if len(wav_obs_spectro[mask_spectro_cut]) != 0:
            # If we want to decrease the resolution of the data:
            if global_params.adapt_method[indobs] == 'by_reso':
                mod_spectro[mask_spectro_cut] = resolution_decreasing(global_params, wav_obs_spectro[mask_spectro_cut], [], res_obs_spectro[mask_spectro_cut], wav_mod_nativ, flx_mod_nativ, res_mod_obs[mask_spectro_cut],
                                                    'mod', indobs=indobs)
            else:
                mod_spectro[mask_spectro_cut] = spectres(wav_obs_spectro[mask_spectro_cut], wav_mod_nativ, flx_mod_nativ)

            # If we want to estimate the continuum of the data:
            if cont == 'yes':     
                continuum = continuum_estimate(global_params, wav_obs_spectro[mask_spectro_cut], mod_spectro[mask_spectro_cut], res_mod_obs[mask_spectro_cut], indobs=indobs)
                mod_spectro[mask_spectro_cut] -= continuum


    # Calculate each photometry point.
    for pho_ind, pho in enumerate(obs_photo_ins):
        path_list = __file__.split("/")[:-2]
        separator = '/'
        filter_pho = np.load(separator.join(path_list) + '/phototeque/' + pho + '.npz')
        x_filt = filter_pho['x_filt']
        y_filt = filter_pho['y_filt']
        filter_interp = interp1d(x_filt, y_filt, fill_value="extrapolate")
        y_filt = filter_interp(wav_mod_nativ)

        ind = np.where(np.logical_and(wav_mod_nativ > min(x_filt), wav_mod_nativ < max(x_filt)))
        flx_filt = np.sum(flx_mod_nativ[ind] * y_filt[ind] * (wav_mod_nativ[ind][1] - wav_mod_nativ[ind][0]))
        y_filt_tot = np.sum(y_filt[ind] * (wav_mod_nativ[ind][1] - wav_mod_nativ[ind][0]))
        flx_filt = flx_filt / y_filt_tot
        mod_photo[pho_ind] = flx_filt

    return mod_spectro, mod_photo

# ----------------------------------------------------------------------------------------------------------------------


def convolve_and_sample(wv_channels, sigmas_wvs, model_wvs, model_fluxes, num_sigma=3, force_int=False): # num_sigma = 3 is a good compromise between sampling enough the gaussian and fast interpolation
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


    if np.sum(lsf) != 0:
        left_fill = model_fluxes[model_in_range][0]
        right_fill = model_fluxes[model_in_range][-1]
        model_interp = interp1d(model_wvs, model_fluxes, kind='cubic', bounds_error=False, fill_value=(left_fill,right_fill))
        filter_model = model_interp(filter_wv_coords)

        output_model = np.nansum(filter_model * lsf, axis=1) / np.sum(lsf, axis=1)
    else:
        if force_int == True:
            model_interp = interp1d(model_wvs, model_fluxes, kind='cubic', bounds_error=False)
            output_model = model_interp(wv_channels)            
        else:
            output_model = model_fluxes

    return output_model

# ----------------------------------------------------------------------------------------------------------------------


def resolution_decreasing(global_params, wav_obs, flx_obs, res_obs, wav_mod_nativ, flx_mod_nativ, res_mod_obs, obs_or_mod, indobs=0):
    """
    Decrease the resolution of a spectrum (data or model). The function calculates the FWHM as a function of the
    wavelengths for the data, the model, and for a custom spectral resolution (optional) and estimates the highest one
    for each wavelength (the lowest spectral resolution). It then calculates a sigma to decrease the resolution of the
    spectrum to this lowest FWHM for each wavelength and resample it on the wavelength grid of the data using the
    function 'convolve_and_sample'.

    Args:
        global_params   (object): Class containing each parameter used in ForMoSA
        wav_obs          (array): Wavelength grid of the data
        flx_obs          (array): Flux of the data
        res_obs          (array): Spectral resolution of the data
        wav_mod_nativ    (array): Wavelength grid of the model
        flx_mod_nativ    (array): Flux of the model
        res_mod_obs      (array): Spectral resolution of the model as a function of the wavelength grid of the data
        obs_or_mod         (str): Parameter to identify if you want to manage a data or a model spectrum. 'obs' or 'mod'
        indobs             (int): Index of the current observation looping
    Returns:
        - flx_obs_final  (array): Flux of the spectrum with a decreased spectral resolution, re-sampled on the data wavelength grid

    Author: Simon Petrus
    """
    # Estimate of the FWHM of the data as a function of the wavelength
    fwhm_obs = wav_obs / res_obs
    # Estimate of the FWHM of the model as a function of the wavelength
    fwhm_mod = wav_obs / res_mod_obs

    # Estimate of the FWHM of the custom resolution (if defined) as a function of the wavelength
    if global_params.custom_reso[indobs] != 'NA':
        fwhm_custom = wav_obs / float(global_params.custom_reso[indobs])
    else:
        fwhm_custom = wav_obs * np.nan


    # Estimate of the sigma for the convolution as a function of the wavelength and decrease the resolution
    max_fwhm = np.nanmax([fwhm_obs, fwhm_mod, fwhm_custom], axis=0)
    if obs_or_mod == 'obs':
        fwhm_conv = np.sqrt(max_fwhm ** 2 - fwhm_obs ** 2)
        sigma_conv = fwhm_conv / 2.355
        flx_obs_final = convolve_and_sample(wav_obs, sigma_conv, wav_obs, flx_obs)
    else:
        fwhm_conv = np.sqrt(max_fwhm ** 2 - fwhm_mod ** 2)
        sigma_conv = fwhm_conv / 2.355
        flx_obs_final = convolve_and_sample(wav_obs, sigma_conv, wav_mod_nativ, flx_mod_nativ, force_int=True)

    return flx_obs_final

# ----------------------------------------------------------------------------------------------------------------------


def continuum_estimate(global_params, wav, flx, res, indobs=0):
    """
    Decrease the resolution of a spectrum (data or model). The function calculates the FWHM as a function of the
    wavelengths of the custom spectral resolution (estimated for the continuum). It then calculates a sigma to decrease
    the resolution of the spectrum to this custom FWHM for each wavelength using a gaussian filter and resample it on
    the wavelength grid of the data.

    Args:
        global_params (object): Class containing each parameter used in ForMoSA
        wav            (array): Wavelength grid of the spectrum for which you want to estimate the continuum
        flx            (array): Flux of the spectrum for which you want to estimate the continuum
        res              (int): Spectral resolution of the spectrum for which you want to estimate the continuum
        indobs           (int): Index of the current observation looping
    Returns:
        - continuum    (array): Estimated continuum of the spectrum re-sampled on the data wavelength grid

    Author: Simon Petrus, Matthieu Ravet

    """

    # Redifined a spectrum only composed by the wavelength ranges used to estimate the continuum
    for wav_for_cont_cut_ind, wav_for_cont_cut in enumerate(global_params.wav_for_continuum[indobs].split('/')):
        wav_for_cont_cut = wav_for_cont_cut.split(',')
        ind_cont_cut = np.where((float(wav_for_cont_cut[0]) <= wav) & (wav <= float(wav_for_cont_cut[1])))
        if wav_for_cont_cut_ind == 0:
            wav_for_cont_final = wav[ind_cont_cut]
            flx_for_cont_final = flx[ind_cont_cut]
        else:
            wav_for_cont_final = np.concatenate((wav_for_cont_final, wav[ind_cont_cut]))
            flx_for_cont_final = np.concatenate((flx_for_cont_final, flx[ind_cont_cut]))

    model_interp = interp1d(wav_for_cont_final, flx_for_cont_final, kind='linear', bounds_error=False)
    flx = model_interp(wav)

    # # To limit the computing time, the convolution is not as a function of the wavelength but calculated
    # from the median wavelength. We just want an estimate of the continuum here.
    wav_median = np.median(wav)
    dwav_median = np.median(np.abs(wav - np.roll(wav, 1))) # Estimated the median wavelength separation instead of taking wav_median - (wav_median+1) that could be on a border

    fwhm = wav_median / np.median(res)
    fwhm_continuum = wav_median / float(global_params.continuum_sub[indobs])


    fwhm_conv = np.sqrt(fwhm_continuum**2 - fwhm**2)
    sigma = fwhm_conv / (dwav_median * 2.355)
    continuum = gaussian_filter(flx, sigma)

    return continuum

# ----------------------------------------------------------------------------------------------------------------------