from __future__ import division
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from spectres import spectres
import os
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------


def decoupe(second):
    """
    Re-arranged a number of seconds in the hours-minutes-seconds format.

    Args:
        second (float): number of second
    Returns:
        hour, minute, second (float, float, float): hours-minutes-seconds format

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
        idx     (int): Indice of the closest values from the desire value

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
        obs_spectro      (n-array): List containing the sub-spectra defined by the parameter "wav_for_adapt" with decreased resolution
                                    [[wav_1, flx_1, err_1, reso_1], ..., [wav_n, flx_n, err_n, reso_n]]
        obs_photo          (array): List containing the photometry (0 replace the spectral resolution here).
                                    [wav_phot, flx_phot, err_phot, 0]
        obs_spectro_ins    (array): List containing different instruments used for the data (1 per wavelength).
                                    [[instru_range_1], ..., [instru_range_n]]
        obs_photo_ins      (array): List containing different filters used for the data (1 per photometric point).
                                    [filter_phot_1, filter_phot_2, ..., filter_phot_n]
        obs_opt          (n-array): List containing the optional sub-arrays defined by the parameter "wav_for_adapt".
                                    [[cov_1, tran_1, star_1], ..., [cov_n, tran_n, star_n]]  

    Author: Simon Petrus / Adapted: Matthieu Ravet
    """

    # Extract the wavelengths, flux, errors, spectral resolution, and instrument/filter names from the observation file.

    obs_spectro, obs_photo, obs_spectro_ins, obs_photo_ins, obs_opt = adapt_observation_range(global_params, obs_name=obs_name, indobs=indobs)

    # Reduce the spectral resolution for each sub-spectrum.
    for c, cut in enumerate(obs_spectro):
        if len(cut[0]) != 0:
            # Interpolate the resolution of the model onto the wavelength of the data to properly decrease the resolution if necessary
            ind_mod_obs = np.where((wav_mod_nativ <= cut[0][-1]) & (wav_mod_nativ > cut[0][0]))
            wav_mod_obs = wav_mod_nativ[ind_mod_obs]
            res_mod_obs = res_mod_nativ[ind_mod_obs]
            interp_mod_to_obs = interp1d(wav_mod_obs, res_mod_obs, fill_value='extrapolate')
            res_mod_obs = interp_mod_to_obs(cut[0])
            # If we want to decrease the resolution of the data: (if by_sample, the data don't need to be adapted)
            if global_params.adapt_method[indobs] == 'by_reso':
                obs_spectro[c][1] = resolution_decreasing(global_params, cut[0], cut[1], cut[3], wav_mod_nativ, [], res_mod_obs,
                                                    'obs', obs_name=obs_name, indobs=indobs)
            if cont == 'yes':
            # If we want to estimate and substract the continuum of the data:
                obs_spectro[c][1] -= continuum_estimate(global_params, cut[0], cut[1], cut[3], obs_name=obs_name, indobs=indobs)
                
    return obs_spectro, obs_photo, obs_spectro_ins, obs_photo_ins, obs_opt

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
        obs_spectro   (n-array): List containing the sub-spectra defined by the parameter "wav_for_adapt".
                                [[wav_1, flx_1, err_1, reso_1], ..., [wav_n, flx_n, err_n, reso_n]]
        obs_photo       (array): List containing the photometry (0 replace the spectral resolution here).
                                [wav_phot, flx_phot, err_phot, 0]
        obs_spectro_ins (array): List containing different instruments used for the data (1 per wavelength).
                                [[instru_1], ..., [instru_n]]
        obs_photo_ins   (array): List containing different filters used for the data (1 per photometric point).
                                [filter_phot_1, filter_phot_2, ..., filter_phot_n]
        obs_opt       (n-array): List containing the optional sub-arrays defined by the parameter "wav_for_adapt".
                                [[cov_1, tran_1, star_1], ..., [cov_n, tran_n, star_n]]

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
        try:
            transm = hdul[1].data['TRANSM']
        except:
            transm = np.asarray([])
        try:
            star_flx = hdul[1].data['STAR_FLX1'][:,np.newaxis]
            is_star = True
        except:
            star_flx = np.asarray([])   
            is_star = False
        try:
            star_flx = hdul[1].data['STAR FLX'][:,np.newaxis]
        except:
            pass
        try:
            is_system = True
            system = hdul[1].data['SYSTEMATICS1'][:,np.newaxis]
        except:
            is_system = False
            system = np.asarray([])
            
        if is_system:
            i = 2
            while True:  # In case there is multiple systematics
                try:
                    system = np.concatenate((system, hdul[1].data['SYSTEMATICS' + str(i)][:,np.newaxis]),axis=1)
                    i += 1
                except:
                    break
                
        if is_star:
            i = 2
            while True:
                try:
                    star_flx = np.concatenate((star_flx, hdul[1].data['STAR_FLX' + str(i)][:,np.newaxis]),axis=1)
                    i += 1
                except:
                    break
        # Only take the covariance if you use the chi2_covariance likelihood function (will need to be change when new likelihood functions using the
        # covariance matrix will come)
        if global_params.logL_type[indobs] != 'chi2_covariance':
            cov = np.asarray([])

        # Filter the NaN and inf values
        if len(transm) != 0:
            nan_mod_ind = (~np.isnan(flx)) & (~np.isnan(transm)) & (~np.isnan(err)) & (np.isfinite(flx)) & (np.isfinite(transm)) & (np.isfinite(err))
        else:
            nan_mod_ind = (~np.isnan(flx)) & (~np.isnan(err)) & (np.isfinite(flx)) & (np.isfinite(err))
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
            
        # Select the wavelength range(s) for the extraction
        if global_params.wav_for_adapt == '':
            wav_for_adapt_tab = [str(min(wav)) + ',' + str(max(wav))]
        else:
            wav_for_adapt_tab = global_params.wav_for_adapt[indobs].split('/')


        # Photometry part of the data (OUT OF THE WINDOW LOOP)
        ind_photometry = np.where(res == 0.0)
        obs_photo = np.asarray([wav[ind_photometry], flx[ind_photometry], err[ind_photometry],
                res[ind_photometry]])
        obs_photo_ins = np.asarray(ins[ind_photometry])

        # Initiate spectroscopy data numpy arrays
        obs_spectro = np.empty(len(wav_for_adapt_tab), dtype=object)
        obs_opt = np.empty(len(wav_for_adapt_tab), dtype=object)
        obs_spectro_ins = np.empty(len(wav_for_adapt_tab), dtype=object)

        for range_ind, rangee in enumerate(wav_for_adapt_tab):
            rangee = rangee.split(',')
            ind = np.where((float(rangee[0]) <= wav) & (wav <= float(rangee[1])))
            ind_photometry = np.where(res[ind] == 0.0)

            # Spectroscopy part of the data
            wav_spectro = np.delete(wav[ind], ind_photometry)
            flx_spectro = np.delete(flx[ind], ind_photometry)
            err_spectro = np.delete(err[ind], ind_photometry)
            res_spectro = np.delete(res[ind], ind_photometry)
            ins_spectro = np.delete(ins[ind], ind_photometry)
            if len(cov) != 0: # Check if the covariance exists
                cov_spectro = cov[np.ix_(ind[0],ind[0])]
                cov_spectro = np.delete(cov_spectro, ind_photometry, axis=0)
                cov_spectro = np.delete(cov_spectro, ind_photometry, axis=1)
            else:
                cov_spectro = np.asarray([])

            if len(transm) != 0:
                transm_spectro = np.delete(transm[ind], ind_photometry)
            else:
                transm_spectro = np.asarray([])
            
            if len(star_flx) != 0:
                star_flx_spectro = np.delete(star_flx[ind,:], ind_photometry, axis=0)
            else:
                star_flx_spectro = np.asarray([])
                
            if len(system) != 0:
                system_spectro = np.delete(system[ind,:], ind_photometry, axis=0)
            else:
                system_spectro = np.asarray([])

            # Merge spectroscopic data
            obs_spectro[range_ind] = [wav_spectro, flx_spectro, err_spectro, res_spectro]
            obs_opt[range_ind] = [cov_spectro, transm_spectro, star_flx_spectro, system_spectro]
            obs_spectro_ins[range_ind] = ins_spectro
            
        return obs_spectro, obs_photo, obs_spectro_ins, obs_photo_ins, obs_opt   


# ----------------------------------------------------------------------------------------------------------------------


def adapt_model(global_params, wav_mod_nativ, flx_mod_nativ, res_mod_obs_merge, obs_name='', indobs=0):
    """
    Extracts a synthetic spectrum from a grid and decreases its spectral resolution. The photometry points are
    calculated too. Then each sub-spectrum are merged.

    Args:
        global_params  (object): Class containing each parameter used in ForMoSA
        wav_mod_nativ   (array): Wavelength grid of the model
        wave_reso_tab   (array): Wavelength grid of the model at specified resolution
        flx_mod_nativ   (array): Flux of the model
        res_mod_nativ   (array): Spectral resolution of the model as a function of the wavelength grid
        obs_name          (str): Name of the current observation looping
        indobs            (int): Index of the current observation looping
    Returns:
        flx_mod_extract (array): Flux of the spectrum with a decreased spectral resolution, re-sampled on the data wavelength grid
        mod_photo       (array): List containing the photometry ('0' replace the spectral resolution here).

    Author: Simon Petrus
    """
    # Estimate and subtract the continuum (if needed)
    if global_params.continuum_sub[indobs] != 'NA':
        mod_spectro, mod_photo = extract_model(global_params, wav_mod_nativ, flx_mod_nativ, res_mod_obs_merge, cont='yes', obs_name=obs_name)
    else:
        mod_spectro, mod_photo = extract_model(global_params, wav_mod_nativ, flx_mod_nativ, res_mod_obs_merge, obs_name=obs_name)

    return mod_spectro, mod_photo

# ----------------------------------------------------------------------------------------------------------------------


def extract_model(global_params, wav_mod_nativ, flx_mod_nativ, res_mod_obs_merge, cont='no', obs_name='', indobs=0):
    """
    Extracts a synthetic spectrum from a grid and decreases its spectral resolution. The photometry points are
    calculated too.

    Args:
        global_params (object): Class containing each parameter used in ForMoSA
        wav_mod_nativ  (array): Wavelength grid of the model
        flx_mod_nativ  (array): Flux of the model
        res_mod_nativ  (array): Spectral resolution of the model as a function of the wavelength grid
        cont             (str): Boolean string. If the function is used to estimate the continuum cont='yes'
        obs_name         (str): Name of the current observation looping
        indobs           (int): Index of the current observation looping
    Returns:
        mod_spectro    (array): List containing the sub-spectra defined by the parameter "wav_for_adapt".
        mod_photo      (array): List containing the photometry ('0' replace the spectral resolution here).

    Author: Simon Petrus
    """
    # Take back the extracted data.
    spectrum_obs = np.load(os.path.join(global_params.result_path, f'spectrum_obs_{obs_name}.npz'), allow_pickle=True)
    obs_spectro = spectrum_obs['obs_spectro']
    obs_photo_ins = spectrum_obs['obs_photo_ins']

    # Reduce the spectral resolution for each sub-spectrum.
    for c, cut in enumerate(obs_spectro):
        if len(cut[0]) != 0:
            # If we want to decrease the resolution of the data:
            if global_params.adapt_method[indobs] == 'by_reso':
                mod_cut_flx = resolution_decreasing(global_params, cut[0], [], cut[3], wav_mod_nativ, flx_mod_nativ, res_mod_obs_merge[c],
                                                    'mod', obs_name=obs_name, indobs=indobs)
            else:
                mod_cut_flx = spectres(cut[0], wav_mod_nativ, flx_mod_nativ)

            # If we want to estimate the continuum of the data:
            if cont == 'yes':     
                continuum = continuum_estimate(global_params, cut[0], mod_cut_flx, res_mod_obs_merge[c], obs_name=obs_name, indobs=indobs)
                mod_cut_flx -= continuum

            # Concatenate to speed up the code
            if c==0:
                mod_spectro = mod_cut_flx
            else:
                mod_spectro = np.concatenate((mod_spectro, mod_cut_flx))
              
        else:
            mod_cut_flx = np.asarray([])

    # Calculate each photometry point.
    mod_photo = []
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
        mod_photo.append(flx_filt)

    return mod_spectro, mod_photo

# ----------------------------------------------------------------------------------------------------------------------


def convolve_and_sample(wv_channels, sigmas_wvs, model_wvs, model_fluxes, num_sigma=1):
    """
    Simulate the observations of a model. Convolves the model with a variable Gaussian LSF, sampled at each desired
    spectral channel.

    Args:
        wv_channels (list(floats)): the wavelengths values desired
        sigmas_wvs  (list(floats)): the LSF gaussian standard deviation of each wv_channels [IN UNITS OF model_wvs] 
        model_wvs          (array): the wavelengths of the model 
        model_fluxes       (array): the fluxes of the model 
        num_sigma          (float): number of +/- sigmas to evaluate the LSF to.
    Returns:
        output_model       (array): the fluxes in each of the wavelength channels 

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
      
        model_interp = interp1d(model_wvs, model_fluxes, kind='cubic', bounds_error=False)
        filter_model = model_interp(filter_wv_coords)

        output_model = np.nansum(filter_model * lsf, axis=1) / np.sum(lsf, axis=1)
    else:

        output_model = model_fluxes

    return output_model

# ----------------------------------------------------------------------------------------------------------------------


def resolution_decreasing(global_params, wav_obs, flx_obs, res_obs, wav_mod_nativ, flx_mod_nativ, res_mod_obs, obs_or_mod, obs_name='', indobs=0):
    """
    Decrease the resolution of a spectrum (data or model). The function calculates the FWHM as a function of the
    wavelengths for the data, the model, and for a custom spectral resolution (optional) and estimates the highest one
    for each wavelength (the lowest spectral resolution). It then calculates a sigma to decrease the resolution of the
    spectrum to this lowest FWHM for each wavelength and resample it on the wavelength grid of the data using the
    function 'convolve_and_sample'.

    Args:
        global_params (object): Class containing each parameter used in ForMoSA
        wav_obs        (array): Wavelength grid of the data
        flx_obs        (array): Flux of the data
        res_obs        (array): Spectral resolution of the data
        wav_mod_nativ  (array): Wavelength grid of the model
        flx_mod_nativ  (array): Flux of the model
        res_mod_obs    (array): Spectral resolution of the model as a function of the wavelength grid of the data
        obs_or_mod       (str): Parameter to identify if you want to manage a data or a model spectrum. 'obs' or 'mod'
        obs_name         (str): Name of the current observation looping
        indobs           (int): Index of the current observation looping
    Returns:
        flx_obs_final  (array): Flux of the spectrum with a decreased spectral resolution, re-sampled on the data wavelength grid

    Author: Simon Petrus / Adapted: Matthieu Ravet
    """
    # Estimate of the FWHM of the data as a function of the wavelength
    fwhm_obs = 2 * wav_obs / res_obs
    # Estimate of the FWHM of the model as a function of the wavelength
    fwhm_mod = 2 * wav_obs / res_mod_obs

    # Estimate of the FWHM of the custom resolution (if defined) as a function of the wavelength
    if global_params.custom_reso[indobs] != 'NA':
        fwhm_custom = 2 * wav_obs / float(global_params.custom_reso[indobs])
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
        flx_obs_final = convolve_and_sample(wav_obs, sigma_conv, wav_mod_nativ, flx_mod_nativ)

    return flx_obs_final

# ----------------------------------------------------------------------------------------------------------------------


def continuum_estimate(global_params, wav, flx, res, obs_name='', indobs=0):
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
        obs_name         (str): Name of the current observation looping
        indobs           (int): Index of the current observation looping
    Returns:
        continuum      (array): Estimated continuum of the spectrum re-sampled on the data wavelength grid

    Author: Simon Petrus / Adapted: Matthieu Ravet

    """

    # Redifined a spectrum only composed by the wavelength ranges used to estimate the continuum
    wav_for_continuum = global_params.wav_for_continuum[indobs].split('/')
    for wav_for_cont_cut_ind, wav_for_cont_cut in enumerate(wav_for_continuum):
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

    fwhm = 2 * wav_median / np.median(res)
    fwhm_continuum = 2 * wav_median / float(global_params.continuum_sub[indobs])


    fwhm_conv = np.sqrt(fwhm_continuum**2 - fwhm**2)
    sigma = fwhm_conv / (dwav_median * 2.355)
    continuum = gaussian_filter(flx, sigma)
    # import scipy.signal as sg
    # continuum = sg.savgol_filter(flx, 3001, 2)

    return continuum

# ----------------------------------------------------------------------------------------------------------------------