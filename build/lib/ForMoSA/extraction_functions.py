from __future__ import division
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from spectres import spectres
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------


def decoupe(second):
    """
    Re-arranged a number of seconds in the hours-minutes-seconds format.

    Args:
        second: number of second
    Returns:
        hour, minute, second: hours-minutes-seconds format

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
        array: Array to explore
        value: Desire value
    Returns:
        idx: Indice of the closest values from the desire value

    Author: Simon Petrus
    """
    idx = (np.abs(array - value)).argmin()

    return idx

# ----------------------------------------------------------------------------------------------------------------------


def extract_observation(global_params, wav_mod_nativ, res_mod_nativ, cont='no'):
    """
    Take back the extracted data spectrum from the function 'adapt_observation_range' and decrease its spectral
    resolution.

    Args:
        global_params: Class containing each parameter
        wav_mod_nativ: Wavelength grid of the model
        res_mod_nativ: Spectral resolution of the model
        cont: If the function is used to estimate the continuum cont='yes'
    Returns:
        obs_cut: List containing the sub-spectra defined by the parameter "wav_for_adapt" with decreased resolution
                 [[wav_1, flx_1, err_1, reso_1], ..., [wav_n, flx_n, err_n, reso_n]]
        obs_pho: List containing the photometry ('P' replace the spectral resolution here).
                 [wav_phot, flx_phot, err_phot, 'P']
        obs_cut_ins: List containing different instruments used for the data (1 per wavelength).
                 [[instru_range_1], ..., [instru_range_n]]
        obs_pho_ins: List containing different filters used for the data (1 per photometric point).
                 [filter_phot_1, filter_phot_2, ..., filter_phot_n]

    Author: Simon Petrus
    """

    # Extract the wavelengths, flux, errors, spectral resolution, and instrument/filter names from the observation file.
    obs_cut, obs_pho, obs_cut_ins, obs_pho_ins = adapt_observation_range(global_params)

    # Reduce the spectral resolution for each sub-spectrum.
    for c, cut in enumerate(obs_cut):
        # If we want to decrease the resolution of the data:
        if cont == 'no':
            if global_params.adapt_method == 'by_reso':
                obs_cut[c][1] = resolution_decreasing(global_params, cut, wav_mod_nativ, [], res_mod_nativ, 'obs')
            else:
                pass
        # If we want to estimate the continuum of the data:
        else:
            obs_cut[c][1] = continuum_estimate(global_params, cut[0], cut[0], cut[1], cut[3], 'obs')

    return obs_cut, obs_pho, obs_cut_ins, obs_pho_ins

# ----------------------------------------------------------------------------------------------------------------------


def adapt_observation_range(global_params):
    """
    Extract the wavelengths (um  -  vacuum), flux (W.m-2.um.1), errors (W.m-2.um.1), spectral resolution, and
    instrument/filter names from the observation file. The wavelength range is define by the parameter "wav_for_adapt".

    Args:
        global_params: Class containing each parameter
    Returns:
        obs_cut: List containing the sub-spectra defined by the parameter "wav_for_adapt".
                 [[wav_1, flx_1, err_1, reso_1], ..., [wav_n, flx_n, err_n, reso_n]]
        obs_pho: List containing the photometry ('P' replace the spectral resolution here).
                 [wav_phot, flx_phot, err_phot, 'P']
        obs_cut_ins: List containing different instruments used for the data (1 per wavelength).
                 [[instru_1], ..., [instru_n]]
        obs_pho_ins: List containing different filters used for the data (1 per photometric point).
                 [filter_phot_1, filter_phot_2, ..., filter_phot_n]

    Author: Simon Petrus
    """
    # Extraction
    with fits.open(global_params.observation_path) as hdul:
        wav = hdul[1].data['WAV']     # wavelength (um  -  vacuum)
        flx = hdul[1].data['FLX']     # flux (W.m-2.um.1)
        err = hdul[1].data['ERR']     # error (W.m-2.um.1)
        res = hdul[1].data['RES']     # spectral resolution
        ins = hdul[1].data['INS']     # instrument/filter name

        # Filter the NaN values
        nan_mod_ind = ~np.isnan(flx)
        wav = wav[nan_mod_ind]
        flx = flx[nan_mod_ind]
        err = err[nan_mod_ind]
        res = res[nan_mod_ind]
        ins = ins[nan_mod_ind]

    # Select the wavelength range(s) for the extraction
    if global_params.wav_for_adapt == '':
        wav_for_adapt_tab = [str(min(wav)) + ',' + str(max(wav))]
    else:
        wav_for_adapt_tab = global_params.wav_for_adapt.split('/')

    for range_ind, rangee in enumerate(wav_for_adapt_tab):
        rangee = rangee.split(',')
        ind = np.where((float(rangee[0]) <= wav) & (wav <= float(rangee[1])))

        # Photometry part of the data
        ind_photometry = np.where(res[ind] == 'P')
        obs_pho = [wav[ind][ind_photometry], flx[ind][ind_photometry], err[ind][ind_photometry],
                   res[ind][ind_photometry]]
        obs_pho_ins = ins[ind][ind_photometry]

        # Spectroscopy part of the data
        wav_spectro = np.delete(wav[ind], ind_photometry)
        flx_spectro = np.delete(flx[ind], ind_photometry)
        err_spectro = np.delete(err[ind], ind_photometry)
        res_spectro = np.delete(res[ind], ind_photometry)
        ins_spectro = np.delete(ins[ind], ind_photometry)

        if range_ind == 0:
            obs_cut = [[wav_spectro, flx_spectro, err_spectro, res_spectro]]
            obs_cut_ins = [[ins_spectro]]
        else:
            obs_cut.append([wav_spectro, flx_spectro, err_spectro, res_spectro])
            obs_cut_ins.append([ins_spectro])

    return obs_cut, obs_pho, obs_cut_ins, obs_pho_ins

# ----------------------------------------------------------------------------------------------------------------------


def adapt_model(global_params, wav_mod_nativ, flx_mod_nativ, res_mod_nativ):
    """
    Extracts a synthetic spectrum from a grid and decreases its spectral resolution. The photometry points are
    calculated too. Then each sub-spectrum are merged.

    Args:
        global_params: Class containing each parameter used in ForMoSA
        wav_mod_nativ: Wavelength grid of the model
        flx_mod_nativ: Flux of the model
        res_mod_nativ: Spectral resolution of the model as a function of the wavelength grid
    Returns:
        output_model: Flux of the spectrum with a decreased spectral resolution, re-sampled on the data wavelength grid

    Author: Simon Petrus
    """
    # Extract the synthetic spectra from the model grid
    mod_cut, mod_pho = extract_model(global_params, wav_mod_nativ, flx_mod_nativ, res_mod_nativ)

    # Estimate and subtraction of the continuum (if needed)
    if global_params.continuum_sub != 'NA':
        mod_cut_c, mod_pho_c = extract_model(global_params, wav_mod_nativ, flx_mod_nativ, res_mod_nativ, 'yes')
        for c, cut in enumerate(mod_cut_c):
            mod_cut[c] -= mod_cut_c[c]

    # Merging of each sub-spectrum
    for c, cut in enumerate(mod_cut):
        if c == 0:
            flx_mod_extract = mod_cut[c]
        else:
            flx_mod_extract = np.concatenate((flx_mod_extract, mod_cut[c]))

    return flx_mod_extract, mod_pho

# ----------------------------------------------------------------------------------------------------------------------


def extract_model(global_params, wav_mod_nativ, flx_mod_nativ, res_mod_nativ, cont='no'):
    """
    Extracts a synthetic spectrum from a grid and decreases its spectral resolution. The photometry points are
    calculated too.

    Args:
        global_params: Class containing each parameter used in ForMoSA
        wav_mod_nativ: Wavelength grid of the model
        flx_mod_nativ: Flux of the model
        res_mod_nativ: Spectral resolution of the model as a function of the wavelength grid
        cont: If the function is used to estimate the continuum cont='yes'
    Returns:
        mod_cut: List containing the sub-spectra defined by the parameter "wav_for_adapt".
                 [flx_1, ..., flx_n, err_n, reso_n]
        mod_pho: List containing the photometry ('P' replace the spectral resolution here).

    Author: Simon Petrus
    """
    # Take back the extracted data.
    spectrum_obs = np.load(global_params.result_path + '/spectrum_obs.npz', allow_pickle=True)
    obs_cut = spectrum_obs['obs_cut']
    obs_pho_ins = spectrum_obs['obs_pho_ins']

    # Reduce the spectral resolution for each sub-spectrum.
    mod_cut = []
    for c, cut in enumerate(obs_cut):
        # If we want to decrease the resolution of the data:
        if cont == 'no':
            if global_params.adapt_method == 'by_reso':
                mod_cut_flx = resolution_decreasing(global_params, cut, wav_mod_nativ, flx_mod_nativ, res_mod_nativ,
                                                    'mod')
            else:
                mod_cut_flx = spectres(cut[0], wav_mod_nativ, flx_mod_nativ)
        # If we want to estimate the continuum of the data:
        else:
            mod_cut_flx = continuum_estimate(global_params, cut[0], wav_mod_nativ, flx_mod_nativ, res_mod_nativ, 'mod')
        mod_cut.append(mod_cut_flx)

    # Calculate each photometry point.
    mod_pho = []
    for pho_ind, pho in enumerate(obs_pho_ins):
        filter_pho = np.load('phototeque/' + pho + '.npz')
        x_filt = filter_pho['x_filt']
        y_filt = filter_pho['y_filt']
        filter_interp = interp1d(x_filt, y_filt, fill_value="extrapolate")
        y_filt = filter_interp(wav_mod_nativ)

        ind = np.where(np.logical_and(wav_mod_nativ > min(x_filt), wav_mod_nativ < max(x_filt)))
        flx_filt = np.sum(flx_mod_nativ[ind] * y_filt[ind] * (wav_mod_nativ[ind][1] - wav_mod_nativ[ind][0]))
        y_filt_tot = np.sum(y_filt[ind] * (wav_mod_nativ[ind][1] - wav_mod_nativ[ind][0]))
        flx_filt = flx_filt / y_filt_tot
        mod_pho.append(flx_filt)

    return mod_cut, mod_pho

# ----------------------------------------------------------------------------------------------------------------------


def convolve_and_sample(wv_channels, sigmas_wvs, model_wvs, model_fluxes, num_sigma=3):
    """
    Simulate the observations of a model. Convolves the model with a variable Gaussian LSF, sampled at each desired
    spectral channel.

    Args:
        wv_channels: the wavelengths desired (length of N_output)
        sigmas_wvs: the LSF gaussian stddev of each wv_channels [IN UNITS OF model_wvs] (length of N_output)
        model_wvs: the wavelengths of the model (length of N_model)
        model_fluxes: the fluxes of the model (length of N_model)
        num_sigma (float): number of +/- sigmas to evaluate the LSF to.
    Returns:
        output_model: the fluxes in each of the wavelength channels (length of N_output)

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


def resolution_decreasing(global_params, cut, wav_mod_nativ, flx_mod_nativ, res_mod_nativ, obs_or_mod):
    """
    Decrease the resolution of a spectrum (data or model). The function calculates the FWHM as a function of the
    wavelengths for the data, the model, and for a custom spectral resolution (optional) and estimates the highest one
    for each wavelength (the lowest spectral resolution). It then calculates a sigma to decrease the resolution of the
    spectrum to this lowest FWHM for each wavelength and resample it on the wavelength grid of the data using the
    function 'convolve_and_sample'.

    Args:
        global_params: Class containing each parameter used in ForMoSA
        cut: Sub-spectra defined by the parameter "wav_for_adapt".
             [[wav_1, flx_1, err_1, reso_1], ..., [wav_n, flx_n, err_n, reso_n]]
        wav_mod_nativ: Wavelength grid of the model
        flx_mod_nativ: Flux of the model
        res_mod_nativ: Spectral resolution of the model as a function of the wavelength grid
        obs_or_mod: Parameter to identify if you want to manage a data or a model spectrum. 'obs' or 'mod'
    Returns:
        output_model: Flux of the spectrum with a decreased spectral resolution, re-sampled on the data wavelength grid

    Author: Simon Petrus
    """
    # Estimate of the FWHM of the data as a function of the wavelength
    fwhm_obs = 2 * cut[0] / cut[3]

    # Estimate of the FWHM of the model as a function of the wavelength
    ind_mod_obs = np.where((wav_mod_nativ <= cut[0][-1]) & (wav_mod_nativ > cut[0][0]))
    wav_mod_obs = wav_mod_nativ[ind_mod_obs]
    res_mod_obs = res_mod_nativ[ind_mod_obs]
    interp_mod_to_obs = interp1d(wav_mod_obs, res_mod_obs, fill_value='extrapolate')
    res_mod_obs = interp_mod_to_obs(cut[0])
    fwhm_mod = 2 * cut[0] / res_mod_obs

    # Estimate of the FWHM of the custom resolution (if defined) as a function of the wavelength
    if global_params.custom_reso != 'NA':
        fwhm_custom = 2 * cut[0] / float(global_params.custom_reso)
    else:
        fwhm_custom = cut[0] * np.nan

    # Estimate of the sigma for the convolution as a function of the wavelength and decrease the resolution
    max_fwhm = np.nanmax([fwhm_obs, fwhm_mod, fwhm_custom], axis=0)
    if obs_or_mod == 'obs':
        fwhm_conv = np.sqrt(max_fwhm ** 2 - fwhm_obs ** 2)
        sigma_conv = fwhm_conv / 2.355
        flx_obs_final = convolve_and_sample(cut[0], sigma_conv, cut[0], cut[1])
    else:
        fwhm_conv = np.sqrt(max_fwhm ** 2 - fwhm_mod ** 2)
        sigma_conv = fwhm_conv / 2.355
        flx_obs_final = convolve_and_sample(cut[0], sigma_conv, wav_mod_nativ, flx_mod_nativ)

    return flx_obs_final

# ----------------------------------------------------------------------------------------------------------------------


def continuum_estimate(global_params, wav_cut, wav, flx, res, obs_or_mod):
    """
    Decrease the resolution of a spectrum (data or model). The function calculates the FWHM as a function of the
    wavelengths of the custom spectral resolution (estimated for the continuum). It then calculates a sigma to decrease
    the resolution of the spectrum to this custom FWHM for each wavelength using a gaussian filter and resample it on
    the wavelength grid of the data.

    Args:
        global_params: Class containing each parameter used in ForMoSA
        wav_cut: Wavelength grid of the sub-spectrum data
        wav: Wavelength grid of the spectrum for which you want to estimate the continuum
        flx: Flux of the spectrum for which you want to estimate the continuum
        res: Spectral resolution of the spectrum for which you want to estimate the continuum
        obs_or_mod: Parameter to identify if you want to manage a data or a model spectrum. 'obs' or 'mod'
    Returns:
        output_model: Estimated continuum of the spectrum re-sampled on the data wavelength grid

    Author: Simon Petrus
    """
    # Adapt the model to the data grid wavelength
    if obs_or_mod == 'mod':
        interp_mod_to_obs = interp1d(wav, flx, kind='cubic', bounds_error=False)
        flx = interp_mod_to_obs(wav_cut)
        interp_mod_to_obs = interp1d(wav, res, kind='cubic', bounds_error=False)
        res = interp_mod_to_obs(wav_cut)
        wav = wav_cut

    # Redifined a spectrum only composed by the wavelength ranges used to estimate the continuum
    if global_params.wav_for_continuum != 'full':
        wav_for_continuum = global_params.wav_for_continuum.split('/')
        for wav_for_cont_cut_ind, wav_for_cont_cut in enumerate(wav_for_continuum):
            wav_for_cont_cut = wav_for_cont_cut.split(',')
            if wav_for_cont_cut_ind == 0:
                ind_cont_cut = np.where(wav <= float(wav_for_cont_cut[0]))
            elif wav_for_cont_cut_ind == len(wav_for_continuum)-1:
                ind_cont_cut = np.where(float(wav_for_cont_cut[0]) <= wav)
            else:
                ind_cont_cut = np.where((float(wav_for_cont_cut[0]) <= wav) &
                                        (wav <= float(wav_for_cont_cut[1])))
            if wav_for_cont_cut_ind == 0:
                wav_for_cont_final = wav[ind_cont_cut]
                flx_for_cont_final = flx[ind_cont_cut]
            else:
                wav_for_cont_final = np.concatenate((wav_for_cont_final, wav[ind_cont_cut]))
                flx_for_cont_final = np.concatenate((flx_for_cont_final, flx[ind_cont_cut]))

        model_interp = interp1d(wav_for_cont_final, flx_for_cont_final, kind='linear', bounds_error=False)
        flx = model_interp(wav)

    # To limit the computing time, the sigma for the convolution is not as a function of the wavelength but calculated
    # from the median wavelength. We just want an estimate of the continuum here.
    wav_median = np.median(wav)
    ind_wav_median_obs = find_nearest(wav, wav_median)
    dwav = wav[ind_wav_median_obs+1] - wav[ind_wav_median_obs]

    fwhm = 2 * wav_median / np.median(res)

    fwhm_continuum = 2 * wav_median / float(global_params.continuum_sub)

    fwhm_conv = np.sqrt(fwhm_continuum**2 - fwhm**2)
    sigma = fwhm_conv / (dwav * 2.355)
    continuum = gaussian_filter(flx, sigma)

    return continuum

# ----------------------------------------------------------------------------------------------------------------------
