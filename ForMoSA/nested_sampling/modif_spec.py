import numpy as np
import extinction
from scipy.interpolate import interp1d
from PyAstronomy.pyasl import dopplerShift, rotBroad
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------


def calc_ck(flx_obs_merge, err_obs_merge, new_flx_merge, flx_obs_phot, err_obs_phot, new_flx_phot, r_picked, d_picked,
            analytic='no'):
    """
    Calculation of the dilution factor Ck and re-normalization of the interpolated synthetic spectrum (from the radius
    and distance or analytically).

    Args:
        flx_obs_merge: Flux of the data (spectroscopy)
        err_obs_merge: Error of the data (spectroscopy)
        new_flx_merge: Flux of the interpolated synthetic spectrum (spectroscopy)
        flx_obs_phot: Flux of the data (photometry)
        err_obs_phot: Error of the data (photometry)
        new_flx_phot: Flux of the interpolated synthetic spectrum (photometry)
        r_picked: Radius randomly picked by the nested sampling (in RJup)
        d_picked: Distance randomly picked by the nested sampling (in pc)
        analytic: = 'yes' if Ck needs to be calculated analytically by the formula from Cushing et al. (2008)
    Returns:
        new_flx_merge: Re-normalysed model spectrum
        new_flx_phot: Re-normalysed model photometry
        ck: Ck calculated

    Author: Simon Petrus
    """
    # Calculation of the dilution factor ck as a function of the radius and distance
    if analytic == 'no':
        r_picked *= 69911
        d_picked *= 3.086e+13
        ck = (r_picked/d_picked)**2
    # Calculation of the dilution factor ck analytically
    else:
        if len(flx_obs_merge) != 0:
            ck_top_merge = np.sum((new_flx_merge * flx_obs_merge) / (err_obs_merge * err_obs_merge))
            ck_bot_merge = np.sum((new_flx_merge / err_obs_merge)**2)
        else:
            ck_top_merge = 0
            ck_bot_merge = 0
        if len(flx_obs_phot) != 0:
            ck_top_phot = np.sum((new_flx_phot * flx_obs_phot) / (err_obs_phot * err_obs_phot))
            ck_bot_phot = np.sum((new_flx_phot / err_obs_phot)**2)
        else:
            ck_top_phot = 0
            ck_bot_phot = 0

        ck = (ck_top_merge + ck_top_phot) / (ck_bot_merge + ck_bot_phot)

    # Re-normalization of the interpolated synthetic spectra with ck
    if len(new_flx_merge) != 0:
        new_flx_merge *= ck
    if len(new_flx_phot) != 0:
        new_flx_phot *= ck

    return new_flx_merge, new_flx_phot, ck

# ----------------------------------------------------------------------------------------------------------------------


def doppler_fct(wav_obs_merge, flx_obs_merge, err_obs_merge, new_flx_merge, rv_picked):
    """
    Application of a Doppler shifting to the interpolated synthetic spectrum using the function pyasl.dopplerShift.
    Note: Observation can change due to side effects of the shifting.

    Args:
        wav_obs_merge: Wavelength grid of the data
        flx_obs_merge: Flux of the data
        err_obs_merge: Error of the data
        new_flx_merge: Flux of the interpolated synthetic spectrum
        rv_picked: Radial velocity randomly picked by the nested sampling (in km.s-1)
    Returns:
        wav_obs_merge: New wavelength grid of the data
        flx_obs_merge: New flux of the data
        err_obs_merge: New error of the data
        flx_post_doppler: New flux of the interpolated synthetic spectrum

    Author: Simon Petrus
    """
    # wav_doppler = wav_obs_merge*10000
    # flx_post_doppler, wav_post_doppler = dopplerShift(wav_doppler, new_flx_merge, rv_picked)
    new_wav = wav_obs_merge * ((rv_picked / 299792.458) + 1)

    rv_interp = interp1d(new_wav, new_flx_merge, fill_value="extrapolate")
    flx_post_doppler = rv_interp(wav_obs_merge)

    return wav_obs_merge, flx_obs_merge, err_obs_merge, flx_post_doppler

    # return Spectrum1d.from_array(new_wavelength, new_flux)
    # # Side effects
    # ind_nonan = np.argwhere(~np.isnan(flx_post_doppler))
    # wav_obs_merge = wav_obs_merge[ind_nonan[:, 0]]
    # flx_obs_merge = flx_obs_merge[ind_nonan[:, 0]]
    # err_obs_merge = err_obs_merge[ind_nonan[:, 0]]
    # flx_post_doppler = flx_post_doppler[ind_nonan[:, 0]]

    # return wav_obs_merge, flx_obs_merge, err_obs_merge, flx_post_doppler

# ----------------------------------------------------------------------------------------------------------------------


def reddening_fct(wav_obs_merge, wav_obs_phot, new_flx_merge, new_flx_phot, av_picked):
    """
    Application of a sythetic interstellar extinction to the interpolated synthetic spectrum using the function
    extinction.fm07.

    Args:
        wav_obs_merge: Wavelength grid of the data (spectroscopy)
        wav_obs_phot: Wavelength of the data (photometry)
        new_flx_merge: Flux of the interpolated synthetic spectrum (spectroscopy)
        new_flx_phot: Flux of the interpolated synthetic spectrum (photometry)
        av_picked: Extinction randomly picked by the nested sampling (in mag)
    Returns:
        new_flx_merge: New flux of the interpolated synthetic spectrum (spectroscopy)
        new_flx_phot: New flux of the interpolated synthetic spectrum (photometry)

    Author: Simon Petrus
    """
    if len(wav_obs_merge) != 0:
        dered_merge = extinction.fm07(wav_obs_merge * 10000, av_picked, unit='aa')
        new_flx_merge *= 10**(-0.4*dered_merge)
    if len(wav_obs_phot) != 0:
        dered_phot = extinction.fm07(wav_obs_phot * 10000, av_picked, unit='aa')
        new_flx_phot *= 10**(-0.4*dered_phot)

    return new_flx_merge, new_flx_phot

# ----------------------------------------------------------------------------------------------------------------------


def vsini_fct(wav_obs_merge, new_flx_merge, ld_picked, vsini_picked):
    """
    Application of a rotation velocity (line broadening) to the interpolated synthetic spectrum using the function
    extinction.fm07.

    Args:
        wav_obs_merge: Wavelength grid of the data
        new_flx_merge: Flux of the interpolated synthetic spectrum
        ld_picked: Limd darkening randomly picked by the nested sampling
        vsini_picked: v.sin(i) randomly picked by the nested sampling (in km.s-1)
    Returns:
        new_flx_merge: New flux of the interpolated synthetic spectrum

    Author: Simon Petrus
    """
    # Correct irregulatities in the wavelength grid
    wav_interval = wav_obs_merge[1:] - wav_obs_merge[:-1]
    wav_to_vsini = np.arange(min(wav_obs_merge), max(wav_obs_merge), min(wav_interval) * 2/3)
    vsini_interp = interp1d(wav_obs_merge, new_flx_merge, fill_value="extrapolate")
    flx_to_vsini = vsini_interp(wav_to_vsini)
    # Apply the v.sin(i)
    new_flx = rotBroad(wav_to_vsini, flx_to_vsini, ld_picked, vsini_picked)
    vsini_interp = interp1d(wav_to_vsini, new_flx, fill_value="extrapolate")
    new_flx_merge = vsini_interp(wav_obs_merge)

    return new_flx_merge

# ----------------------------------------------------------------------------------------------------------------------


def modif_spec(global_params, theta, theta_index,
               wav_obs_merge, flx_obs_merge, err_obs_merge, new_flx_merge,
               wav_obs_phot, flx_obs_phot, err_obs_phot, new_flx_phot):
    """
    Modification of the interpolated synthetic spectra with the different extra-grid parameters:
        - Re-calibration on the data
        - Doppler shifting
        - Application of a substellar extinction
        - Application of a rotational velocity

    Args:
        global_params: Class containing each parameter
        theta: Parameter values randomly picked by the nested sampling
        theta_index: Parameter index identificator
        wav_obs_merge: Wavelength grid of the data (spectroscopy)
        flx_obs_merge: Flux of the data (spectroscopy)
        err_obs_merge: Error of the data (spectroscopy)
        new_flx_merge: Flux of the interpolated synthetic spectrum (spectroscopy)
        wav_obs_phot: Wavelength grid of the data (photometry)
        flx_obs_phot: Flux of the data (photometry)
        err_obs_phot: Error of the data (photometry)
        new_flx_phot: Flux of the interpolated synthetic spectrum (photometry)
    Returns:
        wav_obs_merge: New wavelength grid of the data (may change with the Doppler shift)
        flx_obs_merge: New flux of the data (may change with the Doppler shift)
        err_obs_merge: New error of the data (may change with the Doppler shift)
        new_flx_merge: New flux of the interpolated synthetic spectrum (spectroscopy)
        wav_obs_phot: Wavelength grid of the data (photometry)
        flx_obs_phot: Flux of the data (photometry)
        err_obs_phot: Error of the data (photometry)
        new_flx_phot: New flux of the interpolated synthetic spectrum (photometry)

    Author: Simon Petrus
    """
    # Calculation of the dilution factor Ck and re-normalization of the interpolated synthetic spectrum.
    # From the radius and the distance.
    if global_params.r != "NA" and global_params.d != "NA":
        if global_params.r[0] == "constant":
            r_picked = float(global_params.r[1])
        else:
            ind_theta_r = np.where(theta_index == 'r')
            r_picked = theta[ind_theta_r[0][0]]
        if global_params.d[0] == "constant":
            d_picked = float(global_params.d[1])
        else:
            ind_theta_d = np.where(theta_index == 'd')
            d_picked = theta[ind_theta_d[0][0]]
        new_flx_merge, new_flx_phot, ck = calc_ck(flx_obs_merge, err_obs_merge, new_flx_merge,
                                                  flx_obs_phot, err_obs_phot, new_flx_phot, r_picked, d_picked)
    # Analytically
    elif global_params.r == "NA" and global_params.d == "NA":
        new_flx_merge, new_flx_phot, ck = calc_ck(flx_obs_merge, err_obs_merge, new_flx_merge,
                                                  flx_obs_phot, err_obs_phot, new_flx_phot, 0, 0,
                                                  analytic='yes')
    else:
        print('You need to define a radius AND a distance, or set them both to "NA"')
        exit()

    # Correction of the radial velocity of the interpolated synthetic spectrum.
    if global_params.rv != "NA":
        if global_params.rv[0] == 'constant':
            rv_picked = float(global_params.rv[1])
        else:
            ind_theta_rv = np.where(theta_index == 'rv')
            rv_picked = theta[ind_theta_rv[0][0]]
        wav_obs_merge, flx_obs_merge, err_obs_merge, new_flx_merge = doppler_fct(wav_obs_merge, flx_obs_merge,
                                                                                 err_obs_merge, new_flx_merge,
                                                                                 rv_picked)

    # Application of a synthetic interstellar extinction to the interpolated synthetic spectrum.
    if global_params.av != "NA":
        if global_params.av[0] == 'constant':
            av_picked = float(global_params.av[1])
        else:
            ind_theta_av = np.where(theta_index == 'av')
            av_picked = theta[ind_theta_av[0][0]]
        new_flx_merge, new_flx_phot = reddening_fct(wav_obs_merge, wav_obs_phot, new_flx_merge, new_flx_phot, av_picked)


    # Correction of the rotational velocity of the interpolated synthetic spectrum.
    if global_params.vsini != "NA" and global_params.ld != "NA":
        if global_params.vsini[0] == 'constant':
            vsini_picked = float(global_params.vsini[1])
        else:
            ind_theta_vsini = np.where(theta_index == 'vsini')
            vsini_picked = theta[ind_theta_vsini[0][0]]
        if global_params.ld[0] == 'constant':
            ld_picked = float(global_params.ld[1])
        else:
            ind_theta_ld = np.where(theta_index == 'ld')
            ld_picked = theta[ind_theta_ld[0][0]]

        new_flx_merge = vsini_fct(wav_obs_merge, new_flx_merge, ld_picked, vsini_picked)

    elif global_params.vsini == "NA" and global_params.ld == "NA":
        pass

    else:
        print('You need to define a v.sin(i) AND a limb darkening, or set them both to NA')
        exit()

    return wav_obs_merge, flx_obs_merge, err_obs_merge, new_flx_merge, wav_obs_phot, flx_obs_phot, err_obs_phot, new_flx_phot
