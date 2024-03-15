import numpy as np
import xarray as xr
import extinction
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as const
from PyAstronomy.pyasl import dopplerShift, rotBroad
from adapt.extraction_functions import resolution_decreasing, convolve_and_sample
import scipy.ndimage as ndi
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import time 
# ----------------------------------------------------------------------------------------------------------------------


def lsq_fct(flx_obs_merge, err_obs_merge, star_flx_obs_merge, transm_obs_merge, new_flx_merge):
    """
    Estimation of the contribution of the planet and of the star to a spectrum (Used for HiRISE data)

    Args:
        flx_obs_merge: Flux of the data (spectroscopy)
        err_obs_merge: Error of the data (spectroscopy)
        star_flx_obs_merge: Flux of star observation data (spectroscopy)
        transm_obs_merge: Transmission (Atmospheric + Instrumental)
        new_flx_merge: Flux of interpolated synthetic spectrum (spectroscopy)
        wav_obs_merge : Wavelength grid of the data (spectroscopy)
        
    Returns:
        cp: Planetary contribution to the data (Spectroscopy)
        cs: Stellar contribution to the data (Spectroscopy)
        flx_obs_merge: New flux of the data (Spectroscopy)
        err_obs_merge: New error of the data (Spectroscopy)
    """
    
    new_flx_merge = new_flx_merge * transm_obs_merge
    
    # # # # # Removal of continuum with high-pass filtering
    #
    # Replacement of nan values by median values because filtering doesn't handle well nan values
    ind_nans_obs, ind_nans_star, ind_nans_mod = np.isnan(flx_obs_merge), np.isnan(star_flx_obs_merge), np.isnan(new_flx_merge)
    
    flx_obs_merge[ind_nans_obs] = np.nanmedian(flx_obs_merge)
    star_flx_obs_merge[ind_nans_star] = np.nanmedian(star_flx_obs_merge)
    new_flx_merge[ind_nans_mod] = np.nanmedian(new_flx_merge)
    
    # Low-pass filtering
    flx_obs_merge_continuum = ndi.gaussian_filter(flx_obs_merge, 300)
    star_flx_obs_merge_continuum = ndi.gaussian_filter(star_flx_obs_merge, 300)
    new_flx_merge_continuum = ndi.gaussian_filter(new_flx_merge, 300)
    
    # Replacement back of nan values
    flx_obs_merge[ind_nans_obs] = np.nan
    star_flx_obs_merge[ind_nans_star] = np.nan
    new_flx_merge[ind_nans_mod] = np.nan
    
    # Removal of low-pass filtered data
    flx_obs_merge = flx_obs_merge - flx_obs_merge_continuum + np.nanmedian(flx_obs_merge)
    star_flx_obs_merge = star_flx_obs_merge - star_flx_obs_merge_continuum + np.nanmedian(star_flx_obs_merge)
    new_flx_merge = new_flx_merge - new_flx_merge_continuum + np.nanmedian(new_flx_merge)
    #
    # # # # #
    
    # # # # # Least squares estimation
    #
    # Removal of nan values in preparation of the least squares
    nans = np.isnan(flx_obs_merge) | np.isnan(new_flx_merge) | np.isnan(err_obs_merge) | np.isnan(star_flx_obs_merge)
    
    final_flx_obs = flx_obs_merge[~nans]
    final_star_flx_obs = star_flx_obs_merge[~nans]
    final_obs_model = new_flx_merge[~nans]
    final_err_obs = err_obs_merge[~nans]
    
    # Construction of the matrix
    A_matrix = np.zeros([np.size(final_obs_model), 3])
    A_matrix[:,0] = final_obs_model * final_err_obs
    A_matrix[:,1] = final_star_flx_obs * final_err_obs
    A_matrix[:,2] = final_err_obs
    
    # Least square 
    res = optimize.lsq_linear(A_matrix, final_flx_obs * final_err_obs)
    
    # Results 
    cp = res.x[0] * new_flx_merge       # Estimated planetry contribution to the data
    cs = res.x[1] * star_flx_obs_merge  # Estimated stellar contribution to the data
    #
    # # # # #
    
    return cp, cs, flx_obs_merge


def calc_ck(flx_obs_merge, err_obs_merge, new_flx_merge, flx_obs_phot, err_obs_phot, new_flx_phot, r_picked, d_picked,
            analytic='no'):
    """
    Calculation of the dilution factor Ck and re-normalization of the interpolated synthetic spectrum (from the radius
    and distance or analytically).

    Args:
        flx_obs_merge   : Flux of the data (spectroscopy)
        err_obs_merge   : Error of the data (spectroscopy)
        new_flx_merge   : Flux of the interpolated synthetic spectrum (spectroscopy)
        flx_obs_phot    : Flux of the data (photometry)
        err_obs_phot    : Error of the data (photometry)
        new_flx_phot    : Flux of the interpolated synthetic spectrum (photometry)
        r_picked        : Radius randomly picked by the nested sampling (in RJup)
        d_picked        : Distance randomly picked by the nested sampling (in pc)
        analytic        : = 'yes' if Ck needs to be calculated analytically by the formula from Cushing et al. (2008)
    Returns:
        new_flx_merge   : Re-normalysed model spectrum
        new_flx_phot    : Re-normalysed model photometry
        ck              : Ck calculated

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
        wav_obs_merge       : Wavelength grid of the data
        flx_obs_merge       : Flux of the data
        err_obs_merge       : Error of the data
        new_flx_merge       : Flux of the interpolated synthetic spectrum
        rv_picked           : Radial velocity randomly picked by the nested sampling (in km.s-1)
    Returns:
        wav_obs_merge       : New wavelength grid of the data
        flx_obs_merge       : New flux of the data
        err_obs_merge       : New error of the data
        flx_post_doppler    : New flux of the interpolated synthetic spectrum

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
        wav_obs_merge   : Wavelength grid of the data (spectroscopy)
        wav_obs_phot    : Wavelength of the data (photometry)
        new_flx_merge   : Flux of the interpolated synthetic spectrum (spectroscopy)
        new_flx_phot    : Flux of the interpolated synthetic spectrum (photometry)
        av_picked       : Extinction randomly picked by the nested sampling (in mag)
    Returns:
        new_flx_merge   : New flux of the interpolated synthetic spectrum (spectroscopy)
        new_flx_phot    : New flux of the interpolated synthetic spectrum (photometry)

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
        wav_obs_merge   : Wavelength grid of the data
        new_flx_merge   : Flux of the interpolated synthetic spectrum
        ld_picked       : Limd darkening randomly picked by the nested sampling
        vsini_picked    : v.sin(i) randomly picked by the nested sampling (in km.s-1)
    Returns:
        new_flx_merge   : New flux of the interpolated synthetic spectrum

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

def bb_cpd_fct(wav_obs_merge, wav_obs_phot, new_flx_merge, new_flx_phot, distance, bb_T_picked, bb_R_picked):
    ''' Function to add the effect of a cpd (circum planetary disc) to the models
    Args:  
        wav_obs_merge   : Wavelength grid of the data (spectroscopy)
        wav_obs_phot    : Wavelength of the data (photometry)
        new_flx_merge   : Flux of the interpolated synthetic spectrum (spectroscopy)
        new_flx_phot    : Flux of the interpolated synthetic spectrum (photometry)
        bb_temp         : Temperature value randomly picked by the nested sampling in K units
        bb_rad          : Radius randomly picked by the nested sampling in units of planetary radius
    
    Returns:
        new_flx_merge   : New flux of the interpolated synthetic spectrum (spectroscopy)
        new_flx_phot    : New flux of the interpolated synthetic spectrum (photometry)

    Author: P. Palma-Bifani
    '''

    bb_T_picked *= u.K
    bb_R_picked *= u.Rjup
    distance *= u.pc

    def planck(wav, T):
        a = 2.0*const.h*const.c**2
        b = const.h*const.c/(wav*const.k_B*T)
        intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
        return intensity
    
    bb_intensity    = planck(wav_obs_merge*u.um, bb_T_picked)
    bb_intensity_f    = planck(wav_obs_phot*u.um, bb_T_picked)

    #flux_bb_lambda   = ( np.pi * (bb_R_picked)**2 / ( ck*u.km **2) * bb_intensity ).to(u.W/u.m**2/u.micron)
    flux_bb_lambda   = ( 4*np.pi*bb_R_picked**2/(distance**2) * bb_intensity ).to(u.W/u.m**2/u.micron)

    #flux_bb_lambda_f = ( np.pi * (bb_R_picked)**2 / ( ck*u.km **2) * bb_intensity_f ).to(u.W/u.m**2/u.micron)
    flux_bb_lambda_f = ( 4*np.pi*bb_R_picked**2/(distance**2) * bb_intensity_f ).to(u.W/u.m**2/u.micron)


    # add to model flux of the atmosphere
    new_flx_merge  += flux_bb_lambda.value
    new_flx_phot   += flux_bb_lambda_f.value
    # 
    return new_flx_merge, new_flx_phot


# ----------------------------------------------------------------------------------------------------------------------


def reso_fct(global_params, theta, theta_index, wav_obs_merge, new_flx_merge, reso_picked):
    """
    WORKING!
    Function to scale the spectral resolution of the synthetic spectra. This option is currently in test and make use
    of the functions defined in the 'adapt' section of ForMoSA, meaning that they will significantly decrease the speed of
    your inversion as the grid needs to be re-interpolated

    Args:
        global_params   : Class containing each parameter
        theta           : Parameter values randomly picked by the nested sampling
        theta_index     : Parameter index identificator
        wav_obs_merge   : Wavelength grid of the data
        new_flx_merge   : Flux of the interpolated synthetic spectrum
        reso_picked     : Spectral resolution randomly picked by the nested sampling
    Returns:
        None

    Author: Matthieu Ravet
    """

    # Import the grid and set it with the right parameters
    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine="netcdf4")
    wav_mod_nativ = ds["wavelength"].values
    grid = ds['grid']
    attr = ds.attrs
    grid_np = grid.to_numpy()
    model_to_adapt = grid_np[:, theta]

    # Modify the spectrum with the wanted spectral resolution
    flx_mod_extract, mod_pho = adapt_model(global_params, wav_mod_nativ, model_to_adapt, attr['res'], obs_name=obs_name,
                                        indobs=indobs)

    return 


# ----------------------------------------------------------------------------------------------------------------------


def modif_spec(global_params, theta, theta_index,
               wav_obs_merge, flx_obs_merge, err_obs_merge, new_flx_merge,
               wav_obs_phot, flx_obs_phot, err_obs_phot, new_flx_phot, transm_obs_merge = [], star_flx_obs_merge = [], indobs=0):
    """
    Modification of the interpolated synthetic spectra with the different extra-grid parameters:
        - Re-calibration on the data
        - Doppler shifting
        - Application of a substellar extinction
        - Application of a rotational velocity
        - Application of a circumplanetary disk (CPD)
    
    Args:
        global_params   : Class containing each parameter
        theta           : Parameter values randomly picked by the nested sampling
        theta_index     : Parameter index identificator
        wav_obs_merge   : Wavelength grid of the data (spectroscopy)
        flx_obs_merge   : Flux of the data (spectroscopy)
        err_obs_merge   : Error of the data (spectroscopy)
        new_flx_merge   : Flux of the interpolated synthetic spectrum (spectroscopy)
        wav_obs_phot    : Wavelength grid of the data (photometry)
        flx_obs_phot    : Flux of the data (photometry)
        err_obs_phot    : Error of the data (photometry)
        new_flx_phot    : Flux of the interpolated synthetic spectrum (photometry)
        transm_obs_merge: Transmission (Atmospheric + Instrumental)
        star_flx_obs_merge: Flux of star observation data (spectroscopy)
        indobs     (int): Index of the current observation looping (only relevant in MOSAIC, else set to 0)
    Returns:
        wav_obs_merge   : New wavelength grid of the data (may change with the Doppler shift)
        flx_obs_merge   : New flux of the data (may change with the Doppler shift)
        err_obs_merge   : New error of the data (may change with the Doppler shift)
        new_flx_merge   : New flux of the interpolated synthetic spectrum (spectroscopy)
        wav_obs_phot    : Wavelength grid of the data (photometry)
        flx_obs_phot    : Flux of the data (photometry)
        err_obs_phot    : Error of the data (photometry)
        new_flx_phot    : New flux of the interpolated synthetic spectrum (photometry)
    
    Author: Simon Petrus and Paulina Palma-Bifani
    """
    # Correction of the radial velocity of the interpolated synthetic spectrum.
    if len(flx_obs_merge) != 0:
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
    if len(flx_obs_merge) != 0:
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
            print('WARNING: You need to define a v.sin(i) AND a limb darkening, or set them both to NA')
            exit()
    
    # Adding a CPD
    if global_params.bb_T != "NA" and global_params.bb_R != "NA":
        # posteriors T_eff, R_disk
        # Enter 1 or 2 bb
        if global_params.bb_T[0] == 'constant':
            bb_T_picked = float(global_params.bb_T[1])
            bb_R_picked = float(global_params.bb_R[1])
        else:
            ind_theta_bb_T = np.where(theta_index == 'bb_T')
            ind_theta_bb_R = np.where(theta_index == 'bb_R')
            bb_T_picked = theta[ind_theta_bb_T[0][0]]
            bb_R_picked = theta[ind_theta_bb_R[0][0]]
        if global_params.d[0] == "constant":
            d_picked = float(global_params.d[1])
        else:
            ind_theta_d = np.where(theta_index == 'd')
            d_picked = theta[ind_theta_d[0][0]]

        new_flx_merge, new_flx_phot = bb_cpd_fct(wav_obs_merge, wav_obs_phot, new_flx_merge, new_flx_phot, d_picked, bb_T_picked, bb_R_picked)

    elif global_params.bb_T == "NA" and global_params.bb_R == "NA":
        pass

    else:
        print('WARNING: You need to define a blackbody radius and blackbody temperature, or set them to "NA"')
        exit()

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
    # If MOSAIC
    elif global_params.observation_format == 'MOSAIC':
        if global_params.r == "NA" and global_params.d == "NA" and global_params.use_lsqr[indobs] == 'False':
            new_flx_merge, new_flx_phot, ck = calc_ck(flx_obs_merge, err_obs_merge, new_flx_merge,
                                                    flx_obs_phot, err_obs_phot, new_flx_phot, 0, 0,
                                                    analytic='yes')

        elif global_params.r == "NA" and global_params.d == "NA" and global_params.use_lsqr[indobs] == 'True':   
            # global_params.use_lsqr = 'True', so no need to re-normalize the interpolated sythetic spectrum because the least squares automatically does it
                
            _, new_flx_phot, ck = calc_ck(flx_obs_merge, err_obs_merge, new_flx_merge, 
                            flx_obs_phot, err_obs_phot, new_flx_phot, 0, 0,
                            analytic='yes')
            
            cp, cs, flx_obs_merge = lsq_fct(flx_obs_merge, err_obs_merge, star_flx_obs_merge, transm_obs_merge, new_flx_merge)
            new_flx_merge = cp + cs

        else:   # either global_params.r or global_params.d is set to 'NA' 
            print('WARNING: You need to define a radius AND a distance, or set them both to "NA"')
            exit()
    # If Classical mode
    else:
        if global_params.r == "NA" and global_params.d == "NA" and global_params.use_lsqr == 'False':
            new_flx_merge, new_flx_phot, ck = calc_ck(flx_obs_merge, err_obs_merge, new_flx_merge,
                                                    flx_obs_phot, err_obs_phot, new_flx_phot, 0, 0,
                                                    analytic='yes')

        elif global_params.r == "NA" and global_params.d == "NA" and global_params.use_lsqr == 'True':   
            # global_params.use_lsqr = 'True', so no need to re-normalize the interpolated sythetic spectrum because the least squares automatically does it
                
            _, new_flx_phot, ck = calc_ck(flx_obs_merge, err_obs_merge, new_flx_merge, 
                            flx_obs_phot, err_obs_phot, new_flx_phot, 0, 0,
                            analytic='yes')
            
            cp, cs, flx_obs_merge = lsq_fct(flx_obs_merge, err_obs_merge, star_flx_obs_merge, transm_obs_merge, new_flx_merge)
            new_flx_merge = cp + cs
        
        else:   # either global_params.r or global_params.d is set to 'NA' 
            print('WARNING: You need to define a radius AND a distance, or set them both to "NA"')
            exit()


    return wav_obs_merge, flx_obs_merge, err_obs_merge, new_flx_merge, wav_obs_phot, flx_obs_phot, err_obs_phot, new_flx_phot, ck









