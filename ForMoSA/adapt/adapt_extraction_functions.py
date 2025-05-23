from __future__ import division
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

from ForMoSA.utils_spec import resolution_decreasing, continuum_estimate


def adapt_observation(global_params, wav_mod_nativ, res_mod_nativ, obs_name, indobs=0):
    """
    Take back the extracted data spectrum from the function 'extract_observation', decrease its spectral
    resolution and remove the continuum if necessary

    Args:
        global_params     (object): Class containing each parameter
        wav_mod_nativ      (array): Wavelength grid of the model
        res_mod_nativ      (array): Spectral resolution of the model
        obs_name             (str): Name of the current observation looping
        indobs               (int): Index of the current observation looping

    Returns:
        - obs_dict          (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)

    Author: Simon Petrus, Matthieu Ravet
    """

    # Extract the wavelengths, flux, errors, spectral resolution, and instrument/filter names from the observation file.
    obs_dict = extract_observation(global_params, indobs=indobs)

    # Decrease the resolution and remove the continuum if necessary
    if len(obs_dict['wav_spectro']) != 0:

        # - - - - - -

        # Setup target resolution for the observation
        # Interpolate the resolution of the model onto the wavelength of the data to properly decrease the resolution if necessary
        interp_mod_to_obs = interp1d(wav_mod_nativ, res_mod_nativ, fill_value='extrapolate')
        res_mod_obs = interp_mod_to_obs(obs_dict['wav_spectro'])

        if global_params.target_res_obs[indobs % len(global_params.target_res_obs)] == 'obs': # Keeping the resolution of the observation except where its higher than the model's
            target_res_obs = np.min([obs_dict['res_spectro'], res_mod_obs], axis=0)
        else:                                             # Using a custom resolution except where its higher than the model's or the observation's
            res_custom = np.full(len(obs_dict['res_spectro']), float(global_params.target_res_obs[indobs % len(global_params.target_res_obs)]))
            target_res_obs = np.min([obs_dict['res_spectro'], res_mod_obs, res_custom], axis=0)

        # - - - - - -

        # If we want to decrease the resolution of the spectroscopic data:
        obs_dict['flx_spectro'] = resolution_decreasing(obs_dict['wav_spectro'],
                                                obs_dict['flx_spectro'],
                                                obs_dict['res_spectro'],
                                                obs_dict['wav_spectro'],
                                                target_res_obs)
        
        obs_dict['transm'] = resolution_decreasing(obs_dict['wav_spectro'],
                                                obs_dict['transm'],
                                                obs_dict['res_spectro'],
                                                obs_dict['wav_spectro'],
                                                target_res_obs)
        
        obs_dict['star_flx'] = np.asarray([resolution_decreasing(obs_dict['wav_spectro'],
                                                obs_dict['star_flx'][:,i],
                                                obs_dict['res_spectro'],
                                                obs_dict['wav_spectro'],
                                                target_res_obs)
                                                for i in range(obs_dict['star_flx'].shape[-1])]).T
    
        obs_dict['system'] = np.asarray([resolution_decreasing(obs_dict['wav_spectro'],
                                                obs_dict['system'][:,i],
                                                obs_dict['res_spectro'],
                                                obs_dict['wav_spectro'],
                                                target_res_obs)
                                                for i in range(obs_dict['system'].shape[-1])]).T
    
        
        # Since the resolution of the observation might have change, we need to save the new one
        obs_dict['res_spectro'] = target_res_obs

        # Compute the inverse covariance and log-determinant (if necessary)
        if len(obs_dict['cov']) != 0:
            obs_dict['inv_cov'] = np.linalg.inv(obs_dict['cov']) # Save the inverse covariance to speed up the inversion
        else:
            pass
        
        # If we want to estimate and substract the continuum of the data:
        if global_params.res_cont[indobs % len(global_params.res_cont)] != 'NA':
            print()
            print(obs_name + ' will have a R=' + global_params.res_cont[indobs % len(global_params.res_cont)] + ' continuum removed using a ' 
                  + global_params.wav_cont[indobs % len(global_params.wav_cont)] + ' wavelength range')
            print()

            obs_dict['flx_spectro_cont'] = continuum_estimate(obs_dict['wav_spectro'],
                                                              obs_dict['flx_spectro'],
                                                              obs_dict['res_spectro'],
                                                              global_params.wav_cont[indobs % len(global_params.wav_cont)],
                                                              float(global_params.res_cont[indobs % len(global_params.res_cont)]))
            
            # If you don't use hc models, the data continuum is directly removed
            if global_params.hc_type[indobs % len(global_params.hc_type)] == 'NA':
                obs_dict['flx_spectro'] -= obs_dict['flx_spectro_cont']

            else: # If you use hc models, the data is kept; we just need to estimate the continuum of the star flux as well
                obs_dict['star_flx_cont'] = continuum_estimate(obs_dict['wav_spectro'],
                                                            obs_dict['star_flx'][:,len(obs_dict['star_flx'][0]) // 2], # Continuum of the star on the central pixel
                                                            obs_dict['res_spectro'],
                                                            global_params.wav_cont[indobs % len(global_params.wav_cont)],
                                                            float(global_params.res_cont[indobs % len(global_params.res_cont)]))
            
            
                
    return obs_dict

# ----------------------------------------------------------------------------------------------------------------------


def extract_observation(global_params, indobs=0):
    """
    Extract the information from the observation file, including the wavelengths (um - vacuum), flux (W.m-2.um.1), errors (W.m-2.um.1), covariance (W.m-2.um.1)**2, spectral resolution, 
    instrument/filter name, transmission (Atmo+inst) and star flux (W.m-2.um.1). The wavelength range is define by the parameter "wav_for_adapt".

    Args:
        global_params  (object): Class containing each parameter
        obs_name          (str): Name of the current observation looping
        indobs            (int): Index of the current observation looping

    Returns:
        - obs_dict       (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)

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
        if global_params.logL_type[indobs % len(global_params.logL_type)] != 'chi2_covariance' and global_params.logL_type[indobs % len(global_params.logL_type)] != 'chi2_noisescaling_covariance':
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
            cov = cov[np.ix_(nan_mod_ind, nan_mod_ind)]
        if len(transm) != 0 and len(star_flx) != 0:
            transm = transm[nan_mod_ind]
        if len(star_flx) != 0:
            star_flx = np.delete(star_flx, np.where(~nan_mod_ind), axis=0)
        if len(system) != 0:
            system = np.delete(system, np.where(~nan_mod_ind), axis=0)


        # - - - - - - - - - 

        # Separate photometry from spectroscopy
        mask_photo = (res == 0.0)
        wav_photo, wav_spectro = wav[mask_photo], wav[~mask_photo]
        flx_photo, flx_spectro = flx[mask_photo], flx[~mask_photo]
        ins_photo, res_spectro = ins[mask_photo], res[~mask_photo]
        err_phot, err_spectro = err[mask_photo], err[~mask_photo]
        if len(cov) != 0:
            cov = cov[np.ix_(~mask_photo, ~mask_photo)]
        if len(transm) != 0 and len(star_flx) != 0:
            transm = transm[~mask_photo]
        if len(star_flx) != 0:
            star_flx = np.delete(star_flx, np.where(mask_photo), axis=0)
        if len(system) != 0:
            system = np.delete(system, np.where(mask_photo), axis=0)

        # - - - - - - - - - 

        # Observation dictionary
        obs_dict = {'wav_photo': wav_photo, # Photometry part
                    'flx_photo': flx_photo,
                    'err_photo': err_phot,
                    'ins_photo': ins_photo,
                    'wav_spectro': wav_spectro, # Spectroscopy part
                    'flx_spectro': flx_spectro,
                    'err_spectro': err_spectro,
                    'res_spectro': res_spectro,
                    'cov': cov,
                    'transm': transm,
                    'star_flx': star_flx,
                    'system': system
                    }
        
        return obs_dict   


# ----------------------------------------------------------------------------------------------------------------------


def adapt_model(global_params, obs_dict, wav_mod_nativ, flx_mod_nativ, res_mod_nativ, target_wav_mod, target_res_mod, indobs=0):
    """
    Extracts a synthetic spectrum from a grid and decreases its spectral resolution. The photometry points are
    calculated too.

    Args:
        global_params          (object): Class containing each parameter used in ForMoSA
        obs_dict                 (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)
        wav_mod_nativ           (array): Native wavelength grid of the model
        flx_mod_nativ           (array): Flux of the model
        res_mod_nativ           (array): Spectral resolution of the model
        target_wav_mod          (array): Targeted wavelength grid of the final grid
        target_res_mod          (array): Targeted spectral resolution of the final grid
        indobs                    (int): Index of the current observation looping
    Returns:
        - mod_spectro   (array): List containing the sub-spectra defined by the parameter "wav_for_adapt".
        - mod_photo     (array): List containing the photometry ('0' replace the spectral resolution here).

    Author: Simon Petrus, Matthieu Ravet
    """
    # Spectroscopy part
    mod_spectro = np.empty(len(target_wav_mod), dtype=float)

    # If we want to decrease the resolution of the model
    mod_spectro = resolution_decreasing(wav_mod_nativ,
                                        flx_mod_nativ,
                                        res_mod_nativ,
                                        target_wav_mod,
                                        target_res_mod)
    
    # If we want to estimate and substract the continuum of the data (except for high contrast where we need to keeo the og spectrum):
    if global_params.res_cont[indobs % len(global_params.res_cont)] != 'NA' and global_params.hc_type[indobs % len(global_params.hc_type)] == 'NA':
        mod_spectro -= continuum_estimate(target_wav_mod,
                                          mod_spectro,
                                          target_res_mod,
                                          global_params.wav_cont[indobs % len(global_params.wav_cont)],
                                          float(global_params.res_cont[indobs % len(global_params.res_cont)]))

    # Photometry part
    mod_photo = np.empty(len(obs_dict['wav_photo']), dtype=float)
        
    if len(obs_dict['wav_photo']) != 0:
        # Calculate each photometry point (if necessary)
        for pho_ind, pho in enumerate(obs_dict['ins_photo']):
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