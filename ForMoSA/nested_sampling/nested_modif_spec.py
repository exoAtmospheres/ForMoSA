import numpy as np

from ForMoSA.nested_sampling.nested_highcont_models import hc_model
from ForMoSA.utils_spec import *


def modif_spec(global_params, theta, theta_index,
               obs_dict, 
               flx_mod_spectro, flx_mod_photo,
               wav_mod_spectro, res_mod_obs_spectro, 
               indobs=0):
    """
    Modification of the interpolated synthetic spectra with the different extra-grid parameters.
    It can perform : Re-calibration on the data, Doppler shifting, Application of a substellar extinction, Application of a rotational velocity,
    Application of a circumplanetary disk (CPD).

    Args:
        global_params               (object): Class containing each parameter
        theta                         (list): Parameter values randomly picked by the nested sampling
        theta_index                   (list): Parameter index identificator
        obs_dict                      (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)
        flx_mod_spectro              (array): New flux of the interpolated synthetic spectrum (spectroscopy)
        flx_mod_photo                (array): New flux of the interpolated synthetic spectrum (photometry)
        wav_mod_spectro              (array): Wavelength array of the model (can be different from wav_obs_spectro)
        res_mod_obs_spectro          (array): Spectroscopic resolution of the model interpolated onto wav_obs_spectro
        indobs                         (int): Index of the current observation looping
    Returns:
        - obs_dict                    (dict): New dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)
        - flx_mod_spectro            (array): New flux of the interpolated synthetic spectrum (spectroscopy)
        - flx_mod_photo              (array): New flux of the interpolated synthetic spectrum (photometry)
        - flx_mod_spectro_nativ      (array): New flux of the interpolated synthetic spectrum NOT RESAMPLED (spectroscopy)
        - contributions              (array): Contributions from the high-contrast model
        - scale_spectro              (float): Spectroscopic flux scaling factor
        - scale_photo                (float): Photometric flux scaling factor
 

    Author: Simon Petrus, Paulina Palma-Bifani, Allan Denis and Matthieu Ravet
    """
    
    # Correction of the radial velocity of the interpolated synthetic spectrum.
    if len(global_params.rv) > 3: # If you want separate rv for each observations
        if global_params.rv[indobs*3] != "NA":
            if global_params.rv[indobs*3] == "constant":
                rv_picked = float(global_params.rv[indobs*3+1])
            else:
                ind_theta_rv = np.where(theta_index == f'rv_{indobs}')
                rv_picked = theta[ind_theta_rv[0][0]]
            wav_mod_spectro, flx_mod_spectro = doppler_fct(wav_mod_spectro, flx_mod_spectro, rv_picked)
    else: # If you want 1 common rv for all observations
        if global_params.rv[0] != "NA":
            if global_params.rv[0] == "constant":
                rv_picked = float(global_params.rv[1])
            else:
                ind_theta_rv = np.where(theta_index == 'rv')
                rv_picked = theta[ind_theta_rv[0][0]]
            wav_mod_spectro, flx_mod_spectro = doppler_fct(wav_mod_spectro, flx_mod_spectro, rv_picked)



    # ----------------------------------------------------------------------------------------------------------------------
 


    # Correction of the rotational velocity of the interpolated synthetic spectrum.
    if len(global_params.vsini) > 4 and len(global_params.ld) > 3: # If you want separate vsini/ld for each observations
        if global_params.vsini[indobs*4] != "NA" and global_params.ld[indobs*3] != "NA":
            if global_params.vsini[indobs*4] == 'constant':
                vsini_picked = float(global_params.vsini[indobs*4+1])
            else:
                ind_theta_vsini = np.where(theta_index == f'vsini_{indobs}')
                vsini_picked = theta[ind_theta_vsini[0][0]]

            if global_params.ld[indobs*3] == 'constant':
                ld_picked = float(global_params.ld[indobs*3+1])
            else:
                ind_theta_ld = np.where(theta_index == f'ld_{indobs}')
                ld_picked = theta[ind_theta_ld[0][0]]

            vsini_type = global_params.vsini[indobs*4 + 3]
            flx_mod_spectro, res_mod_obs_spectro = vsini_fct(wav_mod_spectro, flx_mod_spectro, res_mod_obs_spectro, ld_picked, vsini_picked, vsini_type)

        elif global_params.vsini[indobs*4] == "NA" and global_params.ld[indobs*3] == "NA":
            pass
        else:
            print(f'WARNING: You need to define a v.sin(i) AND a limb darkening, or set them both to "NA" for observation {indobs}')
            exit()


    else:# If you want 1 common vsini/ld for all observations
        if global_params.vsini[0] != "NA" and global_params.ld[0] != "NA":
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

            vsini_type = global_params.vsini[3]
            flx_mod_spectro, res_mod_obs_spectro = vsini_fct(wav_mod_spectro, flx_mod_spectro, res_mod_obs_spectro, ld_picked, vsini_picked, vsini_type)

        elif global_params.vsini[0] == "NA" and global_params.ld[0] == "NA":
            pass
        else:
            print('WARNING: You need to define a v.sin(i) AND a limb darkening, or set them both to "NA"')
            exit()



    # ----------------------------------------------------------------------------------------------------------------------       



    # Application of a synthetic interstellar extinction to the interpolated synthetic spectrum.
    if global_params.av[0] != "NA":
        if global_params.av[0] == 'constant':
            av_picked = float(global_params.av[1])
        else:
            ind_theta_av = np.where(theta_index == 'av')
            av_picked = theta[ind_theta_av[0][0]]
        flx_mod_spectro, flx_mod_photo = reddening_fct(wav_mod_spectro, obs_dict['wav_photo'], flx_mod_spectro, flx_mod_photo, av_picked)



    # ----------------------------------------------------------------------------------------------------------------------         


            
    # Adding a CPD
    if global_params.bb_t[0] != "NA" and global_params.bb_r[0] != "NA":
        if global_params.d[0] != "NA":
            if global_params.bb_t[0] == 'constant':
                bb_t_picked = float(global_params.bb_t[1])
            else:
                ind_theta_bb_t = np.where(theta_index == 'bb_t')
                bb_t_picked = theta[ind_theta_bb_t[0][0]]
            if global_params.bb_r[0] == 'constant':
                bb_r_picked = float(global_params.bb_r[1])
            else:
                ind_theta_bb_r = np.where(theta_index == 'bb_r')
                bb_r_picked = theta[ind_theta_bb_r[0][0]]
            if global_params.d[0] == "constant":
                d_picked = float(global_params.d[1])
            else:
                ind_theta_d = np.where(theta_index == 'd')
                d_picked = theta[ind_theta_d[0][0]]

            flx_mod_spectro, flx_mod_photo = bb_cpd_fct(wav_mod_spectro, obs_dict['wav_photo'], flx_mod_spectro, flx_mod_photo, d_picked, bb_t_picked, bb_r_picked)
        else:
            print('WARNING: You need to define a distance if you want to fit for the blackbody')
            exit()            

    elif global_params.bb_t[0] == "NA" and global_params.bb_r[0] == "NA":
        pass

    else:
        print('WARNING: You need to define a blackbody radius, temperature and a distance, or set them all "NA"')
        exit()

        


    # ----------------------------------------------------------------------------------------------------------------------

    
    # Decrease the resolution the model (if necessary). Save the non-resampled model beforehand
    flx_mod_spectro_nativ = np.copy(flx_mod_spectro) 
    # If you don't shift the spectrum and don't have it at higher resolution, you do not need to convolve and resample it
    if len(wav_mod_spectro) == len(obs_dict['wav_spectro']) and global_params.rv[indobs*3 % len(global_params.rv)] == "NA":
        pass
    else:
        flx_mod_spectro = resolution_decreasing(wav_mod_spectro, 
                                                flx_mod_spectro, 
                                                res_mod_obs_spectro, 
                                                obs_dict['wav_spectro'], 
                                                obs_dict['res_spectro'])

    # ----------------------------------------------------------------------------------------------------------------------



    # High contrast model
    if global_params.hc_type[indobs % len(global_params.hc_type)] != "NA":
        # Least Squares inversion
        _, flx_mod_spectro = hc_model(global_params, obs_dict, flx_mod_spectro)    
    else:
        pass



    # ----------------------------------------------------------------------------------------------------------------------



    # Calculation of the flux scaling factor

    if global_params.hc_type[indobs % len(global_params.hc_type)] == "NA": # hc already rescale everything

        # If you need to use the covariance matrix in you estimation of your scaling factor
        if global_params.logL_type[indobs % len(global_params.logL_type)] == 'chi2_covariance' and len(obs_dict['inv_cov']) != 0:
            use_cov = True
        else:
            use_cov = False

        # From the radius and the distance.
        if global_params.r[0] != "NA" and global_params.d[0] != "NA":
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

            # With the extra alpha scaling
            if len(global_params.alpha) > 3: # If you want separate alpha for each observations
                if global_params.alpha[indobs*3] != "NA":
                    if global_params.alpha[indobs*3] == "constant":
                        alpha_picked = float(global_params.alpha[indobs*3+1])
                    else:
                        ind_theta_alpha = np.where(theta_index == f'alpha_{indobs}')
                        alpha_picked = theta[ind_theta_alpha[0][0]]
                    flx_mod_spectro, flx_mod_photo, scale_spectro, scale_photo = calc_flx_scale(obs_dict, flx_mod_spectro, flx_mod_photo, r_picked, d_picked, alpha=alpha_picked, mode='physical', use_cov=use_cov)
                # - - - - - - 
                # SPECIAL CASE FOR MOSAIC WHEN YOU DONT FIT R AND D FOR ONE OF THE OBS BUT STILL WANTS TO FIT IT FOR THE OTHERS !!
                # - - - - - - 
                else:
                    flx_mod_spectro, flx_mod_photo, scale_spectro, scale_photo = calc_flx_scale(obs_dict, flx_mod_spectro, flx_mod_photo, 0, 0, alpha=0, mode='analytic', use_cov=use_cov)
            else: # If you want 1 common alpha for all observations
                if global_params.alpha[0] != "NA":
                    if global_params.alpha[0] == "constant":
                        alpha_picked = float(global_params.alpha[1])
                    else:
                        ind_theta_alpha = np.where(theta_index == 'alpha')
                        alpha_picked = theta[ind_theta_alpha[0][0]]
                    flx_mod_spectro, flx_mod_photo, scale_spectro, scale_photo = calc_flx_scale(obs_dict, flx_mod_spectro, flx_mod_photo, r_picked, d_picked, alpha=alpha_picked, mode='physical', use_cov=use_cov)
                else: # Without the extra alpha scaling
                    flx_mod_spectro, flx_mod_photo, scale_spectro, scale_photo = calc_flx_scale(obs_dict, flx_mod_spectro, flx_mod_photo, r_picked, d_picked, mode='physical', use_cov=use_cov)

        # Analytically
        elif global_params.r[0] == "NA" and global_params.d[0] == "NA":
            # If we compute ck analytically, the resolution decreasing is already included in the function
            flx_mod_spectro, flx_mod_photo, scale_spectro, scale_photo = calc_flx_scale(obs_dict, flx_mod_spectro, flx_mod_photo, 0, 0, alpha=0, mode='analytic', use_cov=use_cov)


        else:   # either global_params.r or global_params.d is set to 'NA'
            print('WARNING: You need to define a radius AND a distance, or set them both to "NA"')
            exit()
            
    else:     
        scale_spectro, scale_photo = 1, 1

    # ----------------------------------------------------------------------------------------------------------------------


    # Outputs
    return obs_dict, flx_mod_spectro, flx_mod_photo, flx_mod_spectro_nativ, scale_spectro, scale_photo









