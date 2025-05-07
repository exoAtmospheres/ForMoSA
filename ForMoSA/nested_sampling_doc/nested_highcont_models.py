import numpy as np
import scipy.optimize as optimize

from ForMoSA.utils_spec import continuum_estimate


def hc_model(global_params, obs_dict, flx_mod_spectro, indobs):
    '''
    For high-contrast companions, where the star speckles signal contaminate the data

    Args:
        global_params      (object): Class containing each parameter used in ForMoSA
        obs_dict             (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)              
        flx_mod_spectro     (array): Model flux of the companion
        indobs                (int): Index of the current observation loop
    
    Returns:
        - results           (array): Results of the high-constrast model
        - flx_mod_spectro   (array): Model of the high-constrast model

    Authors: Allan Denis
    '''

    # Create star master spectrum
    star_flx_master = obs_dict['star_flx'][:, len(obs_dict['star_flx'][0]) // 2]

    if global_params.hc_type[indobs % len(global_params.hc_type)] == 'nofit_rm_spec':   # The user does not want to fit for the contribution of the planet (used for CCF-to-loglikelihood mapping functions)
        flx_cont_mod = continuum_estimate(obs_dict['wav_spectro'], 
                                          flx_mod_spectro, 
                                          obs_dict['res_obs'], 
                                          global_params.wav_cont[indobs % len(global_params.wav_cont)], 
                                          float(global_params.res_cont[indobs % len(global_params.res_cont)]))
        flx_mod_spectro -= flx_cont_mod
        flx_mod_spectro *= obs_dict['transm']
        obs_dict['flx_spectro'] -= star_flx_master / obs_dict['star_flx_cont'] * obs_dict['flx_spectro_cont'] 
        obs_dict['flx_spectro'] /= np.sqrt(np.sum(obs_dict['flx_spectro']**2))
        flx_mod_spectro /= np.sqrt(np.sum(flx_mod_spectro**2))
        obs_dict['star_flx'], results = np.asarray([]), np.asarray([])
        
    else:   
        flx_mod_spectro *= obs_dict['transm']
        flx_cont_mod = continuum_estimate(obs_dict['wav_spectro'], 
                                          flx_mod_spectro, 
                                          obs_dict['res_obs'], 
                                          global_params.wav_cont[indobs % len(global_params.wav_cont)], 
                                          float(global_params.res_cont[indobs % len(global_params.res_cont)]))

        if global_params.bounds_lsq[indobs % len(global_params.bounds_lsq)] != 'NA':
            bounds = (float(global_params.bounds_lsq[indobs % len(global_params.bounds_lsq)][1:-1].split(',')[0]), 
                      float(global_params.bounds_lsq[indobs % len(global_params.bounds_lsq)][1:-1].split(',')[1]))
        
        weights = (1 / obs_dict['err_spectro'])**2  # For now we consider diagonal covariance matrices only
    
        # Select model
        if global_params.hc_type[indobs % len(global_params.hc_type)] == 'nonlinear_fit_spec':
            results, flx_mod_spectro = hc_model_nonlinear_estimate_speckles(obs_dict , flx_mod_spectro, flx_cont_mod, weights, bounds)

        elif global_params.hc_type[indobs % len(global_params.hc_type)] == 'fit_spec':
            results, flx_mod_spectro = hc_model_estimate_speckles(obs_dict, flx_mod_spectro, flx_cont_mod, star_flx_master, weights, bounds)

        elif global_params.hc_type[indobs % len(global_params.hc_type)] == 'rm_spec':
            results, flx_mod_spectro = hc_model_remove_speckles(obs_dict, flx_mod_spectro, flx_cont_mod, star_flx_master, weights, bounds)

        elif global_params.hc_type[indobs % len(global_params.hc_type)] == 'fit_spec_rm_cont':
            results, flx_mod_spectro = hc_model_estimate_speckles_remove_continuum(obs_dict, flx_mod_spectro, flx_cont_mod, weights, bounds)

        elif global_params.hc_type[indobs % len(global_params.hc_type)] == 'fit_spec_fit_cont':
            raise Exception(
                'Continuum fitting-based high-constrast model no implement yet ! Please use another function')
            
    return results, flx_mod_spectro



# ----------------------------------------------------------------------------------------------------------------------



def hc_model_nonlinear_estimate_speckles(obs_dict, flx_mod_spectro, flx_cont_mod, weights, bounds):
    '''
    Non linear high-constrast model of planet and star contributions (see Landman et al. 2023)
    This model is in principle more general than hc_model_estimate_speckles 
    because in the latter case we make the assumption that the star speckels dominate the data which is not the case here

    Args:
        obs_dict             (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)
        flx_mod_spectro     (array): Model of the companion
        flx_cont_mod        (array): Continuum of the model of the companion
        weights             (array): Weights to apply to the data
        bounds              (tuple): Bounds to be applied to the estimated parameters

    Returns:
        - results           (array): Results of the high-constrast model
        - flx_mod_spectro   (array): Model of the high-constrast model

    Authors: Allan Denis
    '''
    ind_star = 1 + len(obs_dict['star_flx'][0])
    # # # # # # # Solve non linear Least Squares full_model(theta) = flx_obs

    # Definition of f
    def f(theta):
        star_speckles = np.dot(theta[1:ind_star], obs_dict['star_flx'].T / obs_dict['star_flx_cont'] * (obs_dict['flx_spectro_cont']  - theta[0] * flx_cont_mod))
        results = theta[0] * flx_mod_spectro + star_speckles
        if len(theta) > ind_star:
            results += np.dot(theta[ind_star:], obs_dict['system'].T) 
        return weights * (results - obs_dict['flx_spectro'])
              

    # Solve non linear Least Squares
    # Initial guess for the planetary contribution
    theta0 = [0]
    for i in range(len(obs_dict['star_flx'][0])):
        # Arbitrary initial guesses for star speckles contribution
        theta0.append(((i+1) / len(obs_dict['star_flx'][0]))**2)
        
    if len(obs_dict['system']) > 0:
        for i in range(len(obs_dict['system'][0])):
            # Arbitrary initial guesses for systematics contribution
            theta0.append(1)
    # Solve non linear Least
    results = optimize.least_squares(f, theta0, bounds=bounds)
    
    
    # Full model
    flx_mod_spectro_full = f(results.x) / weights + obs_dict['flx_spectro']
    obs_dict['star_flx'] = np.dot(results.x[1:ind_star], obs_dict['star_flx'].T / obs_dict['star_flx_cont'] * (obs_dict['flx_spectro_cont']  - results.x[0] * flx_cont_mod))
    obs_dict['system'] = np.dot(results.x[ind_star:], obs_dict['system'].T)

    return results, flx_mod_spectro_full



# ----------------------------------------------------------------------------------------------------------------------



def hc_model_estimate_speckles(obs_dict, flx_mod_spectro, flx_cont_mod, star_flx_master, weights, bounds):
    '''
    Linear high-constrast model of planet and star contributions under the assumtion that the star speckles dominate the data  (see Landman et al. 2023)

    Args:
        obs_dict                     (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)
        flx_mod_spectro             (array): Model of the companion
        flx_cont_mod                (array): Continuum of the model of the companion
        star_flx_master             (array): Master star data
        weights                     (array): Weights to apply to the data
        bounds                      (tuple): Bounds to be applied to the estimated parameters

    Returns:
        - results                   (array): Results of the high-constrast model
        - flx_mod_spectro           (array): Model of the high-constrast model

    Authors: Allan Denis
    '''

    ind_star = 1 + len(obs_dict['star_flx'][0])
    if len(obs_dict['system']) > 0:
        ind_system = ind_star + len(obs_dict['system'][0])
    else:
        ind_system = ind_star

    # # # # # # Solve linear Least Squares A.x = b

    # Build matrix A
    A = np.zeros([np.size(obs_dict['flx_spectro']), ind_system])
    A[:, 0] = weights * (flx_mod_spectro - flx_cont_mod *
                             star_flx_master / obs_dict['star_flx_cont'])

    for star_i in range(len(obs_dict['star_flx'][0])):
        A[:, star_i + 1] = weights * (obs_dict['star_flx'][:, star_i] / obs_dict['star_flx_cont'] * obs_dict['flx_spectro_cont'] )
            
    for system_i in range(ind_system - ind_star):
        A[:, system_i + ind_star] = weights * obs_dict['system'][:, system_i]

    # Build vector b
    b = weights * obs_dict['flx_spectro']
    # Solve linear Least Squares
    results = optimize.lsq_linear(A, b, bounds=bounds)

    # Full model
    flx_mod_spectro = np.dot(A, results.x) / weights
    obs_dict['star_flx'] = np.dot(A[:, 1:ind_star], results.x[1:ind_star]) / weights
    obs_dict['system'] = np.dot(A[:, ind_star:], results.x[ind_star:])


    return results.x, flx_mod_spectro



# ----------------------------------------------------------------------------------------------------------------------



def hc_model_remove_speckles(obs_dict, flx_mod_spectro, flx_cont_mod, star_flx_master, weights, bounds):
    '''
    Linear high-constrast model of planet contribution only where the speckles are filtered out from the data (see Landman et al. 2023)

    Args:
        obs_dict                     (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)
        flx_mod_spectro             (array): Model of the companion
        flx_cont_mod                (array): Continuum of the model of the companion
        star_flx_master             (array): Master star data
        weights                     (array): Weights to apply to the data
        bounds                      (tuple): Bounds to be applied to the estimated parameters

    Returns:
        - results                   (array): Results of the high-constrast model
        - flx_mod_spectro           (array): Model of the high-constrast model

    Authors: Allan Denis
    '''

    if len(obs_dict['system']) > 0:
        ind_system = 1 + len(obs_dict['system'][0])
    else:
        ind_system = 1

    # # # # # # # Solve linear Least Squared A.x = b
    A = np.zeros([np.size(obs_dict['flx_spectro']), ind_system])

    # Build matrix A
    A[:, 0] = weights * (flx_mod_spectro - flx_cont_mod *
                             star_flx_master / obs_dict['star_flx_cont'])
    
    for system_i in range(ind_system-1):
        A[:, system_i + 1] = weights * obs_dict['system'][:, system_i]

    # Build vector b
    b = weights * (obs_dict['flx_spectro'] - star_flx_master /
                       obs_dict['star_flx_cont'] * obs_dict['flx_spectro_cont'] )

    # Solve linear Least Squared
    results = optimize.lsq_linear(A, b, bounds=bounds)

    # Full model
    obs_dict['star_flx'] = star_flx_master / obs_dict['star_flx_cont'] * obs_dict['flx_spectro_cont'] 
    obs_dict['flx_spectro'] = b / weights
    flx_mod_spectro = np.dot(A[:,0], results.x[0]) / weights
    obs_dict['system'] = np.dot(A[:, 1:], results.x[1:])

    return results.x, flx_mod_spectro



# ----------------------------------------------------------------------------------------------------------------------



def hc_model_estimate_speckles_remove_continuum(obs_dict, flx_mod_spectro, flx_cont_mod, weights, bounds):
    '''
    Linear high-constrast model of planet and star contributions where we remove the continuums (see Wang et al. 2021)

    Args:
        obs_dict                     (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)
        flx_mod_spectro             (array): Model of the companion
        flx_cont_mod                (array): Continuum of the model of the companion
        weights                     (array): Weights to apply to the data
        bounds                      (tuple): Bounds to be applied to the estimated parameters

    Returns:
        - results                   (array): Results of the high-constrast model
        - flx_mod_spectro           (array): Model of the high-constrast model

    Authors: Allan Denis
    '''
    
    ind_star = 1 + len(obs_dict['star_flx'][0])
    if len(obs_dict['system']) > 0:
        ind_system = ind_star + len(obs_dict['system'][0])
    else:
        ind_system = ind_star


    # # # # # # Solve linear Least Squares A.x = b

    # Build matrix A
    A = np.zeros([np.size(obs_dict['flx_spectro']), ind_system])
    A[:, 0] = weights * (flx_mod_spectro - flx_cont_mod + np.mean(flx_mod_spectro))

    for star_i in range(len(obs_dict['star_flx'][0])):
        A[:, star_i+1] = weights * (obs_dict['star_flx'][:, star_i] - obs_dict['star_flx_cont'] + np.mean(obs_dict['star_flx'][:, star_i]))
        
    for system_i in range(ind_system-ind_star):
        A[:, system_i + ind_star] = weights * obs_dict['system'][:, system_i]

    # Build vector b
    b = weights * (obs_dict['flx_spectro'] - obs_dict['flx_spectro_cont']  + np.mean(obs_dict['flx_spectro']))

    # Solve linear Least Squares
    results = optimize.lsq_linear(A, b, bounds=bounds)

    # Full model
    flx_mod_spectro = np.dot(A, results.x) / weights
    obs_dict['flx_spectro'] = b / weights
    obs_dict['star_flx'] = np.dot(A[:, 1:ind_star], results.x[1:ind_star])
    obs_dict['system'] = np.dot(A[:, ind_star:], results.x[ind_star:])

    return results.x, flx_mod_spectro



# ----------------------------------------------------------------------------------------------------------------------



def hc_model_estimate_speckles_estimate_continuum():
    '''
    Linear high-constrast model of planet and star contributions where we fit the continuum
    To Be Defined

    Authors: Allan Denis
    '''

    return
