import numpy as np
import scipy.optimize as optimize
from adapt.extraction_functions import continuum_estimate


def forward_model(global_params, wav_mod_spectro, res_mod_spectro, flx_cont_obs, flx_mod, star_flx_obs, star_flx_cont_obs, err_obs, transm_obs, flx_obs, system_obs, indobs):
    '''
    For high-contrast companions, where the star speckles signal contaminate the data

    Args:
        global_params      (object): Class containing each parameter used in ForMoSA
        wav_mod_spectro     (array): Wavelength grid of the model
        res_mod_spectro     (array): Resolution grid of the model
        flx_cont_obs        (array): Continuum of the data
        flx_mod             (array): Model of the companion
        star_flx_obs        (array): Star data
        star_flx_cont_obs   (array): Continuum of star data
        err_obs             (array): Noise of the data
        transm_obs          (array): Transmission
        flx_obs             (array): Data
        system_obs          (array): Systematics
        indobs                (int): Index of the current observation loop
    
    Returns:
        - res                 (array): Residuals of the forward model
        - flx_mod             (array): Model of the forward model
        - flx_obs             (array): Data of the forward model
        - star_flx_obs        (array): Star flux of the forward model
        - system_obs          (array): Systematics of the forward model

    Authors: Allan Denis
    '''

    flx_mod *= transm_obs
    flx_cont_mod = continuum_estimate(global_params, wav_mod_spectro, flx_mod, res_mod_spectro, indobs)

    star_flx_obs_master = star_flx_obs[:, len(star_flx_obs[0]) // 2]

    bounds = (float(global_params.bounds_lsq[indobs][1:-1].split(',')[0]), 
              float(global_params.bounds_lsq[indobs][1:-1].split(',')[1]))
    
    weights = (1 / err_obs)**2  # For now we consider diagonal covariance matrices only

    if global_params.fm_type[indobs] == 'nonlinear_fit_spec':
        res, flx_mod, flx_obs, star_flx_obs, system_obs = forward_model_nonlinear_estimate_speckles(
            flx_cont_obs, flx_mod, flx_cont_mod, star_flx_obs_master, star_flx_obs, star_flx_cont_obs, weights, flx_obs, system_obs, bounds)
    elif global_params.fm_type[indobs] == 'fit_spec':
        res, flx_mod, flx_obs, star_flx_obs, system_obs = forward_model_estimate_speckles(
            flx_cont_obs, flx_mod, flx_cont_mod, star_flx_obs_master, star_flx_obs, star_flx_cont_obs, weights, flx_obs, system_obs, bounds)
    elif global_params.fm_type[indobs] == 'rm_spec':
        res, flx_mod, flx_obs, star_flx_obs, system_obs = forward_model_remove_speckles(
            flx_cont_obs, flx_mod, flx_cont_mod, star_flx_obs_master, star_flx_obs, star_flx_cont_obs, weights, flx_obs, system_obs, bounds)
    elif global_params.fm_type[indobs] == 'fit_spec_rm_cont':
        res, flx_mod, flx_obs, star_flx_obs, system_obs = forward_model_estimate_speckles_remove_continuum(
            flx_cont_obs, flx_mod, flx_cont_mod, star_flx_obs_master, star_flx_obs, star_flx_cont_obs, weights, flx_obs, system_obs, bounds)
    elif global_params.fm_type[indobs] == 'fit_spec_fit_cont':
        raise Exception(
            'Continuum fitting-based forward model no implement yet ! Please use another function')
        
    return res, flx_mod, flx_obs, star_flx_obs, system_obs


def forward_model_nonlinear_estimate_speckles(flx_cont_obs, flx_mod, flx_cont_mod, star_flx_obs_master, star_flx_obs, star_flx_cont_obs, weights, flx_obs, system_obs, bounds):
    '''
    Non linear forward model of planet and star contributions (see Landman et al. 2023)
    This model is in principle more general than forward_model_estimate_speckles 
    because in the latter case we make the assumption that the star speckels dominate the data which is not the case here

    Args:
        flx_cont_obs        (array): Continuum of the data
        flx_mod             (array): Model of the companion
        flx_cont_mod        (array): Continuum of the model of the companion
        star_flx_obs_master (array): Master star data
        star_flx_obs        (array): Star data
        star_flx_cont_obs   (array): Continuum of star data
        weights             (array): Weights to apply to the data
        flx_obs             (array): Data
        system_obs          (array): Systematics
        bounds              (tuple): Bounds to be applied to the estimated parameters

    Returns:
        - res                 (array): Residuals of the forward model
        - flx_mod             (array): Model of the forward model
        - flx_obs             (array): Data of the forward model
        - star_flx_obs        (array): Star flux of the forward model
        - system_obs          (array): Systematics of the forward model

    Authors: Allan Denis
    '''

    ind_star = 1 + len(star_flx_obs[0])
    # # # # # # # Solve non linear Least Squares full_model(theta) = flx_obs

    # Definition of f
    def f(theta):
        res = theta[0] * flx_mod + np.dot(theta[1:ind_star], star_flx_obs / star_flx_cont_obs * (flx_cont_obs - theta[0] * flx_cont_mod))
        if len(theta) > ind_star:
            res += np.dot(theta[ind_star:], system_obs) - flx_obs
            return weights * res
              

    # Solve non linear Least Squares
    # Initial guess for the planetary contribution
    theta0 = [0]
    for i in range(len(star_flx_obs[0])):
        # Arbitrary initial guesses for star speckles contribution
        theta0.append((i / len(star_flx_obs[0]))**2)
        
    if len(system_obs) > 0:
        for i in range(len(system_obs[0])):
            # Arbitrary initial guesses for systematics contribution
            theta0.append(1)
            
    # Solve non linear Least Squares
    res = optimize.least_squares(f, theta0, bounds=bounds)

    # Full model
    flx_mod_full = f(res.x) / weights + flx_obs
    star_flx_obs = np.dot(res.x[1:ind_star], star_flx_obs / star_flx_cont_obs * flx_cont_obs)
    system_obs = np.dot(res.x[ind_star:], system_obs)

    return res.x, flx_mod_full, flx_obs, star_flx_obs, system_obs


def forward_model_estimate_speckles(flx_cont_obs, flx_mod, flx_cont_mod, star_flx_obs_master, star_flx_obs, star_flx_cont_obs, weights, flx_obs, system_obs, bounds):
    '''
    Linear forward model of planet and star contributions under the assumtion that the star speckles dominate the data  (see Landman et al. 2023)

    Args:
        flx_cont_obs        (array): Continuum of the data
        flx_mod             (array): Model of the companion
        flx_cont_mod        (array): Continuum of the model of the companion
        star_flx_obs_master (array): Master star data
        star_flx_obs        (array): Star data
        star_flx_cont_obs   (array): Continuum of star data
        weights             (array): Weights to apply to the data
        flx_obs             (array): Data
        system_obs          (array): Systematics
        bounds              (tuple): Bounds to be applied to the estimated parameters

    Returns:
        - res                 (array): Residuals of the forward model
        - flx_mod             (array): Model of the forward model
        - flx_obs             (array): Data of the forward model
        - star_flx_obs        (array): Star flux of the forward model
        - system_obs          (array): Systematics of the forward model

    Authors: Allan Denis
    '''

    ind_star = 1 + len(star_flx_obs[0])
    if len(system_obs) > 0:
        ind_system = ind_star + len(system_obs[0])
    else:
        ind_system = ind_star

    # # # # # # Solve linear Least Squares A.x = b

    # Build matrix A
    A = np.zeros([np.size(flx_obs), ind_system])
    A[:, 0] = weights * (flx_mod - flx_cont_mod *
                             star_flx_obs_master / star_flx_cont_obs)

    for star_i in range(len(star_flx_obs[0])):
        A[:, star_i + 1] = weights * (star_flx_obs[:, star_i] / star_flx_cont_obs * flx_cont_obs)
            
    for system_i in range(ind_system - ind_star):
        A[:, system_i + ind_star] = weights * system_obs[:, system_i]

    # Build vector b
    b = weights * flx_obs
    # Solve linear Least Squares
    res = optimize.lsq_linear(A, b, bounds=bounds)

    # Full model
    flx_mod_full = np.dot(A, res.x) / weights
    star_flx_obs = np.dot(A[:, 1:ind_star], res.x[1:ind_star]) / weights
    system_obs = np.dot(A[:, ind_star:], res.x[ind_star:])

    return res.x, flx_mod_full, flx_obs, star_flx_obs, system_obs


def forward_model_remove_speckles(flx_cont_obs, flx_mod, flx_cont_mod, star_flx_obs_master, star_flx_cont_obs, weights, flx_obs, system_obs, bounds):
    '''
    Linear forward model of planet contribution only where the speckles are filtered out from the data (see Landman et al. 2023)

    Args:
        flx_cont_obs        (array): Continuum of the data
        flx_mod             (array): Model of the companion
        flx_cont_mod        (array): Continuum of the model of the companion
        star_flx_obs_master (array): Master star data
        star_flx_obs        (array): Star data
        star_flx_cont_obs   (array): Continuum of star data
        weights             (array): Weights to apply to the data
        flx_obs             (array): Data
        system_obs          (array): Systematics
        bounds              (tuple): Bounds to be applied to the estimated parameters

    Returns:
        - res                 (array): Residuals of the forward model
        - flx_mod             (array): Model of the forward model
        - flx_obs             (array): Data of the forward model
        - star_flx_obs        (array): Star flux of the forward model
        - system_obs          (array): Systematics of the forward model

    Authors: Allan Denis
    '''

    if len(system_obs) > 0:
        ind_system = 1 + len(system_obs[0])
    else:
        ind_system = 1

    # # # # # # # Solve linear Least Squared A.x = b
    A = np.zeros([np.size(flx_obs), ind_system])

    # Build matrix A
    A[:, 0] = weights * (flx_mod - flx_cont_mod *
                             star_flx_obs_master / star_flx_cont_obs)
    
    for system_i in range(ind_system-1):
        A[:, system_i + 1] = weights * system_obs[:, system_i]

    # Build vector b
    b = weights * (flx_obs - star_flx_obs_master /
                       star_flx_cont_obs * flx_cont_obs)

    # Solve linear Least Squared
    res = optimize.lsq_linear(A, b, bounds=bounds)

    # Full model
    star_flx_obs = star_flx_obs_master / star_flx_cont_obs + flx_cont_obs
    flx_mod_full = np.dot(A[:,0], res.x[0]) / weights + star_flx_obs
    system_obs = np.dot(A[:, 1:], res.x[1:])

    return res.x, flx_mod_full, flx_obs, star_flx_obs, system_obs


def forward_model_estimate_speckles_remove_continuum(flx_cont_obs, flx_mod, flx_cont_mod, star_flx_obs_master, star_flx_obs, star_flx_cont_obs, weights, flx_obs, system_obs, bounds):
    '''
    Linear forward model of planet and star contributions where we remove the continuums (see Wang et al. 2021)

    Args:
        flx_cont_obs        (array): Continuum of the data
        flx_mod             (array): Model of the companion
        flx_cont_mod        (array): Continuum of the model of the companion
        star_flx_obs_master (array): Master star data
        star_flx_obs        (array): Star data
        star_flx_cont_obs   (array): Continuum of star data
        weights             (array): Weights to apply to the data
        flx_obs             (array): Data
        system_obs          (array): Systematics
        bounds              (tuple): Bounds to be applied to the estimated parameters

    Returns:
        - res                 (array): Residuals of the forward model
        - flx_mod             (array): Model of the forward model
        - flx_obs             (array): Data of the forward model
        - star_flx_obs        (array): Star flux of the forward model
        - system_obs          (array): Systematics of the forward model

    Authors: Allan Denis
    '''
    
    ind_star = 1 + len(star_flx_obs[0])
    if len(system_obs) > 0:
        ind_system = ind_star + len(system_obs[0])
    else:
        ind_system = ind_star


    # # # # # # Solve linear Least Squares A.x = b

    # Build matrix A
    A = np.zeros([np.size(flx_obs), ind_system])
    A[:, 0] = weights * (flx_mod - flx_cont_mod + np.mean(flx_mod))

    for star_i in range(len(star_flx_obs[0])):
        A[:, star_i+1] = weights * (star_flx_obs[:, star_i] - star_flx_cont_obs + np.mean(star_flx_obs[:, star_i]))
        
    for system_i in range(ind_system-ind_star):
        A[:, system_i + ind_star] = weights * system_obs[:, system_i]

    # Build vector b
    b = weights * (flx_obs - flx_cont_obs + np.mean(flx_obs))

    # Solve linear Least Squares
    res = optimize.lsq_linear(A, b, bounds=bounds)

    # Full model
    flx_mod_full = np.dot(A, res.x) / weights
    flx_obs = b / weights
    star_flx_obs = np.dot(A[:, 1:ind_star], res.x[1:ind_star])
    system_obs = np.dot(A[:, ind_star:], res.x[ind_star:])

    return res.x, flx_mod_full, flx_obs, star_flx_obs, system_obs


def forward_model_estimate_speckles_estimate_continuum():
    '''
    Linear forward model of planet and star contributions where we fit the continuum
    To Be Defined

    Authors: Allan Denis
    '''

    return
