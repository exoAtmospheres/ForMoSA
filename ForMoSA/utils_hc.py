import numpy as np
import scipy.optimize as optimize


def hc_model_nonlinear_estimate_speckles(obs_dict_spectro, flx_mod_spectro, flx_cont_mod, weights, bounds):
    '''
    Non linear high-constrast model of planet and star contributions (see Landman et al. 2023)
    This model is in principle more general than hc_model_estimate_speckles 
    because in the latter case we make the assumption that the star speckels dominate the data which is not the case here

    Args:
        obs_dict_spectro     (dict): Dictionay containing all the observationnal entries (spectroscopy)
        flx_mod_spectro     (array): Model of the companion
        flx_cont_mod        (array): Continuum of the model of the companion
        weights             (array): Weights to apply to the data
        bounds              (tuple): Bounds to be applied to the estimated parameters

    Returns:
        - results           (array): Results of the high-constrast model
        - flx_mod_spectro   (array): Model of the high-constrast model

    Authors: Allan Denis
    '''
    ind_star = 1 + len(obs_dict_spectro['star_flx'][0])
    # # # # # # # Solve non linear Least Squares full_model(theta) = flx_obs

    # Definition of f
    def f(theta):
        star_speckles = np.dot(theta[1:ind_star], obs_dict_spectro['star_flx'].T / obs_dict_spectro['star_flx_cont'] * (obs_dict_spectro['flx_cont']  - theta[0] * flx_cont_mod))
        results = theta[0] * flx_mod_spectro + star_speckles
        if len(theta) > ind_star:
            results += np.dot(theta[ind_star:], obs_dict_spectro['system'].T) 
        return weights * (results - obs_dict_spectro['flx'])
              

    # Solve non linear Least Squares
    # Initial guess for the planetary contribution
    theta0 = [0]
    for i in range(len(obs_dict_spectro['star_flx'][0])):
        # Arbitrary initial guesses for star speckles contribution
        theta0.append(((i+1) / len(obs_dict_spectro['star_flx'][0]))**2)
        
    if len(obs_dict_spectro['system']) > 0:
        for i in range(len(obs_dict_spectro['system'][0])):
            # Arbitrary initial guesses for systematics contribution
            theta0.append(1)
    # Solve non linear Least
    results = optimize.least_squares(f, theta0, bounds=bounds)
    
    
    # Full model
    flx_mod_spectro_full = f(results.x) / weights + obs_dict_spectro['flx']
    obs_dict_spectro['star_flx'] = np.dot(results.x[1:ind_star], obs_dict_spectro['star_flx'].T / obs_dict_spectro['star_flx_cont'] * (obs_dict_spectro['flx_cont']  - results.x[0] * flx_cont_mod))
    obs_dict_spectro['system'] = np.dot(results.x[ind_star:], obs_dict_spectro['system'].T)

    return results, flx_mod_spectro_full



# ----------------------------------------------------------------------------------------------------------------------



def hc_model_estimate_speckles(obs_dict_spectro, flx_mod_spectro, flx_mod_spectro_cont, star_flx_master, weights, bounds):
    '''
    Linear high-constrast model of planet and star contributions under the assumtion that the star speckles dominate the data  (see Landman et al. 2023)

    Args:
        obs_dict_spectro            (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)
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

    ind_star = 1 + len(obs_dict_spectro['star_flx'][0])
    if len(obs_dict_spectro['system']) > 0:
        ind_system = ind_star + len(obs_dict_spectro['system'][0])
    else:
        ind_system = ind_star

    # # # # # # Solve linear Least Squares A.x = b

    # Build matrix A
    A = np.zeros([np.size(obs_dict_spectro['flx']), ind_system])
    A[:, 0] = weights * obs_dict_spectro['transm'] * (flx_mod_spectro - flx_mod_spectro_cont *
                             star_flx_master / obs_dict_spectro['star_flx_cont'])

    for star_i in range(len(obs_dict_spectro['star_flx'][0])):
        A[:, star_i + 1] = weights * (obs_dict_spectro['star_flx'][:, star_i] / obs_dict_spectro['star_flx_cont'] * obs_dict_spectro['flx_cont'] )
            
    for system_i in range(ind_system - ind_star):
        A[:, system_i + ind_star] = weights * obs_dict_spectro['system'][:, system_i]

    # Build vector b
    b = weights * obs_dict_spectro['flx']
    # Solve linear Least Squares
    results = optimize.lsq_linear(A, b, bounds=bounds)

    # Full model
    flx_mod_spectro = np.dot(A, results.x) / weights
    # Speckles
    speckles = np.dot(A[:, 1:ind_star], results.x[1:ind_star]) / weights

    return results.x, flx_mod_spectro, speckles



# ----------------------------------------------------------------------------------------------------------------------



def hc_model_remove_speckles(obs_dict_spectro, flx_mod_spectro, flx_mod_spectro_cont, star_flx_master, weights, bounds):
    '''
    Linear high-constrast model of planet contribution only where the speckles are filtered out from the data (see Landman et al. 2023)

    Args:
        obs_dict_spectro                     (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)
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

    if len(obs_dict_spectro['system']) > 0:
        ind_system = 1 + len(obs_dict_spectro['system'][0])
    else:
        ind_system = 1

    # # # # # # # Solve linear Least Squared A.x = b
    A = np.zeros([np.size(obs_dict_spectro['flx']), ind_system])

    # Build matrix A
    A[:, 0] = weights * obs_dict_spectro['transm'] * (flx_mod_spectro - flx_mod_spectro_cont *
                             star_flx_master / obs_dict_spectro['star_flx_cont'])
    
    for system_i in range(ind_system-1):
        A[:, system_i + 1] = weights * obs_dict_spectro['system'][:, system_i]

    # Build vector b
    b = weights * (obs_dict_spectro['flx'] - star_flx_master /
                       obs_dict_spectro['star_flx_cont'] * obs_dict_spectro['flx_cont'] )

    # Solve linear Least Squared
    results = optimize.lsq_linear(A, b, bounds=bounds)

    # Full model
    speckles = obs_dict_spectro['flx'] - b / weights
    flx_mod_spectro = np.dot(A[:,0], results.x[0]) / weights

    return results.x, flx_mod_spectro, speckles


# ----------------------------------------------------------------------------------------------------------------------



def hc_model_estimate_speckles_estimate_continuum():
    '''
    Linear high-constrast model of planet and star contributions where we fit the continuum
    To Be Defined

    Authors: Allan Denis
    '''

    return
