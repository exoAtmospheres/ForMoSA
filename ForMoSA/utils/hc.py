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


def hc_model(obs_dict_spectro, flx_mod_spectro, flx_mod_spectro_cont, weights, bounds, loglike):
    '''
    Linear high-constrast model of planet and star contributions

    Args:
        obs_dict_spectro            (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)
        flx_mod_spectro             (array): Model of the companion
        flx_cont_mod                (array): Continuum of the model of the companion
        star_flx_master             (array): Master star data
        weights                     (array): Weights to apply to the data
        bounds                      (tuple): Bounds to be applied to the estimated parameters
        loglike                       (str): Loglikelihood function

    Returns:
        results                   (array): Results of the high-constrast model
        flx_mod_spectro           (array): Model of the high-constrast model

    Authors: Allan Denis
    '''

    star_flx_master = obs_dict_spectro['star_flx'][:, len(obs_dict_spectro['star_flx'][0]) // 2]
    if not(loglike.startswith('CCF')) and len(obs_dict_spectro['star_flx'][0]) > 1:
        contributions, flx_mod_spectro, speckles = hc_model_estimate_speckles(obs_dict_spectro, flx_mod_spectro, flx_mod_spectro_cont, star_flx_master, weights, bounds)
    elif not(loglike.startswith('CCF')):
        contributions, flx_mod_spectro, speckles = hc_model_estimate_planet(obs_dict_spectro, flx_mod_spectro, flx_mod_spectro_cont, star_flx_master, weights, bounds)
    else:
        flx_mod_spectro, speckles = hc_model_remove_speckles(obs_dict_spectro, flx_mod_spectro, flx_mod_spectro_cont, star_flx_master)
        contributions = 1

    return contributions, flx_mod_spectro, speckles



def hc_model_estimate_speckles(obs_dict_spectro, flx_mod_spectro, flx_mod_spectro_cont, star_flx_master, weights, bounds):
    '''
    Linear high-constrast model of planet and star contributions under the assumtion that the star speckles dominate the data  (see Landman et al. 2023)

    Args:
        obs_dict_spectro             (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)
        flx_mod_spectro             (array): Model of the companion
        flx_cont_mod                (array): Continuum of the model of the companion
        star_flx_master             (array): Master star data
        weights                     (array): Weights to apply to the data
        bounds                      (tuple): Bounds to be applied to the estimated parameters

    Returns:
        results.x                 (array): Results of the high-constrast model
        flx_mod_spectro           (array): Model of the high-constrast model
        speckles                  (array): Estimated speckles

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

    # Model
    flx_mod_spectro = np.dot(A[:, 0], results.x[0]) / weights
    # Speckles
    speckles = np.dot(A[:, 1:ind_star], results.x[1:ind_star]) / weights

    return results.x, flx_mod_spectro, speckles


def hc_model_estimate_planet(obs_dict_spectro, flx_mod_spectro, flx_mod_spectro_cont, star_flx_master, weights, bounds):
    '''
    Linear high-constrast model of planet contribution only where the speckles are filtered out from the data (see Landman et al. 2023)

    Args:
        obs_dict_spectro             (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)
        flx_mod_spectro             (array): Model of the companion
        flx_cont_mod                (array): Continuum of the model of the companion
        star_flx_master             (array): Master star data
        weights                     (array): Weights to apply to the data
        bounds                      (tuple): Bounds to be applied to the estimated parameters

    Returns:
        speckles                  (array): Results of the high-constrast model
        flx_mod_spectro           (array): Model of the high-constrast model

    Authors: Allan Denis
    '''

    speckles = star_flx_master / obs_dict_spectro['star_flx_cont'] * obs_dict_spectro['flx_cont']
    if len(obs_dict_spectro['system']) > 0:
        ind_system = len(obs_dict_spectro['system'][0])
    else:
        ind_system = 0

    # # # # # # Solve linear Least Squares A.x = b

    # Build matrix A
    A = np.zeros([np.size(obs_dict_spectro['flx']), ind_system+1])
    A[:, 0] = weights * obs_dict_spectro['transm'] * (flx_mod_spectro - flx_mod_spectro_cont *
                             star_flx_master / obs_dict_spectro['star_flx_cont'])

    for system_i in range(ind_system):
        A[:, system_i + 1] = weights * obs_dict_spectro['system'][:, system_i]

    # Build vector b
    b = weights * (obs_dict_spectro['flx'] - speckles)
    # Solve linear Least Squares
    results = optimize.lsq_linear(A, b, bounds=bounds)

    # Model
    flx_mod_spectro = np.dot(A[:, 0], results.x[0]) / weights

    return results.x, flx_mod_spectro, speckles


def hc_model_remove_speckles(obs_dict_spectro, flx_mod_spectro, flx_mod_spectro_cont, star_flx_master):
    '''
    Linear high-constrast model of planet contribution only where the speckles are filtered out from the data (see Landman et al. 2023)

    Args:
        obs_dict_spectro             (dict): Dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)
        flx_mod_spectro             (array): Model of the companion
        flx_cont_mod                (array): Continuum of the model of the companion
        star_flx_master             (array): Master star data
        weights                     (array): Weights to apply to the data
        bounds                      (tuple): Bounds to be applied to the estimated parameters

    Returns:
        speckles                  (array): Results of the high-constrast model
        flx_mod_spectro           (array): Model of the high-constrast model

    Authors: Allan Denis
    '''

    # Model
    flx_mod_spectro = obs_dict_spectro['transm'] * (flx_mod_spectro - flx_mod_spectro_cont)
    speckles = star_flx_master / obs_dict_spectro['star_flx_cont'] * obs_dict_spectro['flx_cont']

    return flx_mod_spectro, speckles


