import numpy as np
import scipy.optimize as optimize


def _hc_model_remove_speckles(flx_obs_spectro: np.ndarray, flx_cont_obs_spectro: np.ndarray, transm_obs_spectro: np.ndarray, star_flx_obs_spectro: np.ndarray, star_flx_cont_obs_spectro: np.ndarray, flx_mod_spectro: np.ndarray, flx_cont_mod_spectro: np.ndarray):
    '''
    high-constrast

    Args:
        flx_obs_spectro             (array): Flux of the data
        flx_cont_obs_spectro        (array): Continuum of the data
        transm_obs_spectro          (array): Transmission
        star_flx_obs_spectro        (array): Flux of the star data
        star_flx_cont_obs_spectro   (array): Continuum of the star data
        flx_mod_spectro             (array): Model of the companion
        flx_cont_mod_spectro        (array): Continuum of the model of the companion

    Returns:
        flx_mod_spectro           (array): High-resolution content of planet model
        speckles                  (array): Speckles contribution

    Authors: Allan Denis
    '''

    speckles = star_flx_obs_spectro[:, len(star_flx_obs_spectro[0]) // 2] / star_flx_cont_obs_spectro * flx_cont_obs_spectro  # Speckles modulation (Bidot et al. 2023, Landman et al. 2023)
    flx_mod_spectro = transm_obs_spectro * (flx_mod_spectro - flx_cont_mod_spectro)

    return flx_mod_spectro, speckles


def _hc_model_estimate_speckles(flx_obs_spectro: np.ndarray, flx_cont_obs_spectro: np.ndarray, transm_obs_spectro: np.ndarray, star_flx_obs_spectro: np.ndarray, star_flx_cont_obs_spectro: np.ndarray, flx_mod_spectro: np.ndarray, flx_cont_mod_spectro: np.ndarray, err: np.ndarray, bounds: tuple, system_obs_spectro: np.ndarray = np.array([])):
    '''
    high-constrast model of planet and star contributions

    Args:
        flx_obs_spectro             (array): Flux of the data
        flx_cont_obs_spectro        (array): Continuum of the data
        transm_obs_spectro          (array): Transmission
        star_flx_obs_spectro        (array): Flux of the star data
        star_flx_cont_obs_spectro   (array): Continuum of the star data
        flx_mod_spectro             (array): Model of the companion
        flx_cont_mod_spectro        (array): Continuum of the model of the companion
        weights                     (array): Weights to apply to the data
        bounds                      (tuple): Bounds to be applied to the estimated parameters
        system_obs_spectro          (array): Systematics

    Returns:
        results.x                 (array): Results of the high-constrast model
        flx_mod_spectro           (array): Model of the high-constrast model
        speckles                  (array): Speckles contribution

    Authors: Allan Denis
    '''

    speckles = np.ones((star_flx_obs_spectro.shape))
    for star_i in range(len(star_flx_obs_spectro[0])):
        speckles[:,star_i] = star_flx_obs_spectro[:,star_i] / star_flx_cont_obs_spectro

    weights = 1 / (err**2)
    ind_star = 1 + len(speckles[0])
    if len(system_obs_spectro) > 0:
        ind_system = ind_star + len(system_obs_spectro[0])
    else:
        ind_system = ind_star

    # # # # # # Solve linear Least Squares A.x = b

    # Build matrix A
    A = np.zeros([np.size(flx_obs_spectro), ind_system])
    A[:, 0] = weights * transm_obs_spectro * (flx_mod_spectro - flx_cont_mod_spectro * speckles[:, len(speckles[0]) // 2])

    for star_i in range(len(speckles[0])):
        A[:, star_i + 1] = weights * (speckles[:,star_i] * flx_cont_obs_spectro)

    for system_i in range(ind_system - ind_star):
        A[:, system_i + ind_star] = weights * system_obs_spectro[:, system_i]

    # Build vector b
    b = weights * flx_obs_spectro
    # Solve linear Least Squares
    results = optimize.lsq_linear(A, b, bounds=bounds)

    # Model
    flx_mod_spectro = np.dot(A[:, 0], results.x[0]) / weights
    # Speckles
    speckles = np.dot(A[:, 1:ind_star], results.x[1:ind_star]) / weights

    return results.x, flx_mod_spectro, speckles




