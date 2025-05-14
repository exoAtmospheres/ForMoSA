from scipy.special import ndtri
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------

def uniform_prior(bounds, theta):
    '''
    Uniform prior for nested sampling.

    Args:
        bounds          (list): Uniform prior boundaries.
        theta           (list): Parameter values randomly picked by the nested sampling
    Returns:
        - Evaluated      (float): Evaluated prior

    Author: Simon Petrus
    '''
    arg1 = float(bounds[0])
    arg2 = float(bounds[1])

    return (arg2 - arg1) * theta + arg1

def loguniform_prior(prior_fct_arg, theta):
    '''
    LogUniform prior for nested sampling.

    Args:
        prior_fct_arg   (list): Loguniform prior boundaries.
        theta           (list): Parameter values randomly picked by the nested sampling
    Returns:
        - Evaluated      (float): Evaluated prior

    Author: Simon Petrus
    '''
    arg1 = float(prior_fct_arg[0])
    arg2 = float(prior_fct_arg[1])

    return np.exp(np.log(arg1) + theta * (np.log(arg2) - np.log(arg1))) #arg1 * arg2 / ( (arg2 - arg1 ) * theta + arg1)

def gaussian_prior(mean, std, theta):
    '''
    Gaussian prior for nested sampling.

    Args:
        mean           (float): Gaussian prior mean
        std            (float): Gaussian prior standard deviation
        theta           (list): Parameter values randomly picked by the nested sampling
    Returns:
        - Evaluated      (float): Evaluated prior

    Author: Simon Petrus
    '''

    return mean + std * ndtri(theta)
