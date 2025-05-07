from scipy.special import ndtri
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------

def uniform_prior(prior_fct_arg, theta):
    '''
    Uniform prior for nested sampling.

    Args:
        prior_fct_arg   (list): Two-values list with uniform prior boundaries.
        theta           (list): Parameter values randomly picked by the nested sampling
    Returns:
        - Evaluated      (float): Evaluated prior

    Author: Simon Petrus
    '''
    arg1 = float(prior_fct_arg[0])
    arg2 = float(prior_fct_arg[1])

    return (arg2 - arg1) * theta + arg1

def loguniform_prior(prior_fct_arg, theta):
    '''
    LogUniform prior for nested sampling.

    Args:
        prior_fct_arg   (list): Two-values list with loguniform prior boundaries.
        theta           (list): Parameter values randomly picked by the nested sampling
    Returns:
        - Evaluated      (float): Evaluated prior

    Author: Simon Petrus
    '''
    arg1 = float(prior_fct_arg[0])
    arg2 = float(prior_fct_arg[1])

    if arg1 <=0 or arg2 <= 0:
        print('WARNING : You cannot use negative priors with "loguniform_prior"')
        exit()

    return np.exp(np.log(arg1) + theta * (np.log(arg2) - np.log(arg1))) #arg1 * arg2 / ( (arg2 - arg1 ) * theta + arg1)

def gaussian_prior(prior_fct_arg, theta):
    '''
    Gaussian prior for nested sampling.

    Args:
        prior_fct_arg   (list): Two-values list with uniform prior boundaries.
        theta           (list): Parameter values randomly picked by the nested sampling
    Returns:
        - Evaluated      (float): Evaluated prior

    Author: Simon Petrus
    '''
    arg1 = float(prior_fct_arg[0])
    arg2 = float(prior_fct_arg[1])

    return arg1 + arg2 * ndtri(theta)
