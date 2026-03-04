import numpy as np
from scipy.special import ndtri

from ForMoSA.core.errors import ForMoSAError

def uniform_prior(bounds: list, theta: float) -> float:
    '''
    Uniform prior for nested sampling.

    Parameters
    ----------
    bounds : list
        Uniform prior boundaries
    theta : float
        Parameter value between 0 and 1

    Returns
    -------
    float
        Evaluated prior

    Authors
    -------
    Simon Petrus
'''

    if theta < 0 or theta > 1:
        raise ForMoSAError(f'Wrong value for theta: {theta}. Expected between 0 and 1')

    arg1 = float(bounds[0])
    arg2 = float(bounds[1])

    return (arg2 - arg1) * theta + arg1

def loguniform_prior(prior_fct_arg: list, theta: float) -> float:
    '''
    LogUniform prior for nested sampling.

    Parameters
    ----------
    prior_fct_arg : list
        Loguniform prior boundaries.
    theta : float
        Parameter values randomly picked by the nested sampling

    Returns
    -------
    float
        Evaluated prior

    Authors
    -------
    Simon Petrus
'''

    if theta < 0 or theta > 1:
        raise ForMoSAError(f'Wrong value for theta: {theta}. Expected between 0 and 1')

    arg1 = float(prior_fct_arg[0])
    arg2 = float(prior_fct_arg[1])

    return np.exp(np.log(arg1) + theta * (np.log(arg2) - np.log(arg1))) #arg1 * arg2 / ( (arg2 - arg1 ) * theta + arg1)

def gaussian_prior(mean: float, std: float, theta: float):
    '''
    Gaussian prior for nested sampling.

    Parameters
    ----------
    mean : float
        Gaussian prior mean
    std : float
        Gaussian prior standard deviation
    theta : float
        Parameter values randomly picked by the nested sampling

    Returns
    -------
    float
        Evaluated prior

    Authors
    -------
    Simon Petrus
'''

    if theta < 0 or theta > 1:
        raise ForMoSAError(f'Wrong value for theta: {theta}. Expected between 0 and 1')

    return mean + std * ndtri(theta)
