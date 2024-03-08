import numpy as np

def logL_chi2_classic(delta_flx, err):
    """
    Function to compute logL based on the classical chi2
    under the assumption of gaussian and spectrally uncorrelated noise
    
    Args:
        delta_flx   : residual data-model as a function of wavelength
        err         : error (=standard deviation) of the observed spectrum as a function of wavelength
    Returns:
        logL        : the loglikelihood value
        
    Author: Matthieu Ravet
    """

    chi2 = np.sum((delta_flx / err) ** 2)
    logL = - chi2 / 2

    return logL


def logL_chi2_covariance(delta_flx, inv_cov):
    """
    Function to compute logL based on the generalized chi2
    under the assumption of gaussian and spectrally correlated noise
    
    Args:
        delta_flx   : residual data-model as a function of wavelength
        inv_cov     : inverse of the covariance matrix of the observed spectrum as a function of wavelength
    Returns:
        logL        : the loglikelihood value
        
    Author: Matthieu Ravet
    """

    chi2 = np.dot(delta_flx, np.dot(inv_cov, delta_flx))
    logL = - chi2 / 2

    return logL

def logL_full_covariance(delta_flx, inv_cov):
    """
    Function to compute logL under the assumption of gaussian and spectrally correlated noise.
    This function is a generalized version of the logL_chi2_covariance and is to be used when dealing
    with GP extimation of the covariance matrix
    
    Args:
        delta_flx   : residual data-model as a function of wavelength
        inv_cov     : inverse of the covariance matrix of the observed spectrum as a function of wavelength
    Returns:
        logL        : the loglikelihood value
        
    Author: Matthieu Ravet
    """

    logL = - 1/2 * ( np.dot(delta_flx, np.dot(inv_cov, delta_flx)) + np.log(np.linalg.det(np.linalg.inv(inv_cov))) + len(delta_flx)*np.log(2*np.pi) )

    return logL


def logL_CCF_Brogi(flx_obs, flx_mod):
    """
    Function to compute logL based on the CCF mapping from Brogi et al. 2019
    under the assumption of gaussian and spectrally constant noise
    
    Args:
        flx_obs     : flux of the observation as a function of wavelength
        flx_mod     : flux of the model as a function of wavelength
    Returns:
        logL        : the loglikelihood value
        
    Author: Matthieu Ravet
    """

    N = len(flx_mod)
    Sf2 = 1/N * np.sum(np.square(flx_obs))
    Sg2 = 1/N * np.sum(np.square(flx_mod))
    R = 1/N * np.sum(flx_obs * flx_mod)

    logL = -N/2 * np.log(Sf2 - 2*R + Sg2)

    return logL


def logL_CCF_Zucker(flx_obs, flx_mod):
    """
    Function to compute logL based on the CCF mapping from Zucker 2003
    under the assumption of gaussian and spectrally constant noise
    
    Args:
        flx_obs : flux of the observation as a function of wavelength
        flx_mod : flux of the model as a function of wavelength
    Returns:
        logL    : the loglikelihood value
        
    Author: Matthieu Ravet
    """

    N = len(flx_mod)
    Sf2 = 1/N * np.sum(np.square(flx_obs))
    Sg2 = 1/N * np.sum(np.square(flx_mod))
    R = 1/N * np.sum(flx_obs * flx_mod)
    C2 = (R**2)/(Sf2 * Sg2)

    logL = -N/2 * np.log(1-C2)

    return logL


def logL_CCF_custom(flx_obs, flx_mod, err_obs):
    """
    Function to compute logL based on the custom CCF mapping from Me
    under the assumption of gaussian and spectrally constant noise
    
    Args:
        flx_obs : flux of the observation as a function of wavelength
        flx_mod : flux of the model as a function of wavelength
        err_obs : errors of the observation as a function of wavelength
    Returns:
        logL    : the loglikelihood value
        
    Author: Matthieu Ravet
    """

    N = len(flx_mod)
    Sf2 = 1/N * np.sum(np.square(flx_obs))
    Sg2 = 1/N * np.sum(np.square(flx_mod))
    R = 1/N * np.sum(flx_obs * flx_mod)
    sigma2_weight = 1/(1/N * np.sum(1/err_obs**2))

    logL = -N/(2*sigma2_weight) * (Sf2 + Sg2 - 2*R)

    return logL