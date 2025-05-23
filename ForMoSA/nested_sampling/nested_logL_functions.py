import numpy as np

def logL_chi2(delta_flx, err, log_det, full=False):
    """
    Function to compute logL based on the classical chi2
    under the assumption of gaussian and spectrally uncorrelated noise.
    
    Args:
        delta_flx   (array): residual data-model as a function of wavelength
        err         (array): error (=standard deviation) of the observed spectrum as a function of wavelength
        full         (bool): True or False to add the usual constant terms
        log_det      (float): Log-determinant of the error bars
    Returns:
        - logL (float)     : the loglikelihood value
        
    Author: Matthieu Ravet
    """

    N = len(delta_flx)
    chi2 = np.nansum((delta_flx / err) ** 2)
    logL = - chi2 / 2
    if full == True:
        logL += - N/2 * np.log(2*np.pi) - 1/2 * log_det

    return logL


def logL_chi2_covariance(delta_flx, inv_cov, log_det, full=False):
    """
    Function to compute logL based on the generalized chi2
    under the assumption of gaussian and spectrally correlated noise.
    
    Args:
        delta_flx    (array): residual data-model as a function of wavelength
        inv_cov    (n-array): inverse of the covariance matrix of the observed spectrum as a function of wavelength
        full          (bool): True or False to add the usual constant terms
        log_det      (float): Log-determinant of the covariance matrix
    Returns:
        - logL (float)       : the loglikelihood value
        
    Author: Matthieu Ravet
    """

    N = len(delta_flx)
    chi2 = np.dot(delta_flx, np.dot(inv_cov, delta_flx))
    logL = - chi2 / 2
    if full == True:
        logL += -N/2 * np.log(2*np.pi) - 1/2 * log_det

    return logL


def logL_chi2_noisescaling(delta_flx, err, log_det, full=False):
    """
    Function to compute logL based on the chi2 with a fitted noise scaling s (marginalized)
    under the assumption of gaussian and spectrally uncorrelated noise.
    
    Args:
        delta_flx   (array): residual data-model as a function of wavelength
        err         (array): error (=standard deviation) of the observed spectrum as a function of wavelength
        full         (bool): True or False to add the usual constant terms
        log_det      (float): Log-determinant of the error bars matrix
    Returns:
        - logL (float)     : the loglikelihood value
        
    Author: Allan Denis and Matthieu Ravet
    """
    
    N = len(delta_flx)
    chi2 = np.nansum((delta_flx / err) ** 2)
    logL = -N/2 * np.log(chi2/N)
    if full == True:
        logL += -N/2 - N/2 * np.log(2*np.pi) - 1/2 * log_det # X²/s2 = N/2
    
    return logL


def logL_chi2_noisescaling_covariance(delta_flx, inv_cov, log_det, full=False):
    """
    Function to compute logL based on the chi2 with a fitted noise scaling s (marginalized)
    under the assumption of gaussian and spectrally correlated noise.
    
    Args:
        delta_flx    (array): residual data-model as a function of wavelength
        inv_cov    (n-array): inverse of the covariance matrix of the observed spectrum as a function of wavelength
        full          (bool): True or False to add the usual constant terms
        log_det      (float): Log-determinant of the covariance matrix
    Returns:
        - logL (float)     : the loglikelihood value
        
    Author: Allan Denis and Matthieu Ravet
    """

    N = len(delta_flx)
    chi2 = np.dot(delta_flx, np.dot(inv_cov, delta_flx))
    logL = -N/2 * np.log(chi2/N)
    if full == True:
        logL += -N/2 - N/2 * np.log(2*np.pi) - 1/2 * log_det # det(A) = 1 / det(A-1) and X²/s2 = N/2
    
    return logL


def logL_CCF_Brogi(flx_obs, flx_mod):
    """
    Function to compute logL based on the CCF mapping from Brogi et al. 2019
    under the assumption of gaussian and spectrally constant noise.
    
    Args:
        flx_obs     (array): flux of the observation as a function of wavelength
        flx_mod     (array): flux of the model as a function of wavelength
    Returns:
        - logL (float)     : the loglikelihood value
        
    Author: Matthieu Ravet
    """

    N = len(flx_mod)
    Sf2 = 1/N * np.nansum(np.square(flx_obs))
    Sg2 = 1/N * np.nansum(np.square(flx_mod))
    R = 1/N * np.nansum(flx_obs * flx_mod)

    logL = -N/2 * np.log(Sf2 - 2*R + Sg2)

    return logL


def logL_CCF_Zucker(flx_obs, flx_mod):
    """
    Function to compute logL based on the CCF mapping from Zucker 2003
    under the assumption of gaussian and spectrally constant noise.
    
    Args:
        flx_obs     (array): flux of the observation as a function of wavelength
        flx_mod     (array): flux of the model as a function of wavelength
    Returns:
        - logL (float)      : the loglikelihood value
        
    Author: Matthieu Ravet
    """

    N = len(flx_mod)
    Sf2 = 1/N * np.nansum(np.square(flx_obs))
    Sg2 = 1/N * np.nansum(np.square(flx_mod))
    R = 1/N * np.nansum(flx_obs * flx_mod)
    C2 = (R**2)/(Sf2 * Sg2)

    logL = -N/2 * np.log(1-C2)

    return logL


def logL_CCF_custom(flx_obs, flx_mod, err_obs):
    """
    Function to compute logL based on the custom CCF mapping from Me
    under the assumption of gaussian and spectrally constant noise.
    
    Args:
        flx_obs     (array): flux of the observation as a function of wavelength
        flx_mod     (array): flux of the model as a function of wavelength
        err_obs     (array): errors of the observation as a function of wavelength
    Returns:
        - logL (float)       : the loglikelihood value
        
    Author: Matthieu Ravet
    """

    N = len(flx_mod)
    Sf2 = 1/N * np.nansum(np.square(flx_obs))
    Sg2 = 1/N * np.nansum(np.square(flx_mod))
    R = 1/N * np.nansum(flx_obs * flx_mod)
    sigma2_weight = 1/(1/N * np.nansum(1/err_obs**2))

    logL = -N/(2*sigma2_weight) * (Sf2 + Sg2 - 2*R)

    return logL