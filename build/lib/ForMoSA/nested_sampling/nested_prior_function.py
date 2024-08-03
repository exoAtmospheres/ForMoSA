from scipy.special import ndtri
# ----------------------------------------------------------------------------------------------------------------------

def uniform_prior(prior_fct_arg, theta):
    '''
    Uniform prior for nested sampling

    Args:
        prior_fct_arg   (list): Two-values list with uniform prior boundaries.
        theta           (list): Parameter values randomly picked by the nested sampling
    
    Returns:
        Evaluated      (float): Evaluated prior
    '''
    arg1 = float(prior_fct_arg[0])
    arg2 = float(prior_fct_arg[1])

    return (arg2 - arg1) * theta + arg1

def gaussian_prior(prior_fct_arg, theta):
    '''
    Gaussian prior for nested sampling

    Args:
        prior_fct_arg   (list): Two-values list with uniform prior boundaries.
        theta           (list): Parameter values randomly picked by the nested sampling
    Returns:
        Evaluated      (float): Evaluated prior
    '''
    arg1 = float(prior_fct_arg[0])
    arg2 = float(prior_fct_arg[1])

    return arg1 + arg2 * ndtri(theta)
