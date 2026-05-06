from scipy.special import ndtri


def uniform_prior(prior_fct_arg, theta):
    arg1 = float(prior_fct_arg[0])
    arg2 = float(prior_fct_arg[1])
    return (arg2 - arg1) * theta + arg1


def gaussian_prior(prior_fct_arg, theta):
    arg1 = float(prior_fct_arg[0])
    arg2 = float(prior_fct_arg[1])
    return arg1 + arg2 * ndtri(theta)
