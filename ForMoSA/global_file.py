from configobj import ConfigObj
import glob

# ----------------------------------------------------------------------------------------------------------------------

def get_config_value(config, section, key, default, cast=None):
    """
    Helper function to get a config value with a fallback default.

    Args:
        config               (obj): config object
        section              (str): config section name
        key                  (str): config key name
        default                   : default value if key is missing
        cast                      : a function to cast the value (e.g., int, list, eval, etc.)

    Returns:
        The value (possibly cast), and stores it back into self.config if it was missing.

    Author: Matthieu Ravet
    """
    # Ensure section exists
    if section not in config:
        config[section] = {}
    try:
        val = config[section][key]

    # Add key to section
    except:
        val = default
        config[section][key] = val

    if cast:
        try:
            if cast == list and not isinstance(val, list):
                val = [val]
            else:
                val = cast(val)
        except Exception:
            pass  # Fallback to raw

    return val


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class GlobFile:
    '''
    Class that import all the parameters from the config file and make them GLOBAL FORMOSA VARIABLES.

    Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
    '''

    def __init__(self, config_file_path):
        # Generate the confog object
        config = ConfigObj(config_file_path, encoding='utf8')
        self.config=config

        ## Read CONFIG: - - - - - - -

        # [config_path] (4)
        self.observation_path = config['config_path']['observation_path'] + '*'
        self.main_observation_path = config['config_path']['observation_path'] + '*'
        self.model_path = config['config_path']['model_path']
        self.adapt_store_path = config['config_path']['adapt_store_path']
        self.result_path = config['config_path']['result_path']
        grid_name = config['config_path']['model_path'].split('/')
        grid_name = grid_name[-1]
        grid_name = grid_name.split('.nc')
        grid_name = grid_name[0]
        self.grid_name = grid_name
        model_name = grid_name.split('_')
        model_name = model_name[0]
        self.model_name = model_name

        N_obs = len(glob.glob(self.main_observation_path)) # Number of obs to invert

        # [config_adapt] (5)
        self.method = get_config_value(self.config, 'config_adapt', 'method', 'linear', None)
        self.emulator = get_config_value(self.config, 'config_adapt', 'emulator', 'NA', list)
        self.target_res_obs = get_config_value(self.config, 'config_adapt', 'target_res_obs', 'obs', list)
        self.target_res_mod = get_config_value(self.config, 'config_adapt', 'target_res_mod', 'obs', list)
        self.res_cont = get_config_value(self.config, 'config_adapt', 'res_cont', 'NA', list)
        self.wav_cont = get_config_value(self.config, 'config_adapt', 'wav_cont', 'NA', list)

        # [config_inversion] (4)
        self.logL_type = get_config_value(self.config, 'config_inversion', 'logL_type', 'chi2', list)
        self.logL_full = get_config_value(self.config, 'config_inversion', 'logL_full', False, list)
        self.wav_fit = get_config_value(self.config, 'config_inversion', 'wav_fit', '0,100', list)
        self.ns_algo = get_config_value(self.config, 'config_inversion', 'ns_algo', 'nestle', None)
        self.npoint = get_config_value(self.config, 'config_inversion', 'npoint', '100', eval)

        # [config_highcont_models] (2)
        self.hc_type = get_config_value(self.config, 'config_highcont_models', 'hc_type', 'NA', list)
        self.hc_bounds_lsq = get_config_value(self.config, 'config_highcont_models', 'hc_bounds_lsq', 'NA', list)
    
        # [config_parameters] (11)
        self.par1 = get_config_value(self.config, 'config_parameters', 'par1', 'NA', list)
        self.par2 = get_config_value(self.config, 'config_parameters', 'par2', 'NA', list)
        self.par3 = get_config_value(self.config, 'config_parameters', 'par3', 'NA', list)
        self.par4 = get_config_value(self.config, 'config_parameters', 'par4', 'NA', list)
        self.par5 = get_config_value(self.config, 'config_parameters', 'par5', 'NA', list)
        self.r = get_config_value(self.config, 'config_parameters', 'r', 'NA', list)
        self.d = get_config_value(self.config, 'config_parameters', 'd', 'NA', list)
        self.alpha = get_config_value(self.config, 'config_parameters', 'alpha', 'NA', list)
        self.rv = get_config_value(self.config, 'config_parameters', 'rv', 'NA', list)
        self.av = get_config_value(self.config, 'config_parameters', 'av', 'NA', list)
        self.vsini = get_config_value(self.config, 'config_parameters', 'vsini', 'NA', list)
        self.ld = get_config_value(self.config, 'config_parameters', 'ld', 'NA', list)
        self.bb_t = get_config_value(self.config, 'config_parameters', 'bb_t', 'NA', list)
        self.bb_r = get_config_value(self.config, 'config_parameters', 'bb_r', 'NA', list)

        # [config_nestle] (8) (n_ prefix for params)
        self.n_method = get_config_value(self.config, 'config_nestle', 'method', 'single', None)
        self.n_update_interval = get_config_value(self.config, 'config_nestle', 'update_interval', 'None', eval)
        self.n_npdim = get_config_value(self.config, 'config_nestle', 'npdim', 'None', eval)
        self.n_maxiter = get_config_value(self.config, 'config_nestle', 'maxiter', 'None', eval)
        self.n_maxcall = get_config_value(self.config, 'config_nestle', 'maxcall', 'None', eval)
        self.n_dlogz = get_config_value(self.config, 'config_nestle', 'dlogz', 'None', eval)
        self.n_decline_factor = get_config_value(self.config, 'config_nestle', 'decline_factor', 'None', eval)
        self.n_rstate = get_config_value(self.config, 'config_nestle', 'rstate', 'None', eval)

        # [config_pymultinest] (20, pm_ prefix for params)
        self.pm_n_clustering_params = get_config_value(self.config, 'config_pymultinest', 'n_clustering_params', 'None', eval)
        self.pm_wrapped_params = get_config_value(self.config, 'config_pymultinest', 'wrapped_params', 'None', eval)
        self.pm_importance_nested_sampling = get_config_value(self.config, 'config_pymultinest', 'importance_nested_sampling', 'True', eval)
        self.pm_multimodal = get_config_value(self.config, 'config_pymultinest', 'multimodal', 'True', eval)
        self.pm_const_efficiency_mode = get_config_value(self.config, 'config_pymultinest', 'const_efficiency_mode', 'False', eval)
        self.pm_evidence_tolerance = get_config_value(self.config, 'config_pymultinest', 'evidence_tolerance', '0.5', eval)
        self.pm_sampling_efficiency = get_config_value(self.config, 'config_pymultinest', 'sampling_efficiency', '0.8', eval)
        self.pm_n_iter_before_update = get_config_value(self.config, 'config_pymultinest', 'n_iter_before_update', '100', eval)
        self.pm_null_log_evidence = get_config_value(self.config, 'config_pymultinest', 'null_log_evidence', '-1e90', eval)
        self.pm_max_modes = get_config_value(self.config, 'config_pymultinest', 'max_modes', '100', eval)
        self.pm_mode_tolerance = get_config_value(self.config, 'config_pymultinest', 'mode_tolerance', '-1e90', eval)
        self.pm_seed = get_config_value(self.config, 'config_pymultinest', 'seed', '-1', eval)
        self.pm_verbose = get_config_value(self.config, 'config_pymultinest', 'verbose', 'True', eval)
        self.pm_resume = get_config_value(self.config, 'config_pymultinest', 'resume', 'False', eval) # This is the only parameter not set by default to True, you can change it if your inversion crash and you don't want to start anew
        self.pm_context = get_config_value(self.config, 'config_pymultinest', 'context', '0', eval)
        self.pm_log_zero = get_config_value(self.config, 'config_pymultinest', 'log_zero', '-1e100', eval)
        self.pm_max_iter = get_config_value(self.config, 'config_pymultinest', 'max_iter', '0', eval) # Unlimited
        self.pm_init_MPI = get_config_value(self.config, 'config_pymultinest', 'init_MPI', 'False', eval)
        self.pm_dump_callback = get_config_value(self.config, 'config_pymultinest', 'dump_callback', 'None', eval)
        self.pm_use_MPI = get_config_value(self.config, 'config_pymultinest', 'use_MPI', 'True', eval)

        # [config_ultranest] (29, u_ prefix for params)
        self.u_resume = get_config_value(self.config, 'config_ultranest', 'resume', 'subfolder', None)
        self.u_run_num = get_config_value(self.config, 'config_ultranest', 'run_num', 'None', eval)
        self.u_wrapped_params = get_config_value(self.config, 'config_ultranest', 'wrapped_params', 'None', eval)
        self.u_num_test_samples = get_config_value(self.config, 'config_ultranest', 'num_test_samples', '2', eval)
        self.u_vectorized = get_config_value(self.config, 'config_ultranest', 'vectorized', 'False', eval)
        self.u_draw_multiple = get_config_value(self.config, 'config_ultranest', 'draw_multiple', 'True', eval)
        self.u_ndraw_min = get_config_value(self.config, 'config_ultranest', 'ndraw_min', '128', eval)
        self.u_ndraw_max = get_config_value(self.config, 'config_ultranest', 'ndraw_max', '65536', eval)
        self.u_num_bootstraps = get_config_value(self.config, 'config_ultranest', 'num_bootstraps', '30', eval)
        self.u_storage_backend = get_config_value(self.config, 'config_ultranest', 'storage_backend', 'hdf5', None)
        self.u_warmstart_max_tau = get_config_value(self.config, 'config_ultranest', 'warmstart_max_tau', '-1', eval)
        # - - - (run params)
        self.u_update_interval_volume_fraction = get_config_value(self.config, 'config_ultranest', 'update_interval_volume_fraction', '0.8', eval)
        self.u_update_interval_ncall = get_config_value(self.config, 'config_ultranest', 'update_interval_ncall', 'None', eval)
        self.u_log_interval = get_config_value(self.config, 'config_ultranest', 'log_interval', 'None', eval)
        self.u_show_status = get_config_value(self.config, 'config_ultranest', 'show_status', 'True', eval)
        self.u_viz_callback = get_config_value(self.config, 'config_ultranest', 'viz_callback', 'auto', None)
        self.u_dlogz = get_config_value(self.config, 'config_ultranest', 'dlogz', '0.5', eval) 
        self.u_dKL = get_config_value(self.config, 'config_ultranest', 'dKL', '0.5', eval)
        self.u_frac_remain = get_config_value(self.config, 'config_ultranest', 'frac_remain', '0.01', eval)
        self.u_Lepsilon = get_config_value(self.config, 'config_ultranest', 'Lepsilon', '0.001', eval)
        self.u_min_ess = get_config_value(self.config, 'config_ultranest', 'min_ess', '400', eval)
        self.u_max_iters = get_config_value(self.config, 'config_ultranest', 'max_iters', 'None', eval)
        self.u_max_ncalls = get_config_value(self.config, 'config_ultranest', 'max_ncalls', 'None', eval)
        self.u_max_num_improvement_loops = get_config_value(self.config, 'config_ultranest', 'max_num_improvement_loops', '-1', eval)
        self.u_cluster_num_live_points = get_config_value(self.config, 'config_ultranest', 'cluster_num_live_points', '40', eval)
        self.u_insertion_test_zscore_threshold = get_config_value(self.config, 'config_ultranest', 'insertion_test_zscore_threshold', '4', eval)
        self.u_insertion_test_window = get_config_value(self.config, 'config_ultranest', 'insertion_test_window', '10', eval)
        self.u_widen_before_initial_plateau_num_warn = get_config_value(self.config, 'config_ultranest', 'widen_before_initial_plateau_num_warn', '10000', eval)
        self.u_widen_before_initial_plateau_num_max = get_config_value(self.config, 'config_ultranest', 'widen_before_initial_plateau_num_max', '50000', eval)


        # - - - - - - - - - - - - - - 


        ## Save CONFIG: - - - - - - -

        # [config_path] (4)
        config['config_path'].comments['observation_path'] = ['# Path to the observed spectrum file']
        config['config_path'].comments['model_path'] = ['', '# Path to the model']
        config['config_path'].comments['adapt_store_path'] = ['', '# Path to store your interpolated grid']
        config['config_path'].comments['result_path'] = ['', '# Path to store your results']
        
        # [config_adapt] (4)
        config.comments['config_adapt'] = ['']
        config['config_adapt'].comments['method'] = ['# Adaptation method. /!\ For safety reasons, this will also be the interpolation method',
                                                     "# Format : 'linear' or 'nearest' or 'zero' or 'slinear' or 'quadratic' or 'cubic' or 'quintic' or 'pchip' or 'barycentric' or 'krogh' or 'akima' or 'makima'",
                                                     "# MOSAIC : No"]
        config['config_adapt'].comments['emulator'] = ['', '# If you want to use an emulator to fit your grid (smooth out the grid).',
                                                     "# Format : 'NA' or 'PCA, ncomp' or 'NMF, ncomp'",
                                                     "# MOSAIC : No"]
        config['config_adapt'].comments['target_res_obs'] = ['', '# Target resolution to reach for the observation(s).', 
                                                             "# Format : float or 'obs' (if you want to keep the original obs resolution)",
                                                             "# MOSAIC : Yes"]
        config['config_adapt'].comments['target_res_mod'] = ['', '# Target resolution to reach for the model.', 
                                                             "# Format : float or 'obs' (if you want to decrease to adapt the model's resolution to the obs's)", 
                                                             " or 'mod' (if you want to keep the model's resolution during inversion)",
                                                             "# MOSAIC : Yes"]
        config['config_adapt'].comments['res_cont'] = ['', '# Resolution used to estimate the continuum.', 
                                                       "# Format : 'NA' or float",
                                                       "# MOSAIC : Yes"]
        config['config_adapt'].comments['wav_cont'] = ['', '# Wavelength range(s) used to estimate the continuum.', 
                                                       "# Format : 'NA' or 'window1_min / window1_max, window2_min / ... / windowN_max'",
                                                       "# MOSAIC : Yes"]
        
        # [config_inversion] (4)
        config.comments['config_inversion'] = ['']
        config['config_inversion'].comments['logL_type'] = ['# Method to calculate the loglikelihood function used in the nested sampling procedure.', 
                                                            "# Format : 'chi2' or 'chi2_covariance' or 'chi2_noisescaling' or 'chi2_noisescaling_covariance' or 'CCF_Brogi'",
                                                            "# or 'CCF_Zucker' or 'CCF_custom'",
                                                            "# MOSAIC : Yes"]
        config['config_inversion'].comments['logL_full'] = ['', '# If you want to use the constant terms in the computation of your loglikelihood.', 
                                                            "# Format : True or False",
                                                            "# MOSAIC : Yes"]
        config['config_inversion'].comments['wav_fit'] = ['', '# Wavelength range(s) used during the nested sampling procedure.', 
                                                          "# Format : 'window1_min / window1_max, window2_min / ... / windowN_max'",
                                                          "# MOSAIC : Yes"]
        config['config_inversion'].comments['ns_algo'] = ['', '# Nested sampling algorithm used.', 
                                                          "# Format : 'nestle' or 'pymultinest' or 'ultranest'",
                                                          "# MOSAIC : No"]
        config['config_inversion'].comments['npoint'] = ['', '# Number of living points during the nested sampling procedure.', 
                                                         "# Format : int",
                                                         "# MOSAIC : No"]
        
        # [config_highcont_models] (2)
        config.comments['config_highcont_models'] = ['']
        config['config_highcont_models'].comments['hc_type'] = ['# Method to compute the high-contrast model.', 
                                                             "# Format : 'NA' or 'nofit_rm_spec' or 'nonlinear_fit_spec' or 'fit_spec' or 'rm_spec' or 'fit_spec_rm_cont' or 'fit_spec_fit_cont'",
                                                             "# MOSAIC : Yes"]
        config['config_highcont_models'].comments['hc_bounds_lsq'] = ['', '# Least-square bounds.', 
                                                             "# Format : 'NA' or 'lower, upper'",
                                                             "# MOSAIC : Yes"]
        
        # [config_parameters] (11)
        config.comments['config_parameters'] = ['']
        config['config_parameters'].comments['par1'] = ['# Definition of the prior function of each parameter explored by the grid. Please refer to the documentation to check', 
                                                        '# the parameter space explore by each grid. Check prior functions for more infos',
                                                        "# Format : 'function', function_param1, function_param2",
                                                        "# MOSAIC : No"]
        config['config_parameters'].comments['r'] = ['', '# Definition of the prior function of each extra-grid parameter. Check prior functions for more infos',
                                                        "# Format : 'function', function_param1, function_param2",
                                                        "# MOSAIC : Yes and No, check the doc !"]
        
        # [config_nestle] (8, n_ prefix for params)
        config.comments['config_nestle'] = ['']
        config['config_nestle'].comments['method'] = ['# Nestle configuration parameters. For more details, please see: http://kylebarbary.com/nestle/index.html', 
                                                            "# Format : _",
                                                            "# MOSAIC : No"]

        # [config_pymultinest] (20, pm_ prefix for params)
        config.comments['config_pymultinest'] = ['']
        config['config_pymultinest'].comments['n_clustering_params'] = ['# Pymultinest configuration parameters. For more details, please see: https://github.com/JohannesBuchner/PyMultiNest/blob/master/pymultinest/run.py', 
                                                            "# Format : _",
                                                            "# MOSAIC : No"]
        
        # [config_ultranest] (29, u_ prefix for params)
        config.comments['config_ultranest'] = ['']
        config['config_ultranest'].comments['resume'] = ['# Ultranest configuration parameters. For more details, please see: https://johannesbuchner.github.io/UltraNest/readme.html', 
                                                            "# Format : _",
                                                            "# MOSAIC : No"]
        


        config.filename = self.result_path + '/config_file_ref.ini'
        config.write()