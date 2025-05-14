from configobj import ConfigObj
from pathlib import Path
import glob
import logging
import os
from ForMoSA.ForMoSAPaths import ForMoSAPaths
from ForMoSA.nested_sampling import Parameter

# log
_log = logging.getLogger(__name__)

# Format logging for this module
if not _log.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)  # Minimal logging level for this module
    formatter = logging.Formatter('%(asctime)s\t%(levelname)8s\t%(message)s')
    formatter.default_msec_format = '%s.%03d'
    handler.setFormatter(formatter)
    _log.addHandler(handler)

_log.setLevel(logging.INFO)
_log.propagate = False 

class ForMoSAError(Exception):
    pass

class GlobalParams(object):
    '''
    Class that import all the parameters from the config file.

    Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
    '''

    def __init__(self, config_file_path):
        # Generate the config object
        config = ConfigObj(config_file_path, encoding='utf8')
        
        formosa_path = ForMoSAPaths(config_file_path)
        
        grid_name = str(formosa_path.model_path).split('/')[-1].split('.nc')[0]
        N_obs = len(formosa_path.observation_files)   
        
        # Basic inits
        self.paths = formosa_path
        self.grid_name = grid_name
        self.n_obs = N_obs
        self._read_info(config)
        
    ##################################################
    # Representation
    ##################################################

    def __repr__(self) -> str:
        return f'<GlobalParams, n_obs={self.n_obs}, config_file_path={self.paths.config_file_path}>'

    def __format__(self) -> str:
        return self.__repr__()
    
    ##################################################
    # Methods
    ##################################################
    
    @staticmethod
    def _get_config_value(config, section, key, default, N_obs, cast=None):
        """
        Helper function to get a config value with a fallback default.

        Args:
            config               (obj): config object
            section              (str): config section name
            key                  (str): config key name
            default                   : default value if key is missing
            N_obs                (bool): Number of obs to invert. If set to 0, the parameter cannot be set with MOSAIC
            cast                      : a function to cast the value (e.g., int, list, eval, etc.)

        Returns:
            The value (possibly cast), and stores it back into global_params.config if it was missing.

        Author: Matthieu Ravet
        """
        # Ensure section exists
        if section not in config:
            config[section] = {}
        try:
            val = config[section][key]

        # Add key to section
        except:
            if N_obs > 0:
                val = [default for _ in range(N_obs)]
            else:
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
    
    
    @staticmethod 
    def _process_multi_obs_parameter(param_name, param_values, N_obs, step):
        '''
        Treats a multi-observation parameter (alpha, rv, vsini, ld).
    
        Args:
            param_name    (str): Name of the parameter ('alpha', 'rv', 'vsini', 'ld').
            param_values (list): List of values (['uniform', '0', '100', ...]).
            N_obs         (int): Number of obs to invert. If set to 0, the parameter cannot be set with MOSAIC
            step          (int): Step between blocs in param_values (3 for [prior, min, max], 4 for [prior, min, max, vsini_function]).

        Returns:
            name          (str): Name of the parameter accounting for different values for different observations (e.g. 'rv_0', 'rv_1')
            parameter    (list): List of parameters where each element is an instance of :class:`~ForMoSA.Parameter
            
        Authors: Allan Denis
        '''


        parameter = []
        name = []
        vsini_function = None
        
        if len(param_values) > step:   # Multiple observations
            for indobs in range(N_obs):
                prior_type = param_values[indobs * step]
                if prior_type != 'NA':
                    name.append(f"{param_name}_{indobs}")
                    if prior_type in {'uniform', 'log-uniform'}:
                        bounds, mean, std, value = [float(param_values[indobs * step + 1]), float(param_values[indobs * step + 2])], None, None, None
                    elif prior_type == 'gaussian':
                        bounds, mean, std, value = None, float(param_values[indobs * step + 1]), float(param_values[indobs * step + 2]), None
                    elif prior_type == 'constant':
                        bounds, mean, std, value = None, None, None, float(param_values[indobs * step + 1])
                    if param_name == 'vsini':
                        vsini_function = str(param_values[indobs * step + 3])
                    parameter.append(Parameter(name=name[-1], prior=prior_type, bounds=bounds, mean=mean, std=std, value=value, vsini_function = vsini_function)
)
        else:
            prior_type = param_values[0]
            if prior_type != 'NA':
                name.append(param_name)
                if prior_type in {'uniform', 'log-uniform'}:
                    bounds, mean, std, value = [float(param_values[1]), float(param_values[2])], None, None, None
                elif prior_type == 'gaussian':
                    bounds, mean, std, value = float(param_values[1]), float(param_values[2]), None, None
                elif prior_type == 'constant':
                    bounds, mean, std, value = None, None, None, float(param_values[1])
                if param_name == 'vsini':
                    vsini_function = str(param_values[3])
                parameter.append(Parameter(name=name[-1], prior=prior_type, bounds=bounds, mean=mean, std=std, value=value, vsini_function=vsini_function))

        return name, parameter

    
    
    def _read_info(self, config):
        '''
        Read config file information. Put default values if no value is assigned for a parameter

        Returns
        -------
        global_params : 
            
        Authors; Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet and Allan Denis 
        '''
        # [config_adapt] (5)
        method = self._get_config_value(config, 'config_adapt', 'method', 'linear', 0, None)
        emulator = self._get_config_value(config, 'config_adapt', 'emulator', 'NA', 0, list)
        target_res_obs = self._get_config_value(config, 'config_adapt', 'target_res_obs', 'obs', self.n_obs, list)
        target_res_mod = self._get_config_value(config, 'config_adapt', 'target_res_mod', 'obs', self.n_obs, list)
        res_cont = self._get_config_value(config, 'config_adapt', 'res_cont', 'NA', self.n_obs, list)
        wav_cont = self._get_config_value(config, 'config_adapt', 'wav_cont', 'NA', self.n_obs, list)
        self.config_adapt = {'method': method, 'emulator': emulator, 'target_res_obs': target_res_obs, 'target_res_mod': target_res_mod, 'res_cont': res_cont, 'wav_cont': wav_cont}

        # [config_inversion] (4)
        logL_type = self._get_config_value(config, 'config_inversion', 'logL_type', 'chi2', self.n_obs, list)
        wav_fit = self._get_config_value(config, 'config_inversion', 'wav_fit', '0,100', self.n_obs, list)
        ns_algo = self._get_config_value(config, 'config_inversion', 'ns_algo', 'nestle', 0, None)
        npoints = self._get_config_value(config, 'config_inversion', 'npoint', '100', 0, eval)
        self.config_inversion = {'logL_type': logL_type, 'wav_fit': wav_fit, 'ns_algo': ns_algo, 'npoints': npoints}

        # [config_highcont_models] (2)
        hc_type = self._get_config_value(config, 'config_highcont_models', 'hc_type', 'NA', self.n_obs, list)
        hc_bounds_lsq = self._get_config_value(config, 'config_highcont_models', 'hc_bounds_lsq', 'NA', self.n_obs, list)
        self.config_highcont_models = {'hc_type': hc_type, 'hc_bounds_lsq': hc_bounds_lsq}
    
        # [config_parameters] (1)
        grid_parameters = {}        # Refers to the grid parameters (Teff, logg, ...)
        physical_parameters = {}    # Refers to the other parameters (rv, vsini, ...)
        
        for name, raw_value in config['config_parameters'].items():
            if (name == 'rv') or (name == 'vsini') or (name == 'alpha') or (name == 'ls'):
                values = self._get_config_value(config, 'config_parameters', name, 'NA', self.n_obs, list)
            else:
                values = self._get_config_value(config, 'config_parameters', name, 'NA', 0, list)
        
            try:
                if name == 'vsini':
                    name_list, param_list = self._process_multi_obs_parameter(name, values, self.n_obs, 4)
                else:
                    name_list, param_list = self._process_multi_obs_parameter(name, values, self.n_obs, 3)
            
                # Separation by name
                if name.lower().startswith("par") and name[3:].isdigit():
                    if param_list != []:
                        for param_i, name_i in list(zip(param_list, name_list)):
                            grid_parameters[name_i] = param_i
                else:
                    if param_list != []:
                        for param_i, name_i in zip(param_list, name_list):
                            physical_parameters[name_i] = param_i
                            
            except ForMoSAError as e:
                raise e
                self._logger.critical(e)
                
        self.config_parameters = {'grid_parameters': grid_parameters, 'physical_parameters': physical_parameters}

        # [config_nestle] (8) 
        method = self._get_config_value(config, 'config_nestle', 'method', 'single', 0, None)
        update_interval = self._get_config_value(config, 'config_nestle', 'update_interval', 'None', 0, eval)
        npdim = self._get_config_value(config, 'config_nestle', 'npdim', 'None', 0, eval)
        maxiter = self._get_config_value(config, 'config_nestle', 'maxiter', 'None', 0, eval)
        maxcall = self._get_config_value(config, 'config_nestle', 'maxcall', 'None', 0, eval)
        dlogz = self._get_config_value(config, 'config_nestle', 'dlogz', 'None', 0, eval)
        decline_factor = self._get_config_value(config, 'config_nestle', 'decline_factor', 'None', 0, eval)
        rstate = self._get_config_value(config, 'config_nestle', 'rstate', 'None', 0, eval)
        self.config_nestle = {'method': method, 'update_interval': update_interval, 'npdim': npdim, 'maxiter': maxiter, 'maxcall': maxcall, 'dlogz': dlogz, 'decline_factor': decline_factor, 'rstate': rstate}

        # [config_pymultinest] (20, pm_ prefix for params)
        clustering_params = self._get_config_value(config, 'config_pymultinest', 'n_clustering_params', 'None', 0, eval)
        wrapped_params = self._get_config_value(config, 'config_pymultinest', 'wrapped_params', 'None', 0, eval)
        importance_nested_sampling = self._get_config_value(config, 'config_pymultinest', 'importance_nested_sampling', 'True', 0, eval)
        multimodal = self._get_config_value(config, 'config_pymultinest', 'multimodal', 'True', 0, eval)
        const_efficiency_mode = self._get_config_value(config, 'config_pymultinest', 'const_efficiency_mode', 'False', 0, eval)
        evidence_tolerance = self._get_config_value(config, 'config_pymultinest', 'evidence_tolerance', '0.5', 0, eval)
        sampling_efficiency = self._get_config_value(config, 'config_pymultinest', 'sampling_efficiency', '0.8', 0, eval)
        n_iter_before_update = self._get_config_value(config, 'config_pymultinest', 'n_iter_before_update', '100', 0, eval)
        null_log_evidence = self._get_config_value(config, 'config_pymultinest', 'null_log_evidence', '-1e90', 0, eval)
        max_modes = self._get_config_value(config, 'config_pymultinest', 'max_modes', '100', 0, eval)
        mode_tolerance = self._get_config_value(config, 'config_pymultinest', 'mode_tolerance', '-1e90', 0, eval)
        seed = self._get_config_value(config, 'config_pymultinest', 'seed', '-1', 0, eval)
        verbose = self._get_config_value(config, 'config_pymultinest', 'verbose', 'True', 0, eval)
        resume = self._get_config_value(config, 'config_pymultinest', 'resume', 'False', 0, eval) # This is the only parameter not set by default to True, you can change it if your inversion crash and you don't want to start anew
        scontext = self._get_config_value(config, 'config_pymultinest', 'context', '0', 0, eval)
        log_zero = self._get_config_value(config, 'config_pymultinest', 'log_zero', '-1e100', 0, eval)
        max_iter = self._get_config_value(config, 'config_pymultinest', 'max_iter', '0', 0, eval) # Unlimited
        init_MPI = self._get_config_value(config, 'config_pymultinest', 'init_MPI', 'False', 0, eval)
        dump_callback = self._get_config_value(config, 'config_pymultinest', 'dump_callback', 'None', 0, eval)
        use_MPI = self._get_config_value(config, 'config_pymultinest', 'use_MPI', 'True', 0, eval)
        self.config_pymultinest = {'clustering_params': clustering_params, 'wrapped_params': wrapped_params, 'importance_nested_sampling': importance_nested_sampling, 'multimodal': multimodal, 
                                   'const_efficiency_mode': const_efficiency_mode, 'evidence_tolerance': evidence_tolerance, 'sampling_efficiency': sampling_efficiency, 'n_iter_before_update': n_iter_before_update,
                                   'null_log_evidence': null_log_evidence, 'max_modes': max_modes, 'mode_tolerance': mode_tolerance, 'seed': seed, 'verbose': verbose, 'resume': resume, 'scontext': scontext,
                                   'log_zero': log_zero, 'max_iter': max_iter, 'init_MPI': init_MPI, 'dump_callback': dump_callback, 'use_MPI': use_MPI}

        # [config_ultranest] (29)
        resume = self._get_config_value(config, 'config_ultranest', 'resume', 'subfolder', 0, None)
        run_num = self._get_config_value(config, 'config_ultranest', 'run_num', 'None', 0, eval)
        wrapped_params = self._get_config_value(config, 'config_ultranest', 'wrapped_params', 'None', 0, eval)
        num_test_samples = self._get_config_value(config, 'config_ultranest', 'num_test_samples', '2', 0, eval)
        vectorized = self._get_config_value(config, 'config_ultranest', 'vectorized', 'False', 0, eval)
        draw_multiple = self._get_config_value(config, 'config_ultranest', 'draw_multiple', 'True', 0, eval)
        ndraw_min = self._get_config_value(config, 'config_ultranest', 'ndraw_min', '128', 0, eval)
        ndraw_max = self._get_config_value(config, 'config_ultranest', 'ndraw_max', '65536', 0, eval)
        num_bootstraps = self._get_config_value(config, 'config_ultranest', 'num_bootstraps', '30', 0, eval)
        storage_backend = self._get_config_value(config, 'config_ultranest', 'storage_backend', 'hdf5', 0, None)
        warmstart_max_tau = self._get_config_value(config, 'config_ultranest', 'warmstart_max_tau', '-1', 0, eval)
        # - - - (run params)
        update_interval_volume_fraction = self._get_config_value(config, 'config_ultranest', 'update_interval_volume_fraction', '0.8', 0, eval)
        update_interval_ncall = self._get_config_value(config, 'config_ultranest', 'update_interval_ncall', 'None', 0, eval)
        log_interval = self._get_config_value(config, 'config_ultranest', 'log_interval', 'None', 0, eval)
        show_status = self._get_config_value(config, 'config_ultranest', 'show_status', 'True', 0, eval)
        viz_callback = self._get_config_value(config, 'config_ultranest', 'viz_callback', 'auto', 0, None)
        dlogz = self._get_config_value(config, 'config_ultranest', 'dlogz', '0.5', 0, eval) 
        dKL = self._get_config_value(config, 'config_ultranest', 'dKL', '0.5', 0, eval)
        frac_remain = self._get_config_value(config, 'config_ultranest', 'frac_remain', '0.01', 0, eval)
        Lepsilon = self._get_config_value(config, 'config_ultranest', 'Lepsilon', '0.001', 0, eval)
        min_ess = self._get_config_value(config, 'config_ultranest', 'min_ess', '400', 0, eval)
        max_iters = self._get_config_value(config, 'config_ultranest', 'max_iters', 'None', 0, eval)
        max_ncalls = self._get_config_value(config, 'config_ultranest', 'max_ncalls', 'None', 0, eval)
        max_num_improvement_loops = self._get_config_value(config, 'config_ultranest', 'max_num_improvement_loops', '-1', 0, eval)
        cluster_num_live_points = self._get_config_value(config, 'config_ultranest', 'cluster_num_live_points', '40', 0, eval)
        insertion_test_zscore_threshold = self._get_config_value(config, 'config_ultranest', 'insertion_test_zscore_threshold', '4', 0, eval)
        insertion_test_window = self._get_config_value(config, 'config_ultranest', 'insertion_test_window', '10', 0, eval)
        widen_before_initial_plateau_num_warn = self._get_config_value(config, 'config_ultranest', 'widen_before_initial_plateau_num_warn', '10000', 0, eval)
        widen_before_initial_plateau_num_max = self._get_config_value(config, 'config_ultranest', 'widen_before_initial_plateau_num_max', '50000', 0, eval)
        self.config_ultranest = {'resume': resume, 'run_num': run_num, 'wrapped_params': wrapped_params, 'num_test_samples': num_test_samples, 'vectorized': vectorized, 'draw_multiple': draw_multiple, 'ndraw_min': ndraw_min,
                                 'ndraw_max': ndraw_max, 'num_bootstraps': num_bootstraps, 'storage_backend': storage_backend, 'warmstart_max_tau': warmstart_max_tau, 'update_interval_volume_fraction': update_interval_volume_fraction,
                                 'update_interval_ncall': update_interval_ncall, 'log_interval': log_interval, 'show_status': show_status, 'viz_callback': viz_callback, 'dlogz': dlogz, 'dKL': dKL, 'frac_remain': frac_remain,
                                 'Lepsilon': Lepsilon, 'min_ess': min_ess, 'max_iters': max_iters, 'max_ncalls': max_ncalls, 'max_num_improvement_loops': max_num_improvement_loops, 'cluster_num_live_points': cluster_num_live_points, 
                                 'intertion_test_zscore_threshold': insertion_test_zscore_threshold, 'insertion_test_window': insertion_test_window, 'widen_before_initial_plateau_num_warn': widen_before_initial_plateau_num_warn, 'widen_before_initial_plateau_num_max': widen_before_initial_plateau_num_max}
        

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

        # [config_pymultinest] (20, pm_ prefix for params)
        config.comments['config_pymultinest'] = ['']
        config['config_pymultinest'].comments['n_clustering_params'] = ['# Pymultinest configuration parameters. For more details, please see: https://github.com/JohannesBuchner/PyMultiNest/blob/master/pymultinest/run.py', 
                                                            "# Format : _",
                                                            "# MOSAIC : No"]
        
        # [config_ultranest] (29, u_ prefix for params)
        config.comments['config_ultranest'] = ['']
        config['config_ultranest'].comments['resume'] = ['# Ultranest configuration parameters. For more details, please see: https://johannesbuchner.github.io/UltraNest/readme.html', 
                                                            "# Format : _",
                                                            "# MOSAIC : No"]
        
        config.filename = self.paths.result_path / 'config_file_ref.ini'
        config.write()
      