from configobj import ConfigObj
from pathlib import Path
import glob
import logging
import os
from ForMoSA.ForMoSAPaths import ForMoSAPaths
from ForMoSA.NestedSampling_Parameters import Parameter
import numpy as np


class ForMoSAError(Exception):
    pass

class GlobalParams(object):
    '''
    Class that import all the parameters from the config file.

    Parameters
    ----------
    config_file_path (str): Path of the config file
    log_level        (str): Level of the logging

    Authors: Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
    '''

    def __init__(self, config_file_path: str | os.PathLike, log_level: str = 'info'):
        # Generate the config object
        config = ConfigObj(config_file_path, encoding='utf8')

        formosa_path = ForMoSAPaths(config_file_path, log_level=log_level)

        # Basic inits
        self.config = config
        self.paths = formosa_path
        self.n_obs = formosa_path.observation.n_obs
        self.instrument_files = self.paths.observation.instrument_files
        self._read_info()

    ##################################################
    # Representation
    ##################################################

    def __repr__(self) -> str:
        return f'<GlobalParams, config_file_path={self.paths.config_file_path}>'

    def __format__(self) -> str:
        return self.__repr__()


    ##################################################
    # Methods
    ##################################################

    @staticmethod
    def _get_config_value(config: ConfigObj, section: str, key: str, default, n_obs: int = 1, cast=None, instrument_files: dict = dict()):
        """
        Helper function to get a config value with a fallback default.

        Args:
            config               (obj): Config object
            section              (str): Config section name
            key                  (str): Config key name
            default                   : Default value if key is missing
            n_obs                (bool): Number of obs to invert.
            cast                       : A function to cast the value (e.g., int, list, eval, etc.)
            instrument_files    (dict): Dictionary containing the instruments for each observation_file {obs_file_number: [instruments]}
            step                 (int): Step between blocs. Used for the grid and extra-grid parameters. (4 for vsini and 3 for the other parameters)

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
            if n_obs > 1:
                val = [default for _ in range(n_obs)]
            else:
                val = default
            config[section][key] = val

        if cast:
            try:
                if isinstance(val, list):
                    if cast != list:
                        raise ForMoSAError(f" Error is cast function. {val} is a list so the cast function should be 'list'.")
                    else:
                        if len(val) != len(instrument_files) and len(val) != n_obs and val != ['NA']:
                            raise ForMoSAError(f" Error in config.{section}[{key}]. This parameter contains {len(val)} elements while it should contain a number of elements amongst {np.unique([len(instrument_files), n_obs])}.")
                        # Special case where the user filled one parameter for each observation file but at least one observation file contains more than one instrument
                        if len(val) == len(instrument_files) and len(val) != n_obs:
                            index = 0
                            for i in range(len(instrument_files)):
                                nb_instrument_per_obs = len(instrument_files[i])
                                # If obs contains only one instrument, we don't do anything
                                # but if obs contains at least 2 instruments, we repeat
                                index += i
                                for nb_ins in range(nb_instrument_per_obs - 1):
                                    val.insert(index, val[index])
                                index += nb_instrument_per_obs - 1
                elif cast == list:
                    val = [val]
                else:
                    val = cast(val)

            except ForMoSAError as e:
                raise(f' Error {e}')


        return val


    @staticmethod
    def _process_multi_obs_parameter(config, section, param_name, n_obs, instrument_files):
        '''
        Treats a multi-observation parameter (alpha, rv, vsini, ld).

        Args:
            param_name    (str): Name of the parameter ('alpha', 'rv', 'vsini', 'ld').
            param_values (list): List of values (['uniform', '0', '100', ...]).
            N_obs         (int): Number of obs to invert. If set to 0, the parameter cannot be set with MOSAIC

        Returns:
            name          (str): Name of the parameter accounting for different values for different observations (e.g. 'rv_0', 'rv_1')
            parameter    (list): List of parameters where each element is an instance of :class:`~ForMoSA.Parameter

        Authors: Allan Denis
        '''

        # Ensure section exists
        if section not in config:
            config[section] = {}
        try:
            param_values = config[section][param_name]

        # Add key to section
        except:
            param_values = ['NA']
            config[section][param_name] = param_values

        if not(isinstance(param_values, list)):
            param_values = [param_values]

        parameters = []
        names = []

        offset = 0
        obs_index = 0
        total_blocks = 0

        # First parsing to detect the number of blocs of observations for the parameter
        # We also update param_values if for one or more observation files, we have more than one instrument
        while offset < len(param_values) and total_blocks < n_obs:
            prior = param_values[offset]
            if prior == 'NA':
                offset += 1
            elif prior == 'constant':
                offset += 2 + (1 if param_name == 'vsini' else 0) # If vsini we have a vsini function so we need to att 1 to the offset
            else:
                offset += 3  + (1 if param_name == 'vsini' else 0) # For the same reason we add 1 if vsini
            total_blocks += 1

        # Restart the parsing now that we have the total number of blocs
        offset = 0
        obs_index = 0

        while offset < len(param_values) and obs_index < n_obs:
            prior_type = param_values[offset]

            if prior_type == 'NA':
                offset += 1
                obs_index += 1
                continue

            is_single_block = total_blocks == 1
            # If we have only one block of observation we keep the name of the parameter else we rename it param_name_{obs_index}
            param_name_obs = param_name if is_single_block else f"{param_name}_{obs_index}"
            names.append(param_name_obs)

            bounds = mean = std = value = vsini_function = None
            local_step = 0
            if prior_type in {'uniform', 'log-uniform'}:
                bounds = [float(param_values[offset + 1]), float(param_values[offset + 2])]
                local_step = 3
            elif prior_type == 'gaussian':
                mean = float(param_values[offset + 1])
                std = float(param_values[offset + 2])
                local_step = 3
            elif prior_type == 'constant':
                value = float(param_values[offset + 1])
                local_step = 2

            if param_name == 'vsini' and offset + local_step < len(param_values):
                vsini_function = str(param_values[offset + local_step])
                local_step += 1

            parameters.append(Parameter(
                name=param_name_obs,
                prior=prior_type,
                bounds=bounds,
                mean=mean,
                std=std,
                value=value,
                vsini_function=vsini_function
            ))

            offset += local_step
            obs_index += 1

            for i in range(len(instrument_files[obs_index-1])-1):
                param_name_obs = f"{param_name}_{obs_index}"
                parameters.append(Parameter(
                    name=param_name_obs,
                    prior=prior_type,
                    bounds=bounds,
                    mean=mean,
                    std=std,
                    value=value,
                    vsini_function=vsini_function
                ))
                obs_index += 1

        return names, parameters


    def _read_info(self):
        '''
        Read config file information. Put default values if no value is assigned for a parameter

        Returns
        -------
        global_params :

        Authors; Simon Petrus, Paulina Palma-Bifani, Matthieu Ravet and Allan Denis
        '''

        config = self.config

        # [config_adapt] (5)
        method = self._get_config_value(config, 'config_adapt', 'method', 'linear')
        emulator = self._get_config_value(config, 'config_adapt', 'emulator', 'NA', 1, list)
        target_res_obs = self._get_config_value(config, 'config_adapt', 'target_res_obs', 'obs', self.n_obs, list, self.instrument_files)
        target_res_mod = self._get_config_value(config, 'config_adapt', 'target_res_mod', 'obs', self.n_obs, list, self.instrument_files)
        res_cont = self._get_config_value(config, 'config_adapt', 'res_cont', 'NA', self.n_obs, list, self.instrument_files)
        wav_cont = self._get_config_value(config, 'config_adapt', 'wav_cont', 'NA', self.n_obs, list, self.instrument_files)
        adapt = {'method': method, 'emulator': emulator, 'target_res_obs': target_res_obs, 'target_res_mod': target_res_mod, 'res_cont': res_cont, 'wav_cont': wav_cont}

        # [config_inversion] (4)
        logL_type = self._get_config_value(config, 'config_inversion', 'logL_type', 'chi2', self.n_obs, list, self.instrument_files)
        wav_fit = self._get_config_value(config, 'config_inversion', 'wav_fit', '0,100', self.n_obs, list, self.instrument_files)
        ns_algo = self._get_config_value(config, 'config_inversion', 'ns_algo', 'nestle')
        npoints = self._get_config_value(config, 'config_inversion', 'npoint', '100', 1, eval)
        hc_type = self._get_config_value(config, 'config_inversion', 'hc_type', 'NA', self.n_obs, list, self.instrument_files)
        hc_lower_bounds_lsq = self._get_config_value(config, 'config_inversion', 'hc_lower_bounds_lsq', 'NA', self.n_obs, list, self.instrument_files)
        hc_higher_bounds_lsq = self._get_config_value(config, 'config_inversion', 'hc_higher_bounds_lsq', 'NA', self.n_obs, list, self.instrument_files)
        hc_bounds_lsq = [(low_bound, high_bound) for low_bound, high_bound in zip(hc_lower_bounds_lsq, hc_higher_bounds_lsq)]
        inversion = {'logL_type': logL_type, 'wav_fit': wav_fit, 'ns_algo': ns_algo, 'npoints': npoints, 'hc_type': hc_type, 'hc_bounds_lsq': hc_bounds_lsq}

        # [config_parameters] (1)
        grid_parameters = {}        # Refers to the grid parameters (Teff, logg, ...)
        physical_parameters = {}    # Refers to the other parameters (rv, vsini, ...)

        for name in config['config_parameters']:
            name_list, param_list = self._process_multi_obs_parameter(config, 'config_parameters', name, self.n_obs, self.instrument_files)

            # Separation by name
            if name.lower().startswith("par") and name[3:].isdigit():
                if param_list != []:
                    for param_i, name_i in list(zip(param_list, name_list)):
                        grid_parameters[name_i] = param_i
            else:
                if param_list != []:
                    for param_i, name_i in zip(param_list, name_list):
                        physical_parameters[name_i] = param_i

        parameters = {'grid_parameters': grid_parameters, 'physical_parameters': physical_parameters}

        # [config_nestle] (8)
        method = self._get_config_value(config, 'config_nestle', 'method', 'single')
        update_interval = self._get_config_value(config, 'config_nestle', 'update_interval', 'None', 1, eval)
        npdim = self._get_config_value(config, 'config_nestle', 'npdim', 'None', 1, eval)
        maxiter = self._get_config_value(config, 'config_nestle', 'maxiter', 'None', 1, eval)
        maxcall = self._get_config_value(config, 'config_nestle', 'maxcall', 'None', 1, eval)
        dlogz = self._get_config_value(config, 'config_nestle', 'dlogz', 'None', 1, eval)
        decline_factor = self._get_config_value(config, 'config_nestle', 'decline_factor', 'None', 1, eval)
        rstate = self._get_config_value(config, 'config_nestle', 'rstate', 'None', 1, eval)
        config_nestle = {'method': method, 'update_interval': update_interval, 'npdim': npdim, 'maxiter': maxiter, 'maxcall': maxcall, 'dlogz': dlogz, 'decline_factor': decline_factor, 'rstate': rstate}

        # [config_pymultinest] (20, pm_ prefix for params)
        n_clustering_params = self._get_config_value(config, 'config_pymultinest', 'n_clustering_params', 'None', 1, eval)
        wrapped_params = self._get_config_value(config, 'config_pymultinest', 'wrapped_params', 'None', 1, eval)
        importance_nested_sampling = self._get_config_value(config, 'config_pymultinest', 'importance_nested_sampling', 'True', 1, eval)
        multimodal = self._get_config_value(config, 'config_pymultinest', 'multimodal', 'True', 1, eval)
        const_efficiency_mode = self._get_config_value(config, 'config_pymultinest', 'const_efficiency_mode', 'False', 1, eval)
        evidence_tolerance = self._get_config_value(config, 'config_pymultinest', 'evidence_tolerance', '0.5', 1, eval)
        sampling_efficiency = self._get_config_value(config, 'config_pymultinest', 'sampling_efficiency', '0.8', 1, eval)
        n_iter_before_update = self._get_config_value(config, 'config_pymultinest', 'n_iter_before_update', '100', 1, eval)
        null_log_evidence = self._get_config_value(config, 'config_pymultinest', 'null_log_evidence', '-1e90', 1, eval)
        max_modes = self._get_config_value(config, 'config_pymultinest', 'max_modes', '100', 1, eval)
        mode_tolerance = self._get_config_value(config, 'config_pymultinest', 'mode_tolerance', '-1e90', 1, eval)
        seed = self._get_config_value(config, 'config_pymultinest', 'seed', '-1', 1, eval)
        verbose = self._get_config_value(config, 'config_pymultinest', 'verbose', 'True', 1, eval)
        resume = self._get_config_value(config, 'config_pymultinest', 'resume', 'False', 1, eval) # This is the only parameter not set by default to True, you can change it if your inversion crash and you don't want to start anew
        context = self._get_config_value(config, 'config_pymultinest', 'context', '0', 1, eval)
        log_zero = self._get_config_value(config, 'config_pymultinest', 'log_zero', '-1e100', 1, eval)
        max_iter = self._get_config_value(config, 'config_pymultinest', 'max_iter', '0', 1, eval) # Unlimited
        init_MPI = self._get_config_value(config, 'config_pymultinest', 'init_MPI', 'False', 1, eval)
        dump_callback = self._get_config_value(config, 'config_pymultinest', 'dump_callback', 'None', 1, eval)
        use_MPI = self._get_config_value(config, 'config_pymultinest', 'use_MPI', 'True', 1, eval)
        config_pymultinest = {'n_clustering_params': n_clustering_params, 'wrapped_params': wrapped_params, 'importance_nested_sampling': importance_nested_sampling, 'multimodal': multimodal,
                                   'const_efficiency_mode': const_efficiency_mode, 'evidence_tolerance': evidence_tolerance, 'sampling_efficiency': sampling_efficiency, 'n_iter_before_update': n_iter_before_update,
                                   'null_log_evidence': null_log_evidence, 'max_modes': max_modes, 'mode_tolerance': mode_tolerance, 'seed': seed, 'verbose': verbose, 'resume': resume, 'context': context,
                                   'log_zero': log_zero, 'max_iter': max_iter, 'init_MPI': init_MPI, 'dump_callback': dump_callback, 'use_MPI': use_MPI}

        # [config_ultranest] (29)
        resume = self._get_config_value(config, 'config_ultranest', 'resume', 'subfolder')
        run_num = self._get_config_value(config, 'config_ultranest', 'run_num', 'None', 1, eval)
        wrapped_params = self._get_config_value(config, 'config_ultranest', 'wrapped_params', 'None', 1, eval)
        num_test_samples = self._get_config_value(config, 'config_ultranest', 'num_test_samples', '2', 1, eval)
        vectorized = self._get_config_value(config, 'config_ultranest', 'vectorized', 'False', 1, eval)
        draw_multiple = self._get_config_value(config, 'config_ultranest', 'draw_multiple', 'True', 1, eval)
        ndraw_min = self._get_config_value(config, 'config_ultranest', 'ndraw_min', '128', 1, eval)
        ndraw_max = self._get_config_value(config, 'config_ultranest', 'ndraw_max', '65536', 1, eval)
        num_bootstraps = self._get_config_value(config, 'config_ultranest', 'num_bootstraps', '30', 1, eval)
        storage_backend = self._get_config_value(config, 'config_ultranest', 'storage_backend', 'hdf5', 1, None)
        warmstart_max_tau = self._get_config_value(config, 'config_ultranest', 'warmstart_max_tau', '-1', 1, eval)
        # - - - (run params)
        update_interval_volume_fraction = self._get_config_value(config, 'config_ultranest', 'update_interval_volume_fraction', '0.8', 1, eval)
        update_interval_ncall = self._get_config_value(config, 'config_ultranest', 'update_interval_ncall', 'None', 1, eval)
        log_interval = self._get_config_value(config, 'config_ultranest', 'log_interval', 'None', 1, eval)
        show_status = self._get_config_value(config, 'config_ultranest', 'show_status', 'True', 1, eval)
        viz_callback = self._get_config_value(config, 'config_ultranest', 'viz_callback', 'auto', 1, None)
        dlogz = self._get_config_value(config, 'config_ultranest', 'dlogz', '0.5', 1, eval)
        dKL = self._get_config_value(config, 'config_ultranest', 'dKL', '0.5', 1, eval)
        frac_remain = self._get_config_value(config, 'config_ultranest', 'frac_remain', '0.01', 1, eval)
        Lepsilon = self._get_config_value(config, 'config_ultranest', 'Lepsilon', '0.001', 1, eval)
        min_ess = self._get_config_value(config, 'config_ultranest', 'min_ess', '400', 1, eval)
        max_iters = self._get_config_value(config, 'config_ultranest', 'max_iters', 'None', 1, eval)
        max_ncalls = self._get_config_value(config, 'config_ultranest', 'max_ncalls', 'None', 1, eval)
        max_num_improvement_loops = self._get_config_value(config, 'config_ultranest', 'max_num_improvement_loops', '-1', 1, eval)
        cluster_num_live_points = self._get_config_value(config, 'config_ultranest', 'cluster_num_live_points', '40', 1, eval)
        insertion_test_zscore_threshold = self._get_config_value(config, 'config_ultranest', 'insertion_test_zscore_threshold', '4', 1, eval)
        insertion_test_window = self._get_config_value(config, 'config_ultranest', 'insertion_test_window', '10', 1, eval)
        widen_before_initial_plateau_num_warn = self._get_config_value(config, 'config_ultranest', 'widen_before_initial_plateau_num_warn', '10000', 1, eval)
        widen_before_initial_plateau_num_max = self._get_config_value(config, 'config_ultranest', 'widen_before_initial_plateau_num_max', '50000', 1, eval)
        config_ultranest = {'resume': resume, 'run_num': run_num, 'wrapped_params': wrapped_params, 'num_test_samples': num_test_samples, 'vectorized': vectorized, 'draw_multiple': draw_multiple, 'ndraw_min': ndraw_min,
                                 'ndraw_max': ndraw_max, 'num_bootstraps': num_bootstraps, 'storage_backend': storage_backend, 'warmstart_max_tau': warmstart_max_tau, 'update_interval_volume_fraction': update_interval_volume_fraction,
                                 'update_interval_ncall': update_interval_ncall, 'log_interval': log_interval, 'show_status': show_status, 'viz_callback': viz_callback, 'dlogz': dlogz, 'dKL': dKL, 'frac_remain': frac_remain,
                                 'Lepsilon': Lepsilon, 'min_ess': min_ess, 'max_iters': max_iters, 'max_ncalls': max_ncalls, 'max_num_improvement_loops': max_num_improvement_loops, 'cluster_num_live_points': cluster_num_live_points,
                                 'insertion_test_zscore_threshold': insertion_test_zscore_threshold, 'insertion_test_window': insertion_test_window, 'widen_before_initial_plateau_num_warn': widen_before_initial_plateau_num_warn, 'widen_before_initial_plateau_num_max': widen_before_initial_plateau_num_max}

        ns_algo = {'nestle': config_nestle, 'ultranest': config_ultranest, 'pymultinest': config_pymultinest}


        # config plottings
        color = self._get_config_value(config, 'config_plottings', 'color', 'NA', self.n_obs, list, self.instrument_files)
        edgecolor = self._get_config_value(config, 'config_plottings', 'edgecolor', 'NA', self.n_obs, list, self.instrument_files)
        marker = self._get_config_value(config, 'config_plottings', 'marker', 'NA', self.n_obs, list, self.instrument_files)
        size = self._get_config_value(config, 'config_plottings', 'size', 'NA', self.n_obs, list, self.instrument_files)

        plottings = {'color': color, 'edgecolor': edgecolor, 'marker': marker, 'size': size}


        self.config_params = {'adapt': adapt, 'inversion': inversion, 'parameters': parameters, 'ns_algo': ns_algo, 'plottings': plottings}

        ## Save CONFIG: - - - - - - -

        # [config_path] (4)
        config['config_path'].comments['observation_path'] = ['# Path to the observed spectrum file']
        config['config_path'].comments['model_path'] = ['', '# Path to the model']
        config['config_path'].comments['adapt_store_path'] = ['', '# Path to store your interpolated grid']
        config['config_path'].comments['result_path'] = ['', '# Path to store your results']

        # [config_adapt] (6)
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

        # [config_inversion] (6)
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

        config['config_inversion'].comments['hc_type'] = ['# Method to compute the high-contrast model.',
                                                             "# Format : 'NA' or 'nofit_rm_spec' or 'nonlinear_fit_spec' or 'fit_spec' or 'rm_spec' or 'fit_spec_rm_cont' or 'fit_spec_fit_cont'",
                                                             "# MOSAIC : Yes"]
        config['config_inversion'].comments['hc_bounds_lsq'] = ['', '# Least-square bounds.',
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

        # [config_plottings] (4)
        config.comments['config_plottings'] = ['']
        config['config_plottings'].comments['color'] = ['# Plottings colors',
                                                            "# Format: 'NA' or str",
                                                            "# MOSAIC : Yes"]
        config['config_plottings'].comments['edgecolor'] = ['# Plotting edgecolors',
                                                             "# Format: 'NA' or str",
                                                             "# MOSAIC : Yes"]
        config['config_plottings'].comments['marker'] = ['# Plotting edgecolors',
                                                             "# Format: 'NA' or str",
                                                             "# MOSAIC : Yes"]
        config['config_plottings'].comments['size'] = ['# Plotting size',
                                                             "# Format: 'NA' or float",
                                                             "# MOSAIC : Yes"]

        self._save_config_file()


    def _save_config_file(self, path: str | os.PathLike = None, name: str = 'NA') -> None:
        '''
        Method to save the config file to a specific path

        Parameters
        ----------
        path (str | os.PathLike): Path to save the config file to
        name               (str): Name to give to the new saved config file

        Authors: Allan Denis
        '''

        config = self.config
        if path is None:
            config.filename = self.paths.config_file_path
        else:
            config.filename = Path(path).expanduser() / name

        config.write()

