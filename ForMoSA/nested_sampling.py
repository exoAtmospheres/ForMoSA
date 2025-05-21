import astropy.constants as cst
from ForMoSA.ModelGrid import ModelGrid
from ForMoSA.observation import Observation
from ForMoSA.nested_sampling_plotting import NestedSampling_Plotting
from pathlib import Path
import numpy as np
import ForMoSA.utils_spec as us
import ForMoSA.utils_hc as high_contrast
import ForMoSA.utils_logL_functions as logL_functions
import ForMoSA.utils_prior_functions as prior_functions
import os
import time
import nestle
import pickle
import pymultinest
import ultranest
from ultranest import integrator


class ForMoSAError(Exception):
    pass


class Parameter(object):
    '''
    ForMoSA Parameter class. Handles a single parameter for the nested sampling algorithm
    
    Parameters
    ----------
    name              (str): Name of the parameter ('par1', 'par2', 'rv', 'd', ...)
    prior             (str): Prior function of the parameter ('uniform', 'gaussian', 'constant', 'log-uniform', 'computed')
    bounds           (list): Bounds of the prior (for the 'uniform' and 'log-uniform' priors)
    mean            (float): Mean of the prior (for the 'gaussian' prior)
    std             (float): Standard deviation of the prior (for the 'gaussian' prior)
    value           (float): Value of the prior (for the 'constant' prior)
    vsini_function    (str): vsini function used for the prior (only is name starts with vsini)
    
    Authors: Allan Denis
    '''


    def __init__(self, name: str, prior: str, bounds: list = None, mean: float = None, std: float = None, value: float = None, vsini_function: str = None):
        valid_priors = {'uniform', 'gaussian', 'log-uniform', 'constant', 'computed'}
        if prior not in valid_priors:
            msg = f" Prior '{prior}' not valid for the parameter '{name}'. Choose amongst : {', '.join(valid_priors)}."
            raise ForMoSAError(msg)
        
        if prior in {'uniform', 'log-uniform'}:
            if bounds is None:
                msg = f" Prior '{prior}' needs bounds [min, max]."
                raise ForMoSAError(msg)
            if not (isinstance(bounds, list) and len(bounds) == 2):
                msg = " Bounds have to be a list [min, max] with uniform or log uniform priors."
                raise ForMoSAError(msg)
            if (prior == 'log-uniform') and (bounds[0] <= 0 or bounds[1] <= 0):
                msg = " You cannot use negative bounds with log-uniform priors."
                raise ForMoSAError(msg)

        if prior == 'gaussian':
            if mean is None or std is None:
                msg = " Gaussian prior needs argument 'mean' and 'std'."
                raise ForMoSAError(msg)
            if std <= 0:
                msg = " You cannot use negative standard deviations with Gaussian priors."
                raise ForMoSAError(msg)
                
        if prior == 'constant':
            if value is None:
                msg = " Constant prior needs argument 'value'."
                raise ForMoSAError(msg)
                
        if name.startswith('vsini') and vsini_function == None:
            msg = " 'vsini' parameter needs a vsini function."
            raise ForMoSAError(msg)
                   
        self._name = name
        self._prior = prior
        self._bounds = bounds
        self._mean = mean
        self._std = std
        self._value = value
        self._vsini_function = vsini_function
        self._theta = None
        
        
    ##################################################
    # Representation
    ##################################################
        
    def __repr__(self):
        return f"Parameter(name={self.name}, prior={self.prior}, bounds={self.bounds}, " \
               f"mean={self.mean}, std={self.std}, value={self.value}, vsini_function={self.vsini_function}"
               

    ##################################################
    # Properties
    ##################################################
    
    @property 
    def name(self):          # Name
        return self._name 
    
    @property 
    def prior(self):         # Prior type
        return self._prior 
    
    @property 
    def bounds(self):        # Bounds (for Uniform and log-Uniform priors)
        return self._bounds 
    
    @property
    def mean(self):          # Mean (for Gaussian priors)
        return self._mean 
    
    @property 
    def std(self):           # Standard deviation (for Gaussian priors)
        return self._std 
    
    @property 
    def value(self):         # Value (for Constant prior)
        return self._value 
    
    @property 
    def vsini_function(self):
        return self._vsini_function
    
    @property 
    def is_fixed(self):      # Whether the parameter is fixed
        return self.prior == 'constant'
    
    @property 
    def theta(self):         # Current value randomly picked by the nested sampling
        return self._theta
    
    
    ##################################################
    # Methods
    ##################################################
    
    def _apply_prior(self, theta: float) -> float:
        '''
        Method to apply prior to a parameter given a random value theta picked uniformely between [0, 1]

        Parameters
        ----------
        theta (float): Parameter value randomly picked by the nested sampling
        
        Returns:
            - float: Transformed prior value

        Authors: Allan Denis
        '''
        
        if self.prior == 'uniform':
            self._theta = prior_functions.uniform_prior(self.bounds, theta)
            
        if self.prior == 'gaussian':
            self._theta = prior_functions.gaussian_prior(self.mean, self.std, theta)
            
        if self.prior == 'log-uniform':
            self._theta = prior_functions.loguniform_prior(self.bounds, theta)
            
        return self.theta
    

class NestedSampling_Params(object):
    '''
    ForMoSA NestedSampling_Params class. Handles dynamically the parameters of the nested sampling.
    
    Parameters
    ----------
    logger : logger
    
    Authors: Allan Denis
    '''
    
    def __init__(self, logger):
        self._parameters = {}
        self._logger = logger
        

    ##################################################
    # Representation
    ##################################################

    def __repr__(self):
        return f"NestedSampling_Params({self._parameters})"
    
    
    
    ##################################################
    # Properties
    ##################################################
    
    @property
    def parameters(self) -> dict:
        return self._get_parameters('all')
    
    @property
    def free_parameters(self) -> dict:
        return self._get_parameters('free')
    
    @property
    def fixed_parameters(self) -> dict:
        return self._get_parameters('fixed')
    
    @property 
    def list_params_keys(self) -> list:
        return list(self.parameters)
    
    @property 
    def list_free_params_keys(self) -> list:
        return list(self.free_parameters)
        
    @property 
    def list_params_names(self) -> list:
        return [p.name for p in self.parameters.values()]
    
    @property
    def list_free_params_names(self) -> list:
        return [p.name for p in self.free_parameters.values()]
    
    @property 
    def list_fixed_params_names(self) -> list:
        return [p.name for p in self.fixed_parameters.values()]
    
    @property
    def n_parameters(self) -> int:
        return len(self.parameters)
    
    @property
    def n_free_parameters(self) -> int:
        return len(self.free_parameters)
    
    @property
    def n_fixed_parameters(self) -> int:
        return len(self.fixed_parameters)
    
    @property
    def free_param_priors(self):
        return self._get_param_attr('prior', 'free')
    
    @property
    def free_param_bounds(self):
        return self._get_param_attr('bounds', 'free')
    
    @property 
    def theta(self):
        return self._theta

    
    
    ##################################################
    # Methods
    ##################################################
    
    
    def _add_parameter(self, param: Parameter, name) -> None:
        '''
        Method to add a parameter used in the nested sampling algorithm

        Parameters
        ----------
        param : Instance of :class:`~ForMoSA.Parameter`
        name  : Name of the parameter

        Authors: Allan Denis
        '''
        
        if name in self._parameters:
            msg = f" Parameter '{name}' already exists."
            self._logger.error(msg)
            raise ForMoSAError(msg)
    
        self._parameters[name] = Parameter(name, param.prior, param.bounds, param.mean, param.std, param.value, param.vsini_function)
    
    
    def _check_params(self) -> None:
        '''
        Method to check that there is no issue in the nested sampling parameters
        
        Authors: Allan Denis
        '''
         
        nb_vsini, nb_ld, nb_r, nb_d, nb_av, nb_bb_t, nb_bb_r = 0, 0, 0, 0, 0, 0, 0
         
        for key in self.parameters.keys():
            if key.startswith('vsini'):
                nb_vsini += 1
            if key.startswith('ls'):
                nb_ld += 1
            if key.startswith('r'):
                nb_r += 1
            if key.startswith('d'):
                nb_d += 1
            if key.startswith('av'):
                nb_av += 1
            if key.startswith('bb_R'):
                nb_bb_r += 1
            if key.startswith('bb_T'):
                nb_bb_t += 1
         
        if (nb_vsini != nb_ld):
            msg = " You need to define both vsini and limb darkening priors of set them both to 'NA'."
            self._logger.error(msg)
            raise ForMoSAError(msg)
        if (nb_r != nb_d):
            msg = " You need to define both radius and distance priors or set them both to 'NA'."
            self._logger.error(msg)
            raise ForMoSAError(msg)
        if nb_vsini > 1:
            self._logger.warning(' Multiples vsini priors are defined for different observations, which is very unlikely.')
        if nb_r > 1:
            msg = ' Multiples radius and distance priors are defined for different observations. Please use at most 1 prior for these parameters.'
            self._logger.error(msg)
            raise ForMoSAError(msg)
        if nb_av > 1:
            msg = ' Multiples interstellar extinction priors are defined for different observations. Please use at most 1 prior for this parameter.'
            self._logger.error(msg)
            raise ForMoSAError(msg)
        if nb_bb_t != nb_bb_r:
            msg = " You need to define both black body temperature and black body radius or set them both to 'NA'."
            self._logger.error(msg)
            raise ForMoSAError(msg)
        if nb_bb_t != nb_d and nb_bb_t != 0:
            msg = " Iy tou define a blackbody, you also need to define a distance."
            self._logger.error(msg)
            raise ForMoSAError(msg)
            
            
    def _get_param_value(self, name: str, theta: np.ndarray) -> float:
        '''
        Method to get the value of a parameter given its name.
    
        Parameters
        ----------
        name         (str): Name of the parameter (either a key like 'par1', 'rv', 'vsini_0' or a physical name like 'Teff', 'logg')
        theta (np.ndarray): List of parameter values corresponding to list_params_keys
    
        Returns:
            float: Current value of the parameter, either from theta or the constant prior value
    
        Authors: Allan Denis
        '''
        
       
        if not isinstance(theta, (list, np.ndarray)) or len(theta) == 0:
            raise ForMoSAError(f"theta must be a non-empty array, got: {theta}")
    
        theta = np.asarray(theta)
    
        # Determine parameter key
        if name in self.list_params_keys:
            param_key = name
            theta_index = self.list_free_params_keys
        elif name in self.list_params_names:
            param_key = self.list_params_keys[self.list_params_names.index(name)]
            theta_index = self.list_free_params_names
        else:
            msg = f"Invalid parameter name: {name}. Choose from {self.list_params_keys} or {self.list_params_names}"
            self._logger.error(msg)
            raise ForMoSAError(msg)
    
        # Get parameter
        p = self.parameters[param_key]
   
        # Handle constant prior
        if p.prior == 'constant':
            if not isinstance(p.value, (int, float)) or p.value is None:
                msg = f"Parameter {name} has invalid constant value: {p.value}"
                self._logger.error(msg)
                raise ForMoSAError(msg)
            value = float(p.value)
        else:
            try:
                idx = theta_index.index(name)
                value = theta[idx]
            except ValueError:
                msg = f"Parameter {name} not found in theta_index {theta_index}"
                self._logger.error(msg)
                raise ForMoSAError(msg)
    
        # Update parameter state
        p._theta = value
        return value
 
            
    def _get_parameters(self, param_type: str = 'all') -> dict:
        '''
        Returns all the parameters according to their type : 'all', 'free', 'fixed'.
    
        Parameters
        ----------
        param_type (str): Type of parameter to return ('all', 'fixed', 'free')
    
        Returns:
            - dict: Dictionary of parameters filtered.
        
        Authors: Allan Denis
        '''
        
        if param_type not in ['all', 'free', 'fixed']:
            raise ForMoSAError("param_type must be 'all', 'free', or 'fixed'")
    
        if param_type == 'free':
            return {k: p for k, p in self._parameters.items() if not p.is_fixed}
        elif param_type == 'fixed':
            return {k: p for k, p in self._parameters.items() if p.is_fixed}
        
        return self._parameters
    
    
    def _get_param_attr(self, attr: str, param_type: str = 'all') -> dict:
        '''
        Extract a given attribute for each parameter of a given type
    
        Parameters
        ----------
        attr        (str): Name of the attribute to extract
        param_type  (str): Type of parameter ('all', 'fixed', 'free')
    
        Returns:
            - dict: {param_name: attribute_value}
        
        Authors: Allan Denis
        '''
        
        return {k: getattr(p, attr) for k, p in self._get_parameters(param_type).items()}


class NestedSampling(object):
    '''
    ForMoSA Nested_Sampling class, which provides easy access to the parameters of the nested sampling algorithm
    
    Parameters
    ----------
    
    logger          (Logger): Logger
    algorithm          (str): Algorithm used for the nested sampling ('nestle', 'ultranest' or 'pymultinest')
    npoints            (int): Number of living points used for the nested sampling
    config_ns_algo    (dict): Dictionary containing the parametes of the different nested sampling algorithm
    
    Authors: Allan Denis 
    '''
    
    def __init__(self, algorithm: str, npoints: int, logger, config_ns_algo: dict, color_out: str='blue'):
        self._logger = logger
        
        valid_algorithms = ['nestle', 'ultranest', 'pymultinest']
        if algorithm not in valid_algorithms:
            msg = f" {algorithm} is not a supported algorithm. Please choose amongst {', '.join(valid_algorithms)}"
            self._logger.error(msg)
            raise ForMoSAError(msg)
            
        self._algorithm = algorithm
        self._npoints = npoints
        self._logger = logger
        self._params = NestedSampling_Params(logger)
        self._plotting = NestedSampling_Plotting(logger)
        self._results = None
        self._modif_data = dict()
        self._best_model = dict()
        self._logL = dict()
        
        try:   # Check if the dictionary contains the keys 'nestle', 'ultranest' and 'pymultinest'
            if algorithm == 'nestle':
                self._ns_params = config_ns_algo['nestle']
            elif algorithm == 'ultranest':
                self._ns_params = config_ns_algo['ultranest']
            elif algorithm == 'pymultinest':
                self._ns_params = config_ns_algo['pymultinest']
        except KeyError as e:
            self._logger.error(e)
            raise ForMoSAError(e)
            

    ##################################################
    # Representation
    ##################################################
    
    def __repr__(self):
        return f'Nested sampling, algorithm = {self.algorithm}, npoints = {self.npoints}'
    
    def __format__(self) -> str:
        return self.__repr__()    
    
    
    ##################################################
    # Properties
    ##################################################
    
       
    @property 
    def algorithm(self) -> str:                       # Algorithm
        return self._algorithm
    
    @property 
    def logL(self) -> dict:                           # logL function
        return self._logL
    
    @property 
    def n_obs(self) -> int:                           # Number of observations
        return self.grid.n_obs
    
    @property 
    def params(self) -> NestedSampling_Params:        # Priors parameters
        return self._params
    
    @property 
    def ns_params(self) -> dict:                      # Nested sampling parameters      
        return self._ns_params
    
    @property 
    def npoints(self) -> int:                         # Nested sampling number of living points
        return self._npoints 
    
    @property  
    def results(self) -> dict:                        # Results
        return self._results
    
    @property
    def param_samples_dict(self) -> dict:             # Samples of each parameter 
        if not hasattr(self, "results") or self.results is None:
            msg = 'No results found. Please run the sampling algorithm first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)
        
        return {
            name: self.results['samples'][:, i]
            for i, name in enumerate(self.params.list_free_params_names)
        }
    
    @property
    def param_best_dict(self) -> dict:               # Best value of each parameter     
        if not hasattr(self, "results") or self.results is None:
            msg = 'No results found. Please run the sampling algorithm first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)
        
        return {
            name: np.average(self.results['samples'][:, i], weights=self.results['weights'])
            for i, name in enumerate(self.params.list_free_params_names)
        }
    
    @property 
    def list_best_params(self) -> list:              # List of best values for each parameter
        return list(self.param_best_dict.values())
    
    @property 
    def best_logL(self) -> float:                    # Averaged value of logL
        if not hasattr(self, "results") or self.results is None:
            msg = 'No results found. Please run the sampling algorithm first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)
        return np.average(self.results['logl'], weights = self.results['weights'])
    
    @property 
    def modif_data(self) -> dict:                    # Modified data 
        if not hasattr(self, "results") or self.results is None:
            msg = 'No results found. Please run the sampling algorithm first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)
        return self._results['modif_data']

    @property 
    def best_model(self) -> dict:                    # Best model
        if not hasattr(self, "results") or self.results is None:
            msg = 'No results found. Please run the sampling algorithm first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)
        return self.results['best_model']
    
    @property 
    def plotting(self) -> NestedSampling_Plotting:   # NestedSampling_Plotting class
        return self._plotting

                
    def run(self, logL_type: list, results_path: str | os.PathLike, Observation: Observation, ModelGrid: ModelGrid, interp_method: str = 'linear', wav_cont: list = ['NA'], res_cont: list = ['NA'], hc_type: list = ['NA'], bounds_lsq: list = ['NA'], emulator: list = ['NA'], for_plot: str = 'no') -> None:
        '''
        Method to run the nested sampling algorithm using the model, observation and nested sampling parameters.
        
        Parameters
        ----------
        logL_type                  (list): Loglikelihood function  (['chi2'], ['chi2_covariance'], ['CCF_Brogi'], ['CCF_Zucker'], ...)
        results_path   (str | os.PathLike): Path of the output
        Observation         (Observation): Inbstance of :class:`~ForMoSA.Observation`
        ModelGrid           (Observation): Inbstance of :class:`~ForMoSA.ModelGrid`
        interp_method               (str): Interpolation method ('linear', 'cubic', 'spline', ...)
        wav_cont                   (list): List of wavelength grid used for the continuum (used for high contrast)
        res_cont                   (list): List of resolution used for the continuum (used for high contrast)
        bounds_lsq                 (list): List of bounds used for the least squares (used for high contrast)
        emulator                   (list): Emulator of the grid ('PCA', 'NMF')
        
        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''
        
        self._logger.info(f' Run Nested Sampling algorithm using {self.npoints} living points and {self.algorithm}.')
        
        valid_logL_type = ['chi2', 'chi2_covariance', 'CCF_Brogi', 'CCF_Zucker', 'CCF_custom', 'chi2_noisescaling', 'chi2_noisescaling_covariance']
        
        for indobs in range(Observation.n_obs):
            if logL_type[indobs] not in valid_logL_type:
                msg = f' Invalid loglikelihood type. Please choose amongst {valid_logL_type}'
                self._logger.critical(msg)
                raise ForMoSAError(msg)
            else:
                self._logL[indobs] = logL_type[indobs]
            
        n_free_parameters = self._params.n_free_parameters
        
        loglike_gp = lambda theta: self._loglike(theta, Observation, ModelGrid, interp_method=interp_method, wav_cont=wav_cont, res_cont=res_cont, bounds_lsq=bounds_lsq, for_plot=for_plot, emulator=emulator)
        prior_transform_gp = lambda theta: self._prior_transform(theta, ModelGrid)
        
        os.makedirs(str(results_path) + f'/{self.algorithm}/', exist_ok=True)
        
        time1 = time.time()
        
        if self.algorithm == 'nestle':
            res = nestle.sample(loglike_gp, prior_transform_gp, n_free_parameters,
                                npoints=self.npoints,
                                **self.ns_params,
                                callback=nestle.print_progress)
            
            
            logz = [res['logz'], res['logzerr']]
            samples = res['samples']
            weights = res['weights']
            logvol = res['logvol']
            logl = res['logl']
     
        if self.algorithm == 'pymultinest':
            res = pymultinest.solve(LogLikelihood=loglike_gp,
                                    Prior=prior_transform_gp,
                                    n_dims=n_free_parameters,
                                    outputfiles_basename=str(results_path) + '/pymultinest/' + 'RAW_',
                                    **self.ns_params)
            
            # Reformat the result file
            with open(str(results_path) + '/pymultinest/' + 'RAW_stats.dat', 'rb') as f:
                line = f.readline().strip().split()
                logz = [float(line[5]), float(line[7])]
                
            sample_multi, logl_multi, logvol_multi = [], [], []
            with open(str(results_path) + '/pymultinest/' + 'RAW_ev.dat', 'rb') as f:
                for line in f:
                    parts = line.strip().split()
                    sample_multi.append([float(p) for p in parts[:-3]])
                    logl_multi.append(float(parts[-3]))
                    logvol_multi.append(float(parts[-2]))
                    
            samples, weights, logl, logvol = [], [], [], []
            with open(str(results_path) + '/pymultinest/' + 'RAW_.txt', 'rb') as f:
                for line in f:
                    parts = line.strip().split()
                    point = [float(p) for p in parts[2:]]
                    if point in sample_multi:
                        idx = sample_multi.index(point)
                        samples.append(point)
                        weights.append(float(parts[0]))
                        logl.append(logl_multi[idx])
                        logvol.append(logvol_multi[idx])
                            
            samples = np.asarray(samples)
            weights = np.asarray(weights)
            logl = np.asarray(logl)
            logvol = np.asarray(logvol)
            logz = np.asarray(logz)
    
        if self.algorithm == 'ultranest':
            sampler = ultranest.ReactiveNestedSampler(param_names=self.params.list_free_params_keys,
                                    loglike=loglike_gp,
                                    transform=prior_transform_gp,
                                    log_dir=str(results_path) + '/ultranest/',
                                    **self.ns_params)
            
            sampler.run(min_num_live_points=self.npoints, **self.ns_params)
            
            res = integrator.read_file(str(results_path) + '/ultranest/',
                                    x_dim=self.params.n_free_parameters,
                                    num_bootstraps=self.ns_params['num_bootstraps'])
            
            logz = [res[-1]['logz'], res[-1]['logzerr']]
            samples = res[-1]['samples']
            weights = res[-1]['weighted_samples']['weights']
            logvol = res[0]['logvol']  # Not always used in UltraNest
            logl = res[0]['logl']
            
        
        self._results = {"samples": samples,
                         "weights": weights,
                         "logl": logl,
                         "logvol": logvol,
                         "logz": logz}
        
        # Best model
        modif_data, best_model = self._compute_model_from_theta(self.list_best_params, Observation, ModelGrid, interp_method=interp_method, wav_cont=wav_cont, res_cont=res_cont, bounds_lsq=bounds_lsq)

        self._results["modif_data"] = modif_data
        self._results["best_model"] = best_model

        # # number of models to keep
        # N_keep = 1000  
        
        # samples = self._results["samples"]
        # weights = self._results["weights"]
        # weights /= np.sum(weights)
        
        # indices = np.random.choice(len(samples), size=N_keep, replace=False, p=weights)
        # subset_samples = samples[indices]
        
        # # Computation of models
        # subset_models = []
        
        # for theta in subset_samples:
        #     _, model_i = self._compute_model_from_theta(theta, Observation, ModelGrid, interp_method=interp_method, wav_cont=wav_cont, res_cont=res_cont, bounds_lsq=bounds_lsq, target_resolution=Observation.max_resolution)
        #     subset_models.append(model_i)
        
        # # Sauvegarde dans les résultats
        # self._results["subset_samples"] = subset_samples
        # self._results["subset_models"] = subset_models

        
        time_elapsed = time.time() - time1
        if time_elapsed < 60:
            time_spent = f'{time_elapsed:.1f} sec'
        elif time_elapsed < 3600:
            time_spent = f'{time_elapsed/60:.1f} min'
        else:
            time_spent = f'{time_elapsed/3600:.1f} hours'
            
        self._logger.info(f' Time spent: {time_spent}')
        
        
    def _loglike(self, theta: list, Observation: Observation, ModelGrid: ModelGrid, wav_cont: list = ['NA'], res_cont: list = ['NA'], bounds_lsq: list = ['NA'], interp_method: str = 'linear', for_plot='no', emulator: list = ['NA']) -> float | tuple[dict, np.ndarray, np.ndarray]:
        '''
        Compute the loglikelihood for given values of the parameters of the nested sampling
        If 'for_plot' is 'yes', it returns intermediate results.
        
        Parameters
        ---------- 
        theta                      (list): Parameters values picked by the nested sampling
        Observation         (Observation): Inbstance of :class:`~ForMoSA.Observation`
        ModelGrid           (Observation): Inbstance of :class:`~ForMoSA.ModelGrid`
        wav_cont                   (list): List of wavelength grid used for the continuum (used for high contrast)
        res_cont                   (list): List of resolution used for the continuum (used for high contrast)
        bounds_lsq                 (list): List of bounds used for the least squares (used for high contrast)
        for_plot                    (str): When this function is called from the plotting functions module, we use 'yes'
        emulator                   (list): Emulator of the grid ('PCA', 'NMF')
        
        Returns:
            - FINAL_logL     (float): Final evaluated loglikelihood for both spectra and photometry.
        
        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''
        
        if emulator[0] == 'PCA':
            # TODO
            pass

        try:
            modif_data, modif_model = self._compute_model_from_theta(theta, Observation, ModelGrid, interp_method, wav_cont, res_cont, bounds_lsq)

            FINAL_logL = self._compute_loglike_from_model_and_spectra(modif_data, modif_model)
        except ForMoSAError as e:
            self._logger.error(f"Error computing loglikelihood: {e}")
            raise ForMoSAError(e)
        
        return FINAL_logL

 
    def _compute_model_from_theta(self, theta: list, Observation: Observation, ModelGrid: ModelGrid, interp_method: str = 'linear', wav_cont: list = ['NA'], res_cont: list = ['NA'], bounds_lsq: list = ['NA', 'NA'], hc_type: list = ['NA']) -> tuple[dict, dict]:
        '''
        Method to modify the interpolated synthetic spectra with the different extra-grid parameters.
        It can perform : Re-calibration on the data, Doppler shifting, Application of a substellar extinction, Application of a rotational velocity,
        Application of a circumplanetary disk (CPD).

        Parameters
        ----------
        theta                         (list): Parameter values randomly picked by the nested sampling
        Observation            (Observation): Instance of :class:'~ForMoSA.Observation'
        ModelGrid                (ModelGrid): Instance of :class:'~ForMoSA.ModelGrid'
        interp_method                  (str): Method for the interpolation of the grid
        wav_cont                      (list): Wavelength grid for the continuum estimation of the model (used for high contrast)
        res_cont                      (list): Resolution of the continuum (used for high contrast)
        bounds_lsq                    (list): Bounds of the least squares estumatiion (used for high contrast)
        hc_type                        list): High-contrast function
        indobs                         (int): Index of the current observation looping
            
        Returns:
            - modif_spectra (dict): Dictionary of modified spectra {indobs: {'spectro': dict, 'photo': dict}}
            - modif_model   (dict): Dictionary of modified model {indobs: {'spectro': dict, 'photo: dict'}}
                        
        Author: Simon Petrus, Paulina Palma-Bifani, Allan Denis and Matthieu Ravet
        '''
        
        modif_data, modif_model = dict(), dict()
        
        if wav_cont == ['NA']:
            wav_cont = ['NA'] * Observation.n_obs
        if res_cont == ['NA']:
            res_cont= ['NA'] * Observation.n_obs
        if hc_type == ['NA']:
            hc_type = ['NA'] * Observation.n_obs
        if bounds_lsq == ['NA', 'NA']:
            bounds_lsq = ['NA'] * 2 * Observation.n_obs
        
        def get_param(name, indobs):
            name = name if name in self.params.parameters else f"{name}_{indobs}" if f"{name}_{indobs}" in self.params.parameters else None   # treat also the multi-observations parameters
            if name is None:
                return None
            new_theta = self.params._get_param_value(name, theta)
            return new_theta
        
        theta_index = self._params.list_params_keys
        if len(theta) != self._params.n_free_parameters:
            msg = f"theta length ({len(theta)}) does not match expected number of free parameters ({self._params.n_free_parameters})"
            self._logger.critical(msg)
            raise ForMoSAError(msg)
        
        theta_grid = [theta[i] for i, key in enumerate(theta_index) if key.startswith('par')]
        
        
        for indobs in range(Observation.n_obs):
            obs_dict_spectro, obs_dict_photo = Observation.obs_data[indobs]['spectro'], Observation.obs_data[indobs]['photo'] 
            
            wav_mod_spectro, res_mod_obs_spectro = ModelGrid.adapted_grid[indobs]['spectro'].wavelength, ModelGrid.adapted_grid[indobs]['spectro'].resolution
            wav_mod_photo = ModelGrid.adapted_grid[indobs]['photo'].wavelength
            target_wavelength, target_resolution = obs_dict_spectro['wav'], obs_dict_spectro['res']
            
            ins_spectro, ins_photo = ModelGrid.adapted_grid[indobs]['spectro'].instrument, ModelGrid.adapted_grid[indobs]['photo'].instrument
            
            flx_mod_spectro, flx_mod_photo = ModelGrid._interpolate_between_gridpoints(theta_grid, interp_method, indobs)
            flx_mod_spectro, flx_mod_photo = flx_mod_spectro, flx_mod_photo
            
    
            # RV correction
            rv = get_param('rv', indobs)
            if rv is not None:
                wav_mod_spectro, flx_mod_spectro = us.doppler_fct(wav_mod_spectro, flx_mod_spectro, rv)
                
        
            # vsini correction
            vsini = get_param('vsini', indobs)
            ld = get_param('ld', indobs)
            if vsini is not None and ld is not None:
                vsini_function = str(self.params.parameters['vsini'].vsini_function)
                flx_mod_spectro, res_mod_obs_spectro = us.vsini_fct(wav_mod_spectro, flx_mod_spectro, res_mod_obs_spectro, ld, vsini, vsini_function)
                
        
            # Reddening
            av = get_param('av', indobs)
            if av is not None:
                flx_mod_spectro, flx_mod_photo = us.reddening_fct(wav_mod_spectro, obs_dict_photo['wav'], flx_mod_spectro, flx_mod_photo, av)
            
        
            # CPD
            bb_T = get_param('bb_T', indobs)
            bb_R = get_param('bb_R', indobs)
            d = get_param('d', indobs)
            if None not in (bb_T, bb_R, d):
                flx_mod_spectro, flx_mod_photo = us.bb_cpd_fct(wav_mod_spectro, obs_dict_photo['wav'], flx_mod_spectro, flx_mod_photo, d, bb_T, bb_R)
            
        
            # Save native model before resampling
            flx_mod_spectro_nativ = np.copy(flx_mod_spectro)
            if len(wav_mod_spectro) != len(obs_dict_spectro['wav']):
                flx_mod_spectro = us.resolution_decreasing(wav_mod_spectro, flx_mod_spectro, res_mod_obs_spectro, target_wavelength, target_resolution)
            
        
            # High contrast modeling
            if hc_type[indobs] != "NA":
                contributions, flx_mod_spectro = high_contrast.hc_model(hc_type[indobs], wav_cont[indobs], res_cont[indobs], bounds_lsq[2*indobs: 2*indobs + 2], obs_dict_spectro, flx_mod_spectro, indobs=indobs)
            else:
                contributions = 1
        
            # Scaling (ck)
            alpha = get_param('alpha', indobs)
            r = get_param('r', indobs)
            if hc_type[indobs] == "NA" and r is not None and d is not None:
                flx_mod_spectro, flx_mod_photo, ck = us.calc_ck(obs_dict_spectro, obs_dict_photo, flx_mod_spectro, flx_mod_photo, r, d, alpha or 0)
            # Analytical resolution and special case for MOSAIC when you don't fit for R and D for one of the obs but still want to fit it for the others
            elif hc_type[indobs] != 'NA':
                flx_mod_spectro, flx_mod_photo, ck = us.calc_ck(obs_dict_spectro, obs_dict_photo, flx_mod_spectro, flx_mod_photo, 0, 0, alpha=0, analytic='yes')
        
        
            mod_dict_spectro = {'wav': wav_mod_spectro, 'flx': flx_mod_spectro, 'nativ_flx': flx_mod_spectro_nativ, 'res': obs_dict_spectro['res'], 'ins': ins_spectro, 'ck': ck, 'hc_contributions': contributions}
            mod_dict_photo = {'wav': wav_mod_photo, 'flx': flx_mod_photo, 'ins': ins_photo, 'ck': ck}
        
            obs_dict = {'spectro': obs_dict_spectro, 'photo': obs_dict_photo}
            mod_dict = {'spectro': mod_dict_spectro, 'photo': mod_dict_photo}
            
            modif_data[indobs] = obs_dict
            modif_model[indobs] = mod_dict
            
        return modif_data, modif_model
    
    
    def _compute_loglike_from_model_and_spectra(self, obs: dict, model: dict):
        '''
        Method to compute the loglikelihood from the modified observation and model

        Parameters
        ----------
        obs    (dict): Dictionary of observation modified by the nested sampling {indobs: {'spectro': dict, 'photo': dict}}
        model  (dict): Dictionary of model modified by the nested sampling {indobs: {'spectro': dict 'photo': dict}}

        Returns:
            - Final_logL (float): Final loglikelihood value

        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''
        
        FINAL_logL = 0
        for indobs in obs.keys():
            # LogL Photometry
            photo_data = obs[indobs]['photo']
            photo_model = model[indobs]['photo']
            logL_photo = 0 if len(photo_data['wav']) == 0 else logL_functions.logL_chi2(photo_data['flx'] - photo_model['flx'], photo_data['err'])
            
            # LogL Spectroscopy
            spec_data = obs[indobs]['spectro']
            spec_model = model[indobs]['spectro']
            logL_spectro = 0
            if len(spec_data['wav']) > 0:
                residual = spec_data['flx'] - spec_model['flx']
                ll_type = self.logL[indobs]
                logL_dict = {'chi2': lambda: logL_functions.logL_chi2(residual, spec_data['err']),
                             'chi2_covariance': lambda: logL_functions.logL_chi2_covariance(residual, spec_data['inv_cov']),
                             'CCF_Brogi': lambda: logL_functions.logL_CCF_Brogi(spec_data['flx'], spec_model['flx']),
                             'CCF_Zucker': lambda: logL_functions.logL_CCF_Zucker(spec_data['flx'], spec_model['flx']),
                             'CCF_custom': lambda: logL_functions.logL_CCF_custom(spec_data['flx'], spec_model['flx'], spec_data['err']),
                             'chi2_noisescaling': lambda: logL_functions.logL_chi2_noisescaling(residual, spec_data['err']),
                             'chi2_noisescaling_covariance': lambda: logL_functions.logL_chi2_noisescaling_covariance(residual, spec_data['inv_cov'])}
                logL_spectro = logL_dict.get(ll_type, lambda: 0)()
    
            FINAL_logL += logL_photo + logL_spectro
            
            if FINAL_logL < -1e6:
                self._logger.warning(f"[loglike WARNING] Unusually low loglike: {FINAL_logL}")
                for name in self.params.parameters:
                    value = self.params._get_param_value(name, self.params.theta)
                    self._logger.warning(f"[low loglike] {name} = {value}")
    
                self._logger.info(f"LogL_spectro for obs {indobs}: {logL_spectro}")
                self._logger.info(f"LogL_photo for obs {indobs}: {logL_photo}")
                self._logger.info(f"Total LogL after obs {indobs}: {FINAL_logL}")
                
                self._logger.info(f"{max(spec_model['flx'].values)}")
                self._logger.info(f"{max(spec_data['flx'])}")
                self._logger.info(f"{max(residual.values)}")
                
                import matplotlib.pyplot as plt 
                plt.figure()
                plt.plot(spec_data['wav'], spec_data['flx'], label = 'model')
                plt.legend()
                plt.show()
                raise ForMoSAError()

                
        return FINAL_logL
    
     
    def _prior_transform(self, theta: list, ModelGrid: ModelGrid) -> list:
        '''
        Method to define the priors to be used for the inversion.
        We check that the boundaries are consistent with the grid extension.
    
        Parameters
        ----------
        theta               (list): Parameter values randomly picked by the nested sampling
        MModelGrid     (ModelGrid): Instance of :class:'~ModelGrid'
    
        Return:
            - prior   (list): List of parameter values transformed by the prior laws, in the original order
    
        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''
        
        prior = []
        theta_index_free = self.params.free_parameters.keys()
    
        for i, param_name in enumerate(theta_index_free):
            param = self.params.parameters[param_name]
            theta_val = theta[i]
            prior_value = param._apply_prior(theta_val)
    
            if param_name.startswith('par'):
                # Clamp within the grid bounds
                prior_value = max(min(prior_value, ModelGrid.lims_params_grid[param_name][1]), ModelGrid.lims_params_grid[param_name][0])
                param._theta = prior_value
    
            prior.append(prior_value)
            
        self.params._theta = prior    # Update the current drawn values for the parameters
    
        return prior
    
    
    def _save_results(self, results_path: str | os.PathLike) -> None:
        '''
        Method to save the results to the path results_path

        Parameters
        ----------
        results_path (str | os.PathLike): Path to save the results to

        Authors: Allan Denis
        '''
        
        self._logger.info(' Save results')
        
        results_file = Path(results_path)  / f'results_{self.algorithm}.pic'
        
        self._logger.debug(f'< Save to path {results_file}')
        with open(results_file, 'wb') as f:
            pickle.dump(self._results, f)
            
            
    def _load_results(self, results_path: str | os.PathLike) -> None:
        '''
        Method to load the results from the path results_paths

        Parameters
        ----------
        results_path (str | os.PathLike): Path to save the results to

        Authors: Allan Denis
        '''
        
        self._logger.info(' Load results')
        
        results_file = Path(results_path)  / f'results_{self.algorithm}.pic'
        
        if not(results_file.exists()):
            self._logger.error(f' {results_file} does not exist. Please make sure to use an existing result file.')
            
        self._logger.debug(f'< load {results_file}')
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
            
        logz, samples, weights, logvol, logl, modif_data, best_model = [results['logz'][0], results['logz'][1]], results['samples'], results['weights'], results['logvol'], results['logl'], results['modif_data'], results['best_model']
        self._results = {"samples": samples,
                         "weights": weights,
                         "logl": logl,
                         "logvol": logvol,
                         "logz": logz,
                         "modif_data": modif_data,
                         "best_model": best_model}
        
        # Luminosity derivation 
        if 'r' in self.params.list_free_params_keys:
            r_samples = self.param_samples_dict['r']
            Teff_samples = self.param_samples_dict['Teff']
            
            # Stefan-Boltzmann law
            lum = np.log10(4 * np.pi * (r_samples * cst.R_jup.value) ** 2 * Teff_samples ** 4 * cst.sigma_sb.value / cst.L_sun.value)
            lum_param = Parameter(r'log(L/L$\mathrm{_{\odot}}$)', 'computed')
            
            self._results['samples'] = np.hstack([self._results['samples'], lum[:, None]])
            self.params._add_parameter(lum_param, r'log(L/L$\mathrm{_{\odot}}$)')
            
            
        
        
    def _summary(self, sigma: int = 2) -> None:
        '''
        Method to print a summary of the nested sampling results including weighted statistics.
    
        Parameters
        ----------
        sigma     (int): Confidence interval (1 or 2 sigma), default is 1.
        
        Authors: Allan Denis
        '''
        
        if not hasattr(self, '_results') or self._results is None:
            msg = 'No results found. Please run the sampling algorithm first.'
            self._logger.error(msg)
            raise ForMoSAError(msg)
    
        logz, logzerr = self._results['logz']
        samples = np.asarray(self._results['samples'])
        weights = np.asarray(self._results['weights'])
    
        print("\n======== Nested Sampling Summary ========")
        print(f"Algorithm         : {self.algorithm}")
        print(f"LogZ              : {logz:.3f} ± {logzerr:.3f}")
        print(f"Number of samples : {len(samples)}")
        print(f"Number of parameters   : {samples.shape[1] if samples.ndim > 1 else 1}")
    
        # Normalize weights
        if len(weights) != len(samples):
            print("\nWarning: Weights and samples have inconsistent sizes.")
            return
        weights /= np.sum(weights)
    
        # Confidence interval percentiles
        if sigma == 1:
            low_pct, high_pct = 16, 84
        elif sigma == 2:
            low_pct, high_pct = 2, 98
        else:
            raise ValueError("sigma must be 1 or 2.")
    
        print("\nPosterior (weighted):")
        for i in range(samples.shape[1]):
            param_samples = samples[:, i]
    
            # Weighted mean
            mean = np.average(param_samples, weights=weights)
    
            # Weighted percentiles
            sorted_indices = np.argsort(param_samples)
            sorted_samples = param_samples[sorted_indices]
            sorted_weights = weights[sorted_indices]
            cumsum_weights = np.cumsum(sorted_weights)
    
            def weighted_percentile(p):
                return np.interp(p / 100, cumsum_weights, sorted_samples)
    
            low = weighted_percentile(low_pct)
            high = weighted_percentile(high_pct)
    
            print(f" {self.params.list_free_params_names[i]}: {mean:.4f} [{low:.4f}, {high:.4f}] ({sigma}σ)")
    
        print("=========================================\n")




