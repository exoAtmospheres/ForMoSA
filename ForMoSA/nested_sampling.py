import ForMoSA
from ForMoSA.ModelGrid import ModelGrid
from ForMoSA.observation import Observation
import numpy as np
import ForMoSA.utils_spec as us
import ForMoSA.utils_hc as high_contrast
import ForMoSA.utils_logL_functions as logL_functions
import ForMoSA.utils_prior_functions as prior_functions
import os
import time
import nestle
import pickle

class ForMoSAError(Exception):
    pass


class Parameter(object):
    '''
    ForMoSA Parameter class. Handles a single parameter for the nested sampling algorithm
    '''

    VALID_PRIORS = {'uniform', 'gaussian', 'log-uniform', 'constant'}

    def __init__(self, name: str, prior: str, bounds: list = None, mean: float = None, std: float = None, value: float = None, vsini_function: str = None):
        if prior not in self.VALID_PRIORS:
            raise ForMoSAError(f" Prior '{prior}' not valid for the parameter '{name}'. "
                             f" Choose amongst : {', '.join(self.VALID_PRIORS)}.")
        
        if prior in {'uniform', 'log-uniform'}:
            if bounds is None:
                raise ForMoSAError(f" Prior '{prior}' needs bounds [min, max].")
            if not (isinstance(bounds, list) and len(bounds) == 2):
                raise ForMoSAError(" Bounds have to be a list [min, max] with uniform or log uniform priors.")
            if (prior == 'log-uniform') and (bounds[0] <= 0 or bounds[1] <= 0):
                raise ForMoSAError(" You cannot use negative bounds with log-uniform priors")

        if prior == 'gaussian':
            if mean is None or std is None:
                raise ForMoSAError(" Gaussian prior needs argument 'mean' and 'std'.")
            if std <= 0:
                raise ForMoSAError(" You cannot use negative standard deviations with Gaussian priors")
                
        if prior == 'constant':
            if value is None:
                raise ForMoSAError(" Constant prior needs argument 'value'.")
        
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
        return self.theta
    
    
    ##################################################
    # Methods
    ##################################################
    
    def _apply_prior(self, theta:float):
        '''
        Method to apply prior to a parameter given a random value theta picked uniformely between [0, 1]

        Parameters
        ----------
        theta (float): Parameter value randomly picked by the nested sampling

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
    config_dict (dict): Configuration file dictionary
    
    Authors: Allan Denis
    '''
    
    def __init__(self, config_dict: dict):
        self._parameters = {}
        self._from_config(config_dict)
        

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
            raise ForMoSAError(f" Parameter '{name}' already exists.")
    
        self._parameters[name] = Parameter(name, param.prior, param.bounds, param.mean, param.std, param.value, param.vsini_function)
        self._theta = []


    def _from_config(self, config_dict: dict) -> None:
        '''
        Method to build the dictionnary of parameters from the config file dictionary

        Parameters
        ----------
        config_dict (dict): Configuration file dictionary

        Authors: Allan Denis
        '''
        for param_type in ['grid_parameters', 'physical_parameters']:    # Retrieve 'grid parameters' and 'physical parameters' objects
            for name, param in config_dict[param_type].items():        # (e.g. 'par1', 'rv_0", 'vsini_1')
                self._add_parameter(param, name)
                

    ##################################################
    # Representation
    ##################################################

    def __repr__(self):
        return f"NestedSampling_Params({self._parameters})"
    
    
    ##################################################
    # Properties
    ##################################################

    @property 
    def parameters(self) -> dict:            # Parameters
        return self._parameters

    @property 
    def list_params_keys(self) -> list:     # List of parameters names
        return list(self.parameters)
    
    @property 
    def list_params_names(self) -> list:     # List of parameters names
        return [self.parameters[key].name for key in self.list_params_keys]
    
    @property 
    def list_params_values(self) -> list:    # List of parameters values
        return list(self.parameters.values())
    
    @property 
    def n_free_parameters(self) -> int:      # Number of free parameters
        return sum(1 for p in self.parameters.values() if not p.is_fixed)
    
    @property 
    def n_fixed_parameters(self) -> int:     # Number of fixed parameters
        return len(self.list_params_names) - self.n_free_parameters
    
    @property 
    def free_parameters(self) -> dict:       # Dictionary of free parameters
        return {name: p for name, p in self._parameters.items() if not p.is_fixed}
    
    @property 
    def fixed_parameters(self) -> dict:      # Dictionary of fixed parameters
        return {name: p for name, p in self._parameters.items() if p.is_fixed}
    
    @property 
    def theta(self):                         # List of values randomly picked by the nested sampling
        return self._theta
    
    
    ##################################################
    # Methods
    ##################################################
    
    
    def _check_params(self) -> None:
        '''
        Method to check that there is no issue in the nested sampling parameters
        
        Authors: Allan Denis
        '''
         
        nb_vsini, nb_ld, nb_r, nb_d, nb_av, nb_bb_t, nb_bb_r = 0, 0, 0, 0, 0, 0, 0
         
        for key in self.keys():
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
        if nb_bb_t != nb_d:
            msg = " You need to define both black body and distance or set them both to 'NA'."
            self._logger.error(msg)
            raise ForMoSAError(msg)
            
            
    def _get_param_value(self, name):
        theta_index = self.list_params_keys
        p = self.parameters[name]
        if p.prior == 'constant':
            p._theta = float(p.value)
        else:
            idx = np.where(theta_index == name)[0]
            p._theta = self.theta[idx[0]] if idx.size > 0 else 0



class NestedSampling(object):
    '''
    ForMoSA Nested_Sampling class, which provides easy access to the parameters of the nested sampling algorithm
    
    Parameters
    ----------
    grid          (ModelGrid): Instance of :class:`~ForMoSA.ModelGrid`
    observation (Observation): Instance of :class:`~ForMoSA.Observation`
    
    Authors: Allan Denis 
    '''
    
    def __init__(self, grid: ModelGrid, observation: Observation, logger, params: NestedSampling_Params):
        self._grid = grid
        self._observation = observation
        self._logger = logger
        self._params = params
        self._results = None
        self._logL = dict()
        
        # Replace 'parX' names by associated physical parameters ('Teff', 'logg', ...)
        for name in self.params.parameters.keys():
            if name.startswith('par'):  # Detect grid parameters
                self.params.parameters[name]._name = self.grid.titles[self.grid.keys.index(name)]  # Rename parameter with title associated to 'parX'
                
                
    def run(self, logL_type: list, npoints: int, algorithm: str, nestle_params: dict, ultranest_params: dict, pymultinest_params: dict, result_path: str | os.PathLike, interp_method: str = 'linear', wav_cont: list = ['NA'], res_cont: list = ['NA'], bounds_lsq: list = ['NA'], emulator: list = ['NA'], for_plot: str = 'no') -> None:
        '''
        Run the nested sampling algorithm using the model, observation and nested sampling parameters.
        
        Parameters
        ----------
        logL_type          (list): Loglikelihood function  (['chi2'], ['chi2_covariance'], ['CCF_Brogi'], ['CCF_Zucker'], ...)
        npoints             (int): Number of living points
        algorithm           (str): Algorithm used for the nested sampling ('nestle', 'ultranest', 'pymultinest')
        nestle_params      (dict): Dictionary of nestle parameters
        ultranest_params   (dict): Dictionary of ultranest parameters
        pymultinest_params (dict): Dictionary of pymultinest parameters
        interp_method       (str): Interpolation method ('linear', 'cubic', 'spline', ...)
        wav_cont           (list): List of wavelength grid used for the continuum (used for high contrast)
        res_cont           (list): List of resolution used for the continuum (used for high contrast)
        bounds_lsq         (list): List of bounds used for the least squares (used for high contrast)
        emulator           (list): Emulator of the grid ('PCA', 'NMF')
        
        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''
        
        self._logger.info(f' Run Nested Sampling algorithm using {npoints} living points and {algorithm}.')
        
        valid_logL_type = ['chi2', 'chi2_covariance', 'CCF_Brogi', 'CCF_Zucker', 'CCF_custom', 'chi2_noisescaling', 'chi2_noisescaling_covariance']
        
        for indobs in range(self.observation.n_obs):
            if logL_type[indobs] not in valid_logL_type:
                msg = f' Invalid loglikelihood type. Please choose amongst {valid_logL_type}'
                self._logger.critical(msg)
                raise ForMoSAError(msg)
            else:
                self._logL[indobs] = logL_type[indobs]
            
        n_free_parameters = self._params.n_free_parameters
        
        loglike_gp = lambda theta: self._loglike(theta, logL_type, interp_method=interp_method, wav_cont=wav_cont, res_cont=res_cont, bounds_lsq=bounds_lsq, for_plot=for_plot, emulator=emulator)
        prior_transform_gp = lambda theta: self._prior_transform(theta)
        
        if algorithm == 'nestle':
            os.makedirs(result_path + '/nestle/', exist_ok=True)
            tmpstot1 = time.time()
            loglike_gp = lambda theta: self._loglike(theta, self._logL, wav_cont, res_cont, bounds_lsq, interp_method, for_plot, emulator)
            prior_transform_gp = lambda theta: self._prior_transform(theta)
            result = nestle.sample(
                                   loglike_gp, prior_transform_gp, n_free_parameters,
                                   npoints=npoints,
                                   method=nestle_params['method'],
                                   update_interval=nestle_params['update_interval'],
                                   npdim=nestle_params['npdim'],
                                   maxiter=nestle_params['maxiter'],
                                   maxcall=nestle_params['maxcall'],
                                   dlogz=nestle_params['dlogz'],
                                   decline_factor=nestle_params['decline_factor'],
                                   rstate=nestle_params['rstate'],
                                   callback=nestle.print_progress
                                   )
            # Reformat the result file
            with open(result_path + '/nestle/RAW.pic', 'wb') as f1:
                pickle.dump(result, f1)
            logz = [result['logz'], result['logzerr']]
            samples = result['samples']
            weights = result['weights']
            logvol = result['logvol']
            logl = result['logl']
            tmpstot2 = time.time()-tmpstot1
            if tmpstot2 < 60:
                time_spent = f'{tmpstot2:.1f} sec'
            elif tmpstot2 < 3600:
                time_spent = f'{tmpstot2/60:.1f} min'
            else:
                time_spent = f'{tmpstot2/3600:.1f} hours'
        
        
        
    def _loglike(self, theta, logL_type: dict, wav_cont: list = ['NA'], res_cont: list = ['NA'], bounds_lsq: list = ['NA'], interp_method: str = 'linear', for_plot='no', emulator: list = ['NA']) -> float | tuple[dict, np.ndarray, np.ndarray]:
        '''
        Compute the loglikelihood for given values of the parameters of the nested sampling
        If 'for_plot' is 'yes', it returns intermediate results.
        
        Parameters
        ----------
        theta      (list): Parameters values picked by the nested sampling
        logL_type  (dict): Dictionary of loglikelihood function for each observation
        wav_cont   (list): List of wavelength grid used for the continuum (used for high contrast)
        res_cont   (list): List of resolution used for the continuum (used for high contrast)
        bounds_lsq (list): List of bounds used for the least squares (used for high contrast)
        for_plot    (str): When this function is called from the plotting functions module, we use 'yes'
        emulator   (list): Emulator of the grid ('PCA', 'NMF')
        
        Returns:
            - FINAL_logL     (float): Final evaluated loglikelihood for both spectra and photometry.
        
        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''
  
        FINAL_logL = 0
        self.params._theta = theta
        theta_index = self._params.list_params_keys
        theta_grid = [theta[i] for i, key in enumerate(theta_index) if key.startswith('par')]
    
        for indobs in range(self.observation.n_obs):
            obs_data = self.observation.obs_data[indobs]
            mod_data = self.grid.adapted_grid[indobs]
    
            wav_mod_spectro = mod_data['spectro'].wavelength
            res_mod_spectro = mod_data['spectro'].resolution
    
            w_cont = wav_cont[indobs]
            r_cont = res_cont[indobs]
            bounds = bounds_lsq[2 * indobs: 2 * indobs + 2]
    
            interp_spectro, interp_photo = self.grid._interpolate_between_gridpoints(theta_grid, interp_method, indobs)
    
            if emulator[0] == 'PCA':
                # TODO
                pass
            else:
                flx_mod_spectro, flx_mod_photo = interp_spectro, interp_photo
    
            obs_dict_modif, flx_mod_spec, flx_mod_photo, *_ = self._modif_spec(self.params.parameters, obs_data['spectro'], obs_data['photo'], wav_mod_spectro, res_mod_spectro, flx_mod_spectro, flx_mod_photo, w_cont, r_cont, bounds, indobs=indobs)
    
            # LogL Photometry
            photo_data = obs_dict_modif['photo']
            logL_photo = 0 if not photo_data['wav_photo'] else logL_functions.logL_chi2(
                photo_data['flx'] - flx_mod_photo, photo_data['err']
            )
    
            # LogL Spectroscopy
            spec_data = obs_dict_modif['spectro']
            logL_spectro = 0
            if spec_data['wav']:
                residual = spec_data['flx'] - flx_mod_spec
                ll_type = logL_type[indobs]
                logL_dict = {'chi2': lambda: logL_functions.logL_chi2(residual, spec_data['err']),
                             'chi2_covariance': lambda: logL_functions.logL_chi2_covariance(residual, spec_data['inv_cov']),
                             'CCF_Brogi': lambda: logL_functions.logL_CCF_Brogi(spec_data['flx'], flx_mod_spec),
                             'CCF_Zucker': lambda: logL_functions.logL_CCF_Zucker(spec_data['flx'], flx_mod_spec),
                             'CCF_custom': lambda: logL_functions.logL_CCF_custom(spec_data['flx'], flx_mod_spec, spec_data['err']),
                             'chi2_noisescaling': lambda: logL_functions.logL_chi2_noisescaling(residual, spec_data['err']),
                             'chi2_noisescaling_covariance': lambda: logL_functions.logL_chi2_noisescaling_covariance(residual, spec_data['inv_cov'])}
                logL_spectro = logL_dict.get(ll_type, lambda: 0)()
    
            FINAL_logL += logL_photo + logL_spectro
    
        return FINAL_logL if for_plot == 'no' else (obs_dict_modif, flx_mod_spec, flx_mod_photo)

        

    def _modif_spec(self, obs_dict_spectro: dict, obs_dict_photo: dict, wav_mod_spectro: np.ndarray, res_mod_obs_spectro: np.ndarray, flx_mod_spectro: np.ndarray, flx_mod_photo: np.ndarray, wav_cont: np.ndarray = np.array([]), res_cont: np.ndarray = np.array([]), bounds_lsq: tuple = 'NA', hc_type: str = 'NA', indobs: int=0) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        '''
        Method to modify the interpolated synthetic spectra with the different extra-grid parameters.
        It can perform : Re-calibration on the data, Doppler shifting, Application of a substellar extinction, Application of a rotational velocity,
        Application of a circumplanetary disk (CPD).

        Args:
            theta                         (list): Parameter values randomly picked by the nested sampling
            params                        (dict): Dictionary containing all the prior parameters
            obs_dict_spectro              (dict): Dictionay containing all the observationnal entries (spectroscopy)
            obs_dict_photo                (dict): Dictionay containing all the observationnal entries (photometry)
            flx_mod_spectro         (np.ndarray): New flux of the interpolated synthetic spectrum (spectroscopy)
            flx_mod_photo           (np.ndarray): New flux of the interpolated synthetic spectrum (photometry)
            wav_cont                (np.ndarray): Wavelength grid for the continuum estimation of the model (used for high contrast)
            res_cont                (np.ndarray): Resolution of the continuum (used for high contrast)
            bounds_lsq                   (tuple): Bounds of the least squares estumatiion (used for high contrast)
            hc_type                        (str): High-contrast function
            indobs                         (int): Index of the current observation looping
        Returns:
            - obs_dict                    (dict): New dictionay containing all the observationnal entries (photometry, spectroscopy and/or optional)
            - flx_mod_spectro            (array): New flux of the interpolated synthetic spectrum (spectroscopy)
            - flx_mod_photo              (array): New flux of the interpolated synthetic spectrum (photometry)
            - flx_mod_spectro_nativ      (array): New flux of the interpolated synthetic spectrum NOT RESAMPLED (spectroscopy)
            - contributions              (array): Contributions from the high-contrast model
            - ck                         (float): Scaling factor
     

        Author: Simon Petrus, Paulina Palma-Bifani, Allan Denis and Matthieu Ravet
        '''

        def get_param(name):
            name = name if name in self.params else f"{name}_{indobs}"   # treat also the multi-observations parameters
            self.params._get_param_value(name)
            return self.params.parameters[name]
    
        # RV correction
        rv = get_param('rv')
        if rv is not None:
            wav_mod_spectro, flx_mod_spectro = us.doppler_fct(wav_mod_spectro, flx_mod_spectro, rv)
    
        # vsini correction
        vsini = get_param('vsini')
        ld = get_param('ld')
        if vsini is not None and ld is not None:
            vsini_function = str(self.params.parameters['vsini'].vsini_function)
            flx_mod_spectro, res_mod_obs_spectro = us.vsini_fct(wav_mod_spectro, flx_mod_spectro, res_mod_obs_spectro, ld, vsini, vsini_function)
    
        # Reddening
        av = get_param('av')
        if av is not None:
            flx_mod_spectro, flx_mod_photo = us.reddening_fct(wav_mod_spectro, obs_dict_photo['wav'], flx_mod_spectro, flx_mod_photo, av)
    
        # CPD
        bb_T = get_param('bb_T')
        bb_R = get_param('bb_R')
        d = get_param('d')
        if None not in (bb_T, bb_R, d):
            flx_mod_spectro, flx_mod_photo = us.bb_cpd_fct(wav_mod_spectro, obs_dict_photo['wav'], flx_mod_spectro, flx_mod_photo, d, bb_T, bb_R)
    
        # Save native model before resampling
        flx_mod_spectro_nativ = np.copy(flx_mod_spectro)
        if len(wav_mod_spectro) != len(obs_dict_spectro['wav']):
            flx_mod_spectro = us.resolution_decreasing(wav_mod_spectro, flx_mod_spectro, res_mod_obs_spectro, obs_dict_spectro['wav'], obs_dict_spectro['res'])
    
        # High contrast modeling
        if hc_type != "NA":
            contributions, flx_mod_spectro = high_contrast.hc_model(hc_type, wav_cont, res_cont, bounds_lsq, obs_dict_spectro, flx_mod_spectro)
        else:
            contributions = np.array([])
    
        # Scaling (ck)
        alpha = get_param('alpha')
        r = get_param('r')
        if hc_type != "NA" and r is not None and d is not None:
            flx_mod_spectro, flx_mod_photo, ck = us.calc_ck(obs_dict_spectro, obs_dict_photo, flx_mod_spectro, flx_mod_photo, r, d, alpha or 0)
        # Analytical resolution and special case for MOSAIC when you don't fit for R and D for one of the obs but still want to fit it for the others
        else:
            flx_mod_spectro, flx_mod_photo, ck = us.calc_ck(obs_dict_spectro, obs_dict_photo, flx_mod_spectro, flx_mod_photo, 0, 0, alpha=0, analytic='yes')
    
        obs_dict = {'spectro': obs_dict_spectro, 'photo': obs_dict_photo}
        return obs_dict, flx_mod_spectro, flx_mod_photo, flx_mod_spectro_nativ, contributions, ck

        
     
    def _prior_transform(self, theta: list) -> list:
        '''
        Method to define the priors to be used for the inversion.
        We check that the boundaries are consistent with the grid extension.
    
        Parameters
        ----------
        theta         (list): Parameter values randomly picked by the nested sampling
    
        Return:
            - prior   (list): List of parameter values transformed by the prior laws, in the original order
    
        Authors: Simon Petrus, Matthieu Ravet and Allan Denis
        '''
        
        prior = []
        theta_index = self.params.list_params_keys
    
        for i, param_name in enumerate(theta_index):
            param = self.params.parameters[param_name]
            theta_val = theta[i]
            prior_value = param._apply_prior(theta_val)
    
            if param_name.startswith('par'):
                # Clamp within the grid bounds
                prior_value = max(
                    min(prior_value, self.grid.lims_params_grid[param_name][1]),
                    self.grid.lims_params_grid[param_name][0]
                )
                param._theta = prior_value
    
            prior.append(prior_value)
            
        self.params._theta = prior    # Update the current drawn values for the parameters
    
        return prior


      
    def summary(self):
        return self.results.summary() if self.results else "No results yet."
    
   
    ##################################################
    # Representation
    ##################################################
    
    def __repr__(self):
        return f'Nested sampling'
    
    def __format__(self) -> str:
        return self.__repr__()    
    
   
    ##################################################
    # Properties
    ##################################################
    
    @property 
    def grid(self):                                   # Grid
        return self._grid 
    
    @property 
    def observation(self) -> ForMoSA.observation:     # Observation
        return self._observation
    
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
    def params(self) -> NestedSampling_Params:        # Parameters
        return self._params
    
    

